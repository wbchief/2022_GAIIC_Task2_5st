# coding:utf-8

"""
Field data pretraining
Only use 100w unlabeled data.

Task data pretraining
Only use 4w labeled data and 1w test data.
"""


import gc
import os
import sys
import json
import time
import random
import warnings
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from typing import Tuple
from dataclasses import dataclass
from argparse import ArgumentParser
from collections import defaultdict
from transformers import BertTokenizer, AdamW,  get_linear_schedule_with_warmup

import torch
from torch import multiprocessing
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.cuda.amp import autocast as autocast, GradScaler

sys.path.append('/home/mw/project/code/pre_model')
from modeling_nezha import NeZhaConfig, NeZhaPreTrainedModel, NeZhaModel, BertOnlyMLMHead
sys.path.append('/home/mw/project/code/utils')
from pre_tools import seed_everything


# sys.path.append('src')
warnings.filterwarnings('ignore')
multiprocessing.set_sharing_strategy('file_system')


def compute_loss(y_pred, tao=0.05, device="cuda"):
    idxs = torch.arange(0, y_pred.shape[0], device=device)
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12
    similarities = similarities / tao
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


class NeZhaForMaskedLM(NeZhaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = NeZhaModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            head_mask=None,
            position_ids=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

            pooled_output = outputs[0][1]
            simcse_logits = torch.cat((pooled_output, pooled_output), 1).reshape(-1, pooled_output.shape[-1])
            simcse_loss = compute_loss(simcse_logits)

            loss = loss + simcse_loss

            outputs = (loss,) + outputs
        return outputs  # (ltr_lm_loss), (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # if model is does not use a causal mask then add a dummy token
        if self.config.is_decoder is False:
            assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1
            )

            dummy_token = torch.full(
                (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
            )
            input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


def read_data(args, tokenizer):

    replace_token = '[unused1]'

    inputs = defaultdict(list)

    train_path = os.path.join(args.data_root_path, 'processed_data', 'train.json')
    test_path = os.path.join(args.data_root_path, 'processed_data', 'test.json')
    unlabeled_path = os.path.join(args.data_root_path, 'processed_data', 'unlabeled.json')

    all_lines = []

    print('>> Reading labeled train data ... ...')
    with open(train_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in tqdm(enumerate(lines), desc='Reading from train data ... ... ', total=len(lines)):
            data = json.loads(line)
            words = data['words']
            text = ''
            for i in range(len(words)):
                if words[i] == '' or words[i] == ' ':
                    text += ' '
                else:
                    text += words[i]
            all_lines.append(text)

    print('>> Reading test data ... ...')
    with open(test_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in tqdm(enumerate(lines), desc='Reading from test data ... ... ', total=len(lines)):
            data = json.loads(line)
            words = data['words']
            text = ''
            for i in range(len(words)):
                if words[i] == '' or words[i] == ' ':
                    text += ' '
                else:
                    text += words[i]
            all_lines.append(text)

    print('>> Reading unlabeled data ... ...')
    with open(unlabeled_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in tqdm(enumerate(lines), desc='Reading from unlabeled data ... ... ', total=len(lines)):
            data = json.loads(line)
            words = data['words']
            text = ''
            for i in range(len(words)):
                if words[i] == '':
                    text += ' '
                else:
                    text += words[i]
            all_lines.append(text)

    for i, text in tqdm(enumerate(all_lines), desc='Processing pretrain data', total=len(all_lines)):

        tokens = []
        for t in text.split():
            tokens += tokenizer.tokenize(t)
            tokens += ['[unused1]']
        tokens = tokens[:-1]

        inputs_dict = tokenizer.encode_plus(tokens, add_special_tokens=True,
                                            return_token_type_ids=True, return_attention_mask=True)

        inputs['input_ids'].append(inputs_dict['input_ids'])
        inputs['token_type_ids'].append(inputs_dict['token_type_ids'])
        inputs['attention_mask'].append(inputs_dict['attention_mask'])

    return inputs


class DGDataset(Dataset):
    def __init__(self, data_dict: dict):
        super(Dataset, self).__init__()
        self.data_dict = data_dict

    def __getitem__(self, index: int) -> tuple:
        data = (
            self.data_dict['input_ids'][index],
            self.data_dict['token_type_ids'][index],
            self.data_dict['attention_mask'][index]
        )

        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class DataCollator:
    def __init__(self, max_seq_len: int, tokenizer: BertTokenizer, mlm_probability=0.15):

        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.special_token_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id}

        self.batch_same_length = True
        self.ngram = False

    def pad_and_truncate(self, input_ids_list, token_type_ids_list, attention_mask_list, max_seq_len):

        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        for i in range(len(input_ids_list)):
            seq_len = len(input_ids_list[i])
            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i], dtype=torch.long)
                token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i], dtype=torch.long)
                attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i], dtype=torch.long)
            else:
                input_ids[i] = torch.tensor(input_ids_list[i][:max_seq_len - 1] + [self.tokenizer.sep_token_id],
                                            dtype=torch.long)
                token_type_ids[i] = torch.tensor(token_type_ids_list[i][:max_seq_len], dtype=torch.long)
                attention_mask[i] = torch.tensor(attention_mask_list[i][:max_seq_len], dtype=torch.long)

        return input_ids, token_type_ids, attention_mask

    def _mlm_mask(self, input_ids, max_seq_len):

        cand_indexes = []

        for (i, id_) in enumerate(input_ids):
            if id_ in self.special_token_ids:
                continue
            cand_indexes.append(i)

        num_to_predict = max(1, int(round(len(input_ids) * self.mlm_probability)))

        np.random.shuffle(cand_indexes)

        covered_indexes = set()
        for ci in cand_indexes:
            if len(covered_indexes) >= num_to_predict:
                break
            covered_indexes.add(ci)

        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_ids))]
        mask_labels += [0] * (max_seq_len - len(mask_labels))
        return torch.tensor(mask_labels[:max_seq_len])

    def _ngram_mask(self, input_ids, max_seq_len):

        cand_indexes = []

        for (i, id_) in enumerate(input_ids):
            if id_ in self.special_token_ids:
                continue
            cand_indexes.append([i])

        num_to_predict = max(1, int(round(len(input_ids) * self.mlm_probability)))

        ngrams = np.arange(1, 4, dtype=np.int64)
        pvals = 1. / np.arange(1, 4)
        pvals /= pvals.sum(keepdims=True)

        # favor_shorter_ngram:
        # pvals = pvals[::-1]

        ngram_indexes = []
        for idx in range(len(cand_indexes)):
            ngram_index = []
            for n in ngrams:
                ngram_index.append(cand_indexes[idx:idx + n])
            ngram_indexes.append(ngram_index)

        np.random.shuffle(ngram_indexes)

        covered_indexes = set()
        for cand_index_set in ngram_indexes:
            if len(covered_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes:
                        continue

            n = np.random.choice(ngrams[:len(cand_index_set)],
                                 p=pvals[:len(cand_index_set)] / pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
            # Repeatedly looking for a candidate that does not exceed the
            # maximum number of predictions by trying shorter ngrams.
            while len(covered_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(covered_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)

        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_ids))]
        mask_labels += [0] * (max_seq_len - len(mask_labels))
        return torch.tensor(mask_labels[:max_seq_len])

    def mlm_mask(self, input_ids_list, max_seq_len):
        mask_labels = []
        for i, input_ids in enumerate(input_ids_list):
            if self.ngram:
                mask_label = self._ngram_mask(input_ids, max_seq_len)
            else:
                mask_label = self._mlm_mask(input_ids, max_seq_len)
            mask_labels.append(mask_label)
        return torch.stack(mask_labels, dim=0)

    @staticmethod
    def find_trigram(input_list, label_list, origin_label_list):
        """

        Args:
            input_list: input ids list
            label_list: labels list (include mask (-100) label)
            origin_label_list: labels list (origin , don't masked)

        Returns:
            shuffled_now_inputs: exchanged input id's place
            shuffled_now_labels: undo input id masked

        """
        # 找到 3 个连续的 label == 100 的 index ， 并返回
        candidate_indexes = []
        for i in range(len(input_list) - 2):
            # skip pad token
            if input_list[i] == 0:
                break
            if label_list[i] == -100 and label_list[i + 1] == -100 and label_list[i + 2] == -100:
                candidate_indexes.append([i, i + 1, i + 2])

        candidate_index = random.sample(candidate_indexes, int(len(candidate_indexes) * 0.05))

        """
        case 1 : not change
        case 2 : a -> b, b -> a
        case 3 : 132, 213, 231, 312, 321, 123
        """

        p = random.random()
        if p <= 0.25:
            for c_i in candidate_index:
                exchange_index_a, exchange_index_b = c_i[1], c_i[2]
                a_element, b_element = input_list[exchange_index_a], input_list[exchange_index_b]
                input_list[exchange_index_a], input_list[exchange_index_b] = b_element, a_element
                a_label, b_label = origin_label_list[exchange_index_a], origin_label_list[exchange_index_b]
                label_list[exchange_index_a], label_list[exchange_index_b] = a_label, b_label
            shuffled_now_inputs, shuffled_now_labels = input_list, label_list

        elif 0.25 < p <= 0.4:
            for c_i in candidate_index:
                exchange_index_a, exchange_index_b = c_i[1], c_i[2]
                a_element, b_element = input_list[exchange_index_a], input_list[exchange_index_b]
                input_list[exchange_index_a], input_list[exchange_index_b] = b_element, a_element
                a_label, b_label = origin_label_list[exchange_index_a], origin_label_list[exchange_index_b]
                label_list[exchange_index_a], label_list[exchange_index_b] = a_label, b_label
            shuffled_now_inputs, shuffled_now_labels = input_list, label_list

        elif 0.4 < p <= 0.65:
            for c_i in candidate_index:
                exchange_index_a, exchange_index_b, exchange_index_c = c_i[0], c_i[1], c_i[2]
                a_element, b_element, c_element = input_list[exchange_index_a], input_list[exchange_index_b], \
                                                  input_list[exchange_index_c]
                input_list[exchange_index_a], input_list[exchange_index_b], input_list[exchange_index_c] = b_element, \
                                                                                                           c_element, \
                                                                                                           a_element
                a_label, b_label, c_label = origin_label_list[exchange_index_a], origin_label_list[exchange_index_b], \
                                            origin_label_list[exchange_index_c]
                label_list[exchange_index_a], label_list[exchange_index_b], label_list[exchange_index_c] = a_label, \
                                                                                                           b_label, \
                                                                                                           c_label
            shuffled_now_inputs, shuffled_now_labels = input_list, label_list

        elif 0.65 < p <= 0.8:
            for c_i in candidate_index:
                exchange_index_a, exchange_index_b = c_i[0], c_i[2]
                a_element, b_element = input_list[exchange_index_a], input_list[exchange_index_b]
                input_list[exchange_index_a], input_list[exchange_index_b] = b_element, a_element
                a_label, b_label = origin_label_list[exchange_index_a], origin_label_list[exchange_index_b]
                label_list[exchange_index_a], label_list[exchange_index_b] = a_label, b_label
            shuffled_now_inputs, shuffled_now_labels = input_list, label_list

        else:
            for c_i in candidate_index:
                exchange_index_a, exchange_index_b, exchange_index_c = c_i[0], c_i[1], c_i[2]
                a_element, b_element, c_element = input_list[exchange_index_a], input_list[exchange_index_b], \
                                                  input_list[exchange_index_c]
                input_list[exchange_index_a], input_list[exchange_index_b], input_list[exchange_index_c] = c_element, \
                                                                                                           a_element, \
                                                                                                           b_element
                a_label, b_label, c_label = origin_label_list[exchange_index_a], origin_label_list[exchange_index_b], \
                                            origin_label_list[exchange_index_c]
                label_list[exchange_index_a], label_list[exchange_index_b], label_list[exchange_index_c] = a_label, \
                                                                                                           b_label, \
                                                                                                           c_label
            shuffled_now_inputs, shuffled_now_labels = input_list, label_list

        return shuffled_now_inputs, shuffled_now_labels

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor]:

        labels = inputs.clone()
        origin_labels = inputs.clone()

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]

        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = probability_matrix.bool()

        labels[~masked_indices] = -100

        batch_size = inputs.shape[0]
        for i in range(batch_size):
            now_inputs = inputs[i]
            now_inputs = now_inputs.cpu().numpy().tolist()
            now_labels = labels[i]
            now_origin_labels = origin_labels[i]
            now_labels = now_labels.cpu().numpy().tolist()
            now_origin_labels = now_origin_labels.cpu().numpy().tolist()
            shuffled_now_inputs, shuffled_now_labels = self.find_trigram(now_inputs, now_labels, now_origin_labels)
            shuffled_now_inputs = torch.tensor(shuffled_now_inputs)
            shuffled_now_labels = torch.tensor(shuffled_now_labels)
            inputs[i] = shuffled_now_inputs
            labels[i] = shuffled_now_labels

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(
            torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), size=labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, attention_mask_list = list(zip(*examples))

        if self.batch_same_length:
            cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
            max_seq_len = min(cur_max_seq_len, self.max_seq_len)
        else:
            max_seq_len = self.max_seq_len

        input_ids, token_type_ids, attention_mask = self.pad_and_truncate(input_ids_list,
                                                                          token_type_ids_list,
                                                                          attention_mask_list,
                                                                          max_seq_len)

        batch_mask = self.mlm_mask(input_ids_list, max_seq_len)
        input_ids, mlm_labels = self.mask_tokens(input_ids, batch_mask)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': mlm_labels
        }

        return data_dict


def load_data(args, tokenizer):

    train_data = read_data(args, tokenizer)
    collate_fn = DataCollator(args.max_seq_len, tokenizer)
    train_dataset = DGDataset(train_data)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    return train_dataloader


@dataclass
class LabelSmoother:
    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, logits, labels):
        log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)
        padding_mask = labels.eq(self.ignore_index)
        labels.clamp_min_(0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True)
        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


def build_model_and_tokenizer(args, load_path):
    tokenizer = BertTokenizer.from_pretrained(load_path)
    model_config = NeZhaConfig.from_pretrained(load_path)
    model = NeZhaForMaskedLM.from_pretrained(pretrained_model_name_or_path=load_path, config=model_config)
    model.to(args.device)

    return tokenizer, model


def build_optimizer(args, model, train_steps):
    no_decay = ['bias', 'LayerNorm.weight']

    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.eps)

    total_step = train_steps
    warmup_steps = total_step * args.warmup_ratio

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_step)

    return optimizer, scheduler


def batch2cuda(args, batch):
    return {item: value.to(args.device) for item, value in list(batch.items())}


def create_dirs(path):
    os.makedirs(path, exist_ok=True)


def save_model(args, model, tokenizer, global_steps, is_last=False):
    if isinstance(model, torch.optim.swa_utils.AveragedModel):
        model = model.module
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model_to_save = model.module if hasattr(model, 'module') else model
    if is_last:
        model_save_path = os.path.join(args.model_save_path, f'checkpoint-{global_steps}')
    else:
        model_save_path = os.path.join(args.model_record_save_path, f'checkpoint-{global_steps}')
    model_to_save.save_pretrained(model_save_path)
    tokenizer.save_vocabulary(model_save_path)

    print(f'\n>> model saved in : {model_save_path} .')


def pretrain(args):

    print('\n>> start pretraining ... ...')
    load_path = args.load_pretrain_model_path
    print(f'\n>> loading from pretrain model path -> {load_path}')

    tokenizer, model = build_model_and_tokenizer(args, load_path)

    if args.n_gpus > 1:
        model = torch.nn.DataParallel(model)

    train_dataloader = load_data(args, tokenizer)

    total_steps = args.num_epochs * len(train_dataloader)

    optimizer, scheduler = build_optimizer(args, model, total_steps)

    if args.swa:
        swa_model = AveragedModel(model)
        swa_model.to(args.device)
        swa_model_valid = False
        swa_start_step = int(total_steps * 0.85)
        swa_steps = total_steps - swa_start_step

    if args.use_label_smooth:
        label_smooth_loss_fct = LabelSmoother(epsilon=args.label_smooth_rate)

    global_steps = 0

    if args.fp16:
        scaler = GradScaler()

    total_loss, cur_avg_loss = 0., 0.
    for epoch in range(1, args.num_epochs + 1):

        train_iterator = tqdm(train_dataloader, desc=f'Epoch : {epoch}', total=len(train_dataloader))

        model.train()

        for step, batch in enumerate(train_iterator):

            model.zero_grad()

            batch_cuda = batch2cuda(args, batch)

            if args.fp16:
                with autocast():
                    loss, logits = model(**batch_cuda)[:2]
            else:
                loss, logits = model(**batch_cuda)[:2]

            if args.n_gpus > 1:
                loss = loss.mean()

            if args.use_label_smooth:
                labels = batch_cuda['labels']
                label_smooth_loss = label_smooth_loss_fct(logits, labels)
                if args.fp16:
                    scaler.scale(label_smooth_loss).backward()
                else:
                    label_smooth_loss.backward()
            else:
                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            total_loss += loss.item()
            cur_avg_loss += loss.item()

            if args.fp16:
                scaler.unscale_(optimizer)

            if args.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if args.swa:
                if (global_steps + 1) == swa_start_step:
                    print('\n>>> SWA starting ...')
                    lr = args.min_lr // 2
                    swa_scheduler = SWALR(optimizer, swa_lr=lr)

                if (global_steps + 1) > swa_start_step:
                    if (global_steps + 1) % swa_steps == 0:
                        swa_model_valid = True
                        swa_model.update_parameters(model)
                    swa_scheduler.step()
                else:
                    scheduler.step()
            else:
                scheduler.step()

            optimizer.zero_grad()

            if (global_steps + 1) % args.logging_steps == 0:

                epoch_avg_loss = cur_avg_loss / args.logging_steps
                global_avg_loss = total_loss / (global_steps + 1)

                print(f"\n>> epoch - {epoch},  global steps - {global_steps + 1}, "
                      f"epoch avg loss - {epoch_avg_loss:.4f}, global avg loss - {global_avg_loss:.4f}.")

                cur_avg_loss = 0.

            # if (global_steps + 1) % args.save_steps == 0:
            #     if args.swa:
            #         if global_steps >= swa_start_step:
            #             save_model(args, swa_model, tokenizer, global_steps)
            #         else:
            #             save_model(args, model, tokenizer, global_steps)
            #     else:
            #         save_model(args, model, tokenizer, global_steps)

            global_steps += 1

            lr = scheduler.get_last_lr()[0]
            train_iterator.set_postfix_str(f'loss : {loss.item():.4f}, lr : {lr}, global steps : {global_steps} .')

    print('\n>> saving model at last epoch ... ...')
    if args.swa:
        if swa_model_valid:
            save_model(args, swa_model, tokenizer, global_steps, True)
        else:
            save_model(args, model, tokenizer, global_steps, True)
    else:
        save_model(args, model, tokenizer, global_steps, True)

    del model, tokenizer, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = ArgumentParser()
    data_path = '/home/mw/project/data'
    parser.add_argument('--num_workers', type=int, default=9)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--debug', type=bool, default=False)

    parser.add_argument('--data_root_path', type=str,
                        default=data_path)
    parser.add_argument('--data_cache_path', type=str,
                        default=os.path.join(data_path, 'pretrain'))

    parser.add_argument('--model_save_path', type=str,
                        default=os.path.join(data_path, 'pretrain_model_mlm2_new'))
    parser.add_argument('--model_record_save_path', type=str,
                        default=os.path.join(data_path, 'model_record_save'))

    parser.add_argument('--load_pretrain_model_path', type=str,
                        default='/home/mw/input/data9690/pretrain_model/pretrain_model/prev_trained_model/nezha-cn-base')

    parser.add_argument('--fp16', type=str, default=True)

    parser.add_argument('--num_epochs', type=int, default=10)

    parser.add_argument('--max_seq_len', type=int, default=110)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--lr', type=float, default=2.5e-5)

    parser.add_argument('--eps', type=float, default=1e-8)

    parser.add_argument('--use_label_smooth', type=bool, default=False)
    parser.add_argument('--label_smooth_rate', type=float, default=0.001)

    parser.add_argument('--swa', type=bool, default=False)

    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--logging_steps', type=int, default=10000)
    parser.add_argument('--save_steps', type=int, default=50000)

    parser.add_argument('--seed', type=int, default=1998)
    parser.add_argument('--device', type=str, default='cuda')

    warnings.filterwarnings('ignore')
    args = parser.parse_args()

    create_dirs(args.model_save_path)
    create_dirs(args.model_record_save_path)

    seed_everything(args.seed)

    print('\n >> Start pretrain data ... ...')
    start_time = time.time()
    pretrain(args)
    end_time = time.time()
    print(f'\n >> End pretrain, time cost : {round((end_time - start_time) // 60.00, 4)} min .')


if __name__ == '__main__':
    main()
