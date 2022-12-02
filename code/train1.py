# coding:utf-8
# !/usr/bin/python

import gc
import os
import copy
import shutil
import json
import time
import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertModel, RobertaModel, RobertaConfig, BertConfig, BertTokenizer

from models.loss import loss_fun
from tools.finetune_args import args
from utils.functions_utils import swa
from tools.common import init_logger, logger
import sys

sys.path.append('/home/mw/project/code')
from callback.adversarial import FGM, PGD, EMA
from callback.optimizater.lookahead import Lookahead
from pre_model.modeling_nezha import NeZhaModel
from pre_model.configuration_nezha import NeZhaConfig
from utils.data_loader import load_data, EntDataset3
from models.GlobalPointer import GlobalPointer, MetricsCalculator

device = torch.device("cuda:0")

ent2id = json.load(open(args.ent2id_path, encoding="utf-8"))
ent_type_size = len(ent2id)


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_transformer_model(bert_model_path, model_type='bert'):
    tokenizer = BertTokenizer.from_pretrained(bert_model_path, do_lower_case=True)
    encoder, config = None, None
    if model_type == 'bert':
        config = BertConfig.from_pretrained(bert_model_path, num_labels=ent_type_size)
        encoder = BertModel.from_pretrained(bert_model_path, config=config)
    if model_type == 'nezha':
        config = NeZhaConfig.from_pretrained(bert_model_path)
        config.output_hidden_states = True
        encoder = NeZhaModel.from_pretrained(bert_model_path, config=config)
    if model_type == 'roberta':
        config = RobertaConfig.from_pretrained(bert_model_path, num_labels=ent_type_size)
        encoder = RobertaModel.from_pretrained(bert_model_path, config=config)
    return encoder, config, tokenizer


def build_optimizer_and_scheduler(model, t_total, T_mult=1, rewarm_epoch_num=1):
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    # print('ahead')
    # alpha=0.5, k=6
    # optimizer = Lookahead(optimizer, alpha=1, k=5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, t_total // args.epoch * rewarm_epoch_num,
                                            T_mult, eta_min=5e-6, last_epoch=-1)
    return optimizer, scheduler


def save_model(model, global_step):
    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model_to_save = (model.module if hasattr(model, "module") else model)
    print(f'Saving model & optimizer & scheduler checkpoint to {output_dir}')
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))


def save_model_best(model):
    output_dir = os.path.join(args.output_dir_best)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model_to_save = (model.module if hasattr(model, "module") else model)
    print(f'Saving model & optimizer & scheduler checkpoint to {output_dir}')
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))


def evaluate(model, ner_loader_evl, metrics):
    model.eval()

    eval_metric = {}

    total_f1_, total_precision_, total_recall_ = 0., 0., 0.
    for batch in tqdm(ner_loader_evl, desc="Evaluation"):
        raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
        input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(device), \
                                                         segment_ids.to(device), labels.to(device)

        with torch.no_grad():
            logits, _ = model(input_ids, attention_mask, segment_ids)

        f1, p, r = metrics.get_evaluate_fpr(logits, labels)
        total_f1_ += f1
        total_precision_ += p
        total_recall_ += r

    avg_f1 = total_f1_ / (len(ner_loader_evl))
    avg_precision = total_precision_ / (len(ner_loader_evl))
    avg_recall = total_recall_ / (len(ner_loader_evl))

    eval_metric['f1'], eval_metric['precision'], eval_metric['recall'] = avg_f1, avg_precision, avg_recall

    return eval_metric


def kl_ner(logist1, logist2):
    return torch.sum((F.sigmoid(logist1) - F.sigmoid(logist2)) * (logist1 - logist2), dim=[1, 2, 3])


def kl_loss4gp(p, q):
    """
    https://spaces.ac.cn/archives/9039

    f = sigmoid(x)
    KL_divergence = (f(p) - f(q)) * (p - q)
    """
    return ((torch.sigmoid(p) - torch.sigmoid(q)) * (p - q)).mean()

def compute_loss(y_pred, tao=0.05, device="cuda"):
    idxs = torch.arange(0, y_pred.shape[0], device=device)
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12
    similarities = similarities / tao
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def train_kfold_step():
    # train_data and val_data
    datalist = load_data(args.train_file)
    kfold = KFold(n_splits=args.n_splits, random_state=args.seed, shuffle=True)

    # k-fold
    logger.info(f"开始{args.n_splits}折训练")
    output_dir = args.output_dir
    for fold, (train_index, val_index) in enumerate(kfold.split(range(len(datalist)))):
        logger.info(f"第{fold + 1}折")
        args.output_dir = output_dir
        args.output_dir = os.path.join(args.output_dir, f'fold-{fold + 1}')
        os.makedirs(args.output_dir, exist_ok=True)

        # tokenizer
        encoder, config, tokenizer = build_transformer_model(args.bert_model_path, args.model_type)

        train_data = np.array(datalist)[train_index]
        val_data = np.array(datalist)[val_index]
        logger.info(f"训练集个数:{len(train_data)}, 验证集个数:{len(val_data)}")
        ner_train = EntDataset3(train_data, tokenizer=tokenizer)
        ner_loader_train = DataLoader(ner_train, batch_size=args.batch_size, collate_fn=ner_train.collate,
                                      shuffle=True, num_workers=args.num_workers)

        ner_evl = EntDataset3(val_data, tokenizer=tokenizer)
        ner_loader_evl = DataLoader(ner_evl, batch_size=args.batch_size * 8, collate_fn=ner_evl.collate,
                                    shuffle=False, num_workers=0)

        # GP MODEL
        if args.model == 'gp_bert':
            logger.info("GlobalPointerBert")
            model = GlobalPointerBert(encoder, ent_type_size, 64).to(device)
        else:
            logger.info("GlobalPointer")
            model = GlobalPointer(encoder, ent_type_size, 64).to(device)

        swa_raw_model = copy.deepcopy(model)

        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        t_total = len(ner_loader_train) * args.epoch
        optimizer, scheduler = build_optimizer_and_scheduler(model, t_total)
        logger.info("Training/evaluation parameters %s", args)

        if args.use_ema:
            ema = EMA(model.module if hasattr(model, 'module') else model, decay=args.ema_decay)
            ema.register()

        save_steps = t_total // args.epoch
        args.logging_steps = save_steps

        metrics = MetricsCalculator()
        global_steps, total_loss, cur_avg_loss, best_f1 = 0, 0., 0., 0.

        model.train()

        for epoch in range(args.epoch):

            train_iterator = tqdm(ner_loader_train, desc=f'Fold: {fold + 1} Epoch : {epoch + 1}',
                                  total=len(ner_loader_train))

            for batch in tqdm(ner_loader_train):

                model.zero_grad()

                raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
                input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
                    device), segment_ids.to(device), labels.to(device)
                logits, pooled_output = model(input_ids, attention_mask, segment_ids)
                # time1 = time.time()
                loss = loss_fun(logits, labels)

                if epoch == 0:
                    simcse_logits = torch.cat((pooled_output, pooled_output), 1).reshape(-1, pooled_output.shape[-1])
                    simcse_loss = compute_loss(simcse_logits)
                    loss = loss + 1 * simcse_loss

                # rdrop at last 4 epoch
                if epoch > 0:
                    s_logits, _ = model(input_ids, attention_mask, segment_ids)
                    rdrop_loss = kl_loss4gp(logits, s_logits)
                    loss = loss + 1000 * rdrop_loss

                loss.backward()

                if args.use_fgm and epoch % 2 == 0:
                    fgm = FGM(model, emb_name="word_embeddings", epsilon=args.epsilon)
                    fgm.attack()
                    logits, _ = model(input_ids, attention_mask, segment_ids)
                    loss_adv = loss_fun(logits, labels)
                    loss_adv.backward()
                    fgm.restore()

                if args.use_pgd and epoch % 2 != 0:
                    pgd = PGD(model, emb_name="word_embeddings", epsilon=args.epsilon, alpha=args.alpha)
                    pgd.backup_grad()
                    for _t in range(args.adv_k):
                        pgd.attack(is_first_attack=(_t == 0))
                        if _t != args.adv_k - 1:
                            model.zero_grad()
                        else:
                            pgd.restore_grad()
                        logits, _ = model(input_ids, attention_mask, segment_ids)
                        loss_adv = loss_fun(logits, labels)
                        loss_adv.backward()
                    pgd.restore()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()

                if args.use_ema:
                    ema.update()

                scheduler.step()
                optimizer.zero_grad()
                train_iterator.set_postfix_str(f'running training loss: {loss.item():.4f}')
                global_steps += 1

                if global_steps % args.logging_steps == 0:

                    if args.do_eval:

                        logger.info("\n >> Start evaluating ... ... ")

                        if args.use_ema:
                            ema.apply_shadow()

                        metric = evaluate(model, ner_loader_evl, metrics)

                        f1_score = metric['f1']
                        recall = metric['recall']
                        precision = metric['precision']
                        logger.info("Epoch : {}\t Eval f1 : {}\t Precision : {}\t Recall : {}\t ".
                                    format(epoch + 1, f1_score, precision, recall))

                        if args.use_ema:
                            ema.restore()

                        model.train()

                if global_steps % save_steps == 0:
                    save_model(model, global_steps)

        swa(swa_raw_model, args.output_dir, swa_start=args.swa_start)

        torch.cuda.empty_cache()
        gc.collect()

        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'checkpoint-saw-100000/model.pt'),
                                         map_location='cuda:0'))

        metric = evaluate(model, ner_loader_evl, metrics)

        f1_score = metric['f1']
        recall = metric['recall']
        precision = metric['precision']
        file_dirs = os.listdir(args.output_dir)
        for d in file_dirs:
            if 'saw' in d or 'txt' in d:
                continue
            else:
                print(f"remove {d} from {args.output_dir}")
                shutil.rmtree(os.path.join(args.output_dir, d))
        logger.info("\n >> Average: \t Eval f1 : {}\t Precision : {}\t Recall : {}\t ".
                    format(f1_score, recall, precision))

        optimizer.zero_grad()
        del model, optimizer, scheduler, swa_raw_model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"fold-{fold + 1}train done")
    logger.info("train done")


def train_step1():
    encoder, config, tokenizer = build_transformer_model(args.bert_model_path, args.model_type)

    datalist = load_data(args.train_file)

    unlabeled_path = '/home/mw/project/data/pesudo_data/unlabeled_pesudo_10-fold_best-model_30w_0519_1.json'
    unlabeled = load_data(unlabeled_path)

    logger.info(f"未标注伪标签数量{len(unlabeled)}, path: {unlabeled_path}")
    
    datalist = unlabeled + datalist

    logger.info(f"数量{len(datalist)}")
    logger.info(f"训练数量:{len(datalist)}")


    ner_train = EntDataset3(datalist, tokenizer=tokenizer)
    ner_loader_train = DataLoader(ner_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                  collate_fn=ner_train.collate, shuffle=True, pin_memory=True)

    ner_eval = EntDataset3(datalist[-4000:], tokenizer=tokenizer)
    ner_loader_eval = DataLoader(ner_eval, batch_size=args.batch_size * 8, num_workers=args.num_workers,
                                 collate_fn=ner_eval.collate, shuffle=False, pin_memory=True)

    model = GlobalPointer(encoder, ent_type_size, 64).to(device)

    swa_raw_model = copy.deepcopy(model)

    t_total = len(ner_loader_train) * args.epoch
    optimizer, scheduler = build_optimizer_and_scheduler(model, t_total)
    logger.info("Training/evaluation parameters %s", args)

    if args.use_ema:
        ema = EMA(model.module if hasattr(model, 'module') else model, decay=args.ema_decay)
        ema.register()

    save_steps = t_total // args.epoch
    args.logging_steps = save_steps
    metrics = MetricsCalculator()
    global_steps, total_loss, cur_avg_loss, best_f1 = 0, 0., 0., 0.

    model.train()
    for epoch in range(args.epoch):

        train_iterator = tqdm(ner_loader_train, desc=f'Epoch : {epoch + 1}', total=len(ner_loader_train))

        for batch in train_iterator:
            model.zero_grad()

            raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
            input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
                device), segment_ids.to(device), labels.to(device)
            logits, pooled_output = model(input_ids, attention_mask, segment_ids)
            loss = loss_fun(logits, labels)

            if epoch == 0:
                simcse_logits = torch.cat((pooled_output, pooled_output), 1).reshape(-1, pooled_output.shape[-1])
                simcse_loss = compute_loss(simcse_logits)
                loss = loss + 1 * simcse_loss
            # rdrop at last 4 epoch
            if epoch > 0:
                s_logits, _ = model(input_ids, attention_mask, segment_ids)
                rdrop_loss = kl_loss4gp(logits, s_logits)
                loss = loss + 1000 * rdrop_loss

            loss.backward()

            if epoch == 0 or epoch == 2 or epoch == 4 or epoch == 6:
                fgm = FGM(model, emb_name="word_embeddings", epsilon=args.epsilon)
                fgm.attack()
                logits, _ = model(input_ids, attention_mask, segment_ids)
                loss_adv = loss_fun(logits, labels)
                loss_adv.backward()
                fgm.restore()

            if epoch == 1 or epoch == 3 or epoch == 5 or epoch == 7:
                pgd = PGD(model, emb_name="word_embeddings", epsilon=args.epsilon, alpha=args.alpha)
                pgd.backup_grad()
                for _t in range(args.adv_k):
                    pgd.attack(is_first_attack=(_t == 0))
                    if _t != args.adv_k - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    logits, _ = model(input_ids, attention_mask, segment_ids)
                    loss_adv = loss_fun(logits, labels)
                    loss_adv.backward()
                pgd.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

            optimizer.step()

            if args.use_ema:
                ema.update()

            scheduler.step()
            optimizer.zero_grad()

            train_iterator.set_postfix_str(f'running training loss: {loss.item():.4f}')
            global_steps += 1

            if global_steps % args.logging_steps == 0:

                if args.do_eval:

                    logger.info("\n >> Start evaluating ... ... ")

                    if args.use_ema:
                        ema.apply_shadow()

                    metric = evaluate(model, ner_loader_eval, metrics)

                    f1_score = metric['f1']
                    recall = metric['recall']
                    precision = metric['precision']

                    if f1_score > best_f1:
                        best_f1 = f1_score
                        save_model_best(model)

                    logger.info("Epoch : {}\t Eval f1 : {}\t Precision : {}\t Recall : {}\t ".
                                format(epoch + 1, f1_score, precision, recall))

                    if args.use_ema:
                        ema.restore()

                    model.train()

            if global_steps % save_steps == 0:
                save_model(model, global_steps)

    swa(swa_raw_model, args.output_dir, swa_start=args.swa_start)

    torch.cuda.empty_cache()
    gc.collect()

    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'checkpoint-saw-100000/model.pt'),
                                     map_location='cuda:0'))

    metric = evaluate(model, ner_loader_eval, metrics)

    f1_score = metric['f1']
    recall = metric['recall']
    precision = metric['precision']

    logger.info("\n >> Average: \t Eval f1 : {}\t Precision : {}\t Recall : {}\t ".
                format(f1_score, recall, precision))

    optimizer.zero_grad()

    del model, optimizer, scheduler, swa_raw_model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("train done")


def train_step():
    encoder, config, tokenizer = build_transformer_model(args.bert_model_path, args.model_type)

    datalist = load_data(args.train_file)

    # unlabeled_path = '/home/mw/project/data/pesudo_data/unlabeled_pesudo_10-fold_best-model_30w_0519_1.json'
    # unlabeled = load_data(unlabeled_path)

    unlabeled_path2 = '/home/mw/project/data/pesudo_data/unlabeled_pesudo_10-fold_best-model_35w_0519_2.json'
    unlabeled2 = load_data(unlabeled_path2)

    logger.info(f"未标注伪标签数量{len(unlabeled2)}, path: {unlabeled_path2}")
    datalist = unlabeled2 + datalist

    logger.info(f"数量{len(datalist)}")
    logger.info(f"训练数量:{len(datalist)}")


    ner_train = EntDataset3(datalist, tokenizer=tokenizer)
    ner_loader_train = DataLoader(ner_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                  collate_fn=ner_train.collate, shuffle=True, pin_memory=True)

    ner_eval = EntDataset3(datalist[-4000:], tokenizer=tokenizer)
    ner_loader_eval = DataLoader(ner_eval, batch_size=args.batch_size * 8, num_workers=args.num_workers,
                                 collate_fn=ner_eval.collate, shuffle=False, pin_memory=True)

    model = GlobalPointer(encoder, ent_type_size, 64).to(device)

    swa_raw_model = copy.deepcopy(model)

    t_total = len(ner_loader_train) * args.epoch
    optimizer, scheduler = build_optimizer_and_scheduler(model, t_total)
    logger.info("Training/evaluation parameters %s", args)

    if args.use_ema:
        ema = EMA(model.module if hasattr(model, 'module') else model, decay=args.ema_decay)
        ema.register()

    save_steps = t_total // args.epoch
    args.logging_steps = save_steps
    metrics = MetricsCalculator()
    global_steps, total_loss, cur_avg_loss, best_f1 = 0, 0., 0., 0.

    model.train()
    for epoch in range(args.epoch):

        train_iterator = tqdm(ner_loader_train, desc=f'Epoch : {epoch + 1}', total=len(ner_loader_train))

        for batch in train_iterator:
            model.zero_grad()

            raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
            input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
                device), segment_ids.to(device), labels.to(device)
            logits, pooled_output = model(input_ids, attention_mask, segment_ids)
            loss = loss_fun(logits, labels)

            # if epoch == 0:
            #     simcse_logits = torch.cat((pooled_output, pooled_output), 1).reshape(-1, pooled_output.shape[-1])
            #     simcse_loss = compute_loss(simcse_logits)
            #     loss = loss + 1 * simcse_loss
            # # rdrop at last 4 epoch
            # if epoch > 0:
            #     s_logits, _ = model(input_ids, attention_mask, segment_ids)
            #     rdrop_loss = kl_loss4gp(logits, s_logits)
            #     loss = loss + 1000 * rdrop_loss

            loss.backward()

            if epoch == 0 or epoch == 2 or epoch == 4 or epoch == 6:
                fgm = FGM(model, emb_name="word_embeddings", epsilon=args.epsilon)
                fgm.attack()
                logits, _ = model(input_ids, attention_mask, segment_ids)
                loss_adv = loss_fun(logits, labels)
                loss_adv.backward()
                fgm.restore()

            if epoch == 1 or epoch == 3 or epoch == 5 or epoch == 7:
                pgd = PGD(model, emb_name="word_embeddings", epsilon=args.epsilon, alpha=args.alpha)
                pgd.backup_grad()
                for _t in range(args.adv_k):
                    pgd.attack(is_first_attack=(_t == 0))
                    if _t != args.adv_k - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    logits, _ = model(input_ids, attention_mask, segment_ids)
                    loss_adv = loss_fun(logits, labels)
                    loss_adv.backward()
                pgd.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

            optimizer.step()

            if args.use_ema:
                ema.update()

            scheduler.step()
            optimizer.zero_grad()

            train_iterator.set_postfix_str(f'running training loss: {loss.item():.4f}')
            global_steps += 1

            if global_steps % args.logging_steps == 0:

                if args.do_eval:

                    logger.info("\n >> Start evaluating ... ... ")

                    if args.use_ema:
                        ema.apply_shadow()

                    metric = evaluate(model, ner_loader_eval, metrics)

                    f1_score = metric['f1']
                    recall = metric['recall']
                    precision = metric['precision']

                    if f1_score > best_f1:
                        best_f1 = f1_score
                        save_model_best(model)

                    logger.info("Epoch : {}\t Eval f1 : {}\t Precision : {}\t Recall : {}\t ".
                                format(epoch + 1, f1_score, precision, recall))

                    if args.use_ema:
                        ema.restore()

                    model.train()

            if global_steps % save_steps == 0:
                save_model(model, global_steps)

    swa(swa_raw_model, args.output_dir, swa_start=args.swa_start)

    torch.cuda.empty_cache()
    gc.collect()

    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'checkpoint-saw-100000/model.pt'),
                                     map_location='cuda:0'))

    metric = evaluate(model, ner_loader_eval, metrics)

    f1_score = metric['f1']
    recall = metric['recall']
    precision = metric['precision']

    logger.info("\n >> Average: \t Eval f1 : {}\t Precision : {}\t Recall : {}\t ".
                format(f1_score, recall, precision))

    optimizer.zero_grad()

    del model, optimizer, scheduler, swa_raw_model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("train done")


def main():
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_best, exist_ok=True)

    time_ = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    init_logger(log_file=args.output_dir + f'/{args.model_type}-{time_}.txt')

    logger.info("\n >>> Arguments ")
    show_table = PrettyTable(['epoch', 'max_length', 'batch_size',
                              'fgm', 'pgd', 'ema', 'lookahead', 'swa_start',
                              'warmup_ratio', 'weight_decay'])

    show_table.add_row([args.epoch, args.max_length, args.batch_size,
                        args.use_fgm, args.use_pgd, args.use_ema, args.use_lookahead, args.swa_start,
                        args.warmup_ratio, args.weight_decay])
    logger.info(show_table)

    same_seeds(args.seed)
    # train_step()
    if args.kfold:
        train_kfold_step()
    else:
        train_step()


if __name__ == '__main__':
    main()
