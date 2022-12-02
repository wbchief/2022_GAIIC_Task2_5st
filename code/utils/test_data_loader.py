# encoding=utf-8

import copy
import json
import logging
import time

import torch
import unicodedata
import os
import sys
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizerFast


logger = logging.getLogger(__name__)
max_len = 110

ent2id = json.load(open('/home/mw/project/data/ent2id.json', encoding="utf-8"))
ent_type_size = len(ent2id)
id2ent = {}
for k, v in ent2id.items():
    id2ent[v] = k


def load_data(path):
    D = []
    for d in json.load(open(path, encoding='utf-8')):
        D.append([d['text']])
        for start, end, label in d['entity_list']:
            if start <= end:
                D[-1].append((start, end, ent2id[label]))
    return D


# import jieba
# from jieba.analyse import *


# 汉字空格问题
class EntDataset(Dataset):
    def __init__(self, data, tokenizer, type='train'):
        self.data = data
        self.tokenizer = tokenizer
        self.type = type
        self.features = self.encoder()

    def __len__(self):
        return len(self.data)

    def _is_control(self, ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    def lowercase_and_normalize(self, text):
        """转小写，并进行简单的标准化
        """

        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
        return text

    def stem(self, token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    def _is_special(self, ch):
        """判断是不是有特殊含义的符号
        """
        special = ['[CLS]', '[SEP]', '[PAD]']
        # special = ['[CLS]', '[SEP]', '[PAD]', '[UNK]']
        if ch in special:
            return True
        else:
            False

    def get_token_mapping(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系"""

        text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            ch = self.lowercase_and_normalize(ch)
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            elif token == '[unused1]' or token == '[UNK]':
                start = offset
                end = offset + 1
                token_mapping.append(char_mapping[start:end])
                offset = end
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping

    def encoder(self):
        features = []
        if self.type != 'test':
            for (ex_index, item) in enumerate(self.data):
                if ex_index % 10000 == 0:
                    logger.info("Writing %s example %d of %d", self.type, ex_index, len(self.data))
                text = item[0]
                tokens = []
                for t in text.split():
                    tokens += self.tokenizer.tokenize(t)
                    tokens += ['[unused1]']
                tokens = tokens[:-1]
                tokens_ = copy.deepcopy(tokens)
                tokens_ = ['[CLS]'] + tokens_ + ['[SEP]']

                mapping = self.get_token_mapping(text, tokens_)
                start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
                end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}

                # 将raw_text的下标 与 token的start和end下标对应
                encoder_txt = self.tokenizer.encode_plus(tokens, max_length=max_len, truncation=True)
                input_ids = encoder_txt["input_ids"]
                token_type_ids = encoder_txt["token_type_ids"]
                attention_mask = encoder_txt["attention_mask"]

                # 获取关键词
                keyword = extract_tags(text, topK=1)
                key_tokens = self.tokenizer.tokenize(keyword[0])

                # 将raw_text的下标 与 token的start和end下标对应
                encoder_key = self.tokenizer.encode_plus(key_tokens, max_length=max_len, truncation=True)
                input_key_ids = encoder_key["input_ids"]
                token_type_key_ids = encoder_key["token_type_ids"]
                attention_key_mask = encoder_key["attention_mask"]

                features.append([text, input_ids, attention_mask, token_type_ids,
                                 start_mapping, end_mapping, item[1:], mapping,
                                 input_key_ids, token_type_key_ids, attention_key_mask])

        else:
            # TODO 测试
            for (ex_index, item) in enumerate(self.data):
                if ex_index % 10000 == 0:
                    logger.info("Writing %s example %d of %d", self.type, ex_index, len(self.data))
                text = item[0]
                # 将raw_text的下标 与 token的start和end下标对应
                tokens = []
                for t in text.split():
                    tokens += self.tokenizer.tokenize(t)
                    tokens += ['[unused1]']
                tokens = tokens[:-1]
                tokens_ = copy.deepcopy(tokens)
                tokens_ = ['[CLS]'] + tokens_ + ['[SEP]']

                mapping = self.get_token_mapping(text, tokens_)

                encoder_txt = self.tokenizer.encode_plus(tokens, max_length=max_len, truncation=True)
                input_ids = encoder_txt["input_ids"]
                token_type_ids = encoder_txt["token_type_ids"]
                attention_mask = encoder_txt["attention_mask"]

                # 获取关键词
                keyword = extract_tags(text, topK=1)
                key_tokens = self.tokenizer.tokenize(keyword[0])

                # 将raw_text的下标 与 token的start和end下标对应
                encoder_key = self.tokenizer.encode_plus(key_tokens, max_length=max_len, truncation=True)
                input_key_ids = encoder_key["input_ids"]
                token_type_key_ids = encoder_key["token_type_ids"]
                attention_key_mask = encoder_key["attention_mask"]
                features.append([text, input_ids, attention_mask, token_type_ids, mapping,
                                 input_key_ids, token_type_key_ids, attention_key_mask])
        return features

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        """Numpy函数，将序列padding到同一长度
        """
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    def collate(self, examples):

        if self.type != 'test':

            raw_text_list, batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids = [], [], [], [], []
            batch_input_key_ids, batch_segment_key_ids, batch_attention_key_mask = [], [], []
            mappings = []
            for item in examples:
                text, input_ids, attention_mask, token_type_ids, start_mapping, end_mapping, label, \
                mapping, input_key_ids, token_type_key_ids, attention_key_mask = item

                raw_text_list.append(text)
                batch_input_ids.append(torch.tensor(input_ids))
                batch_segment_ids.append(torch.tensor(token_type_ids))
                batch_attention_mask.append(torch.tensor(attention_mask))

                batch_input_key_ids.append(torch.tensor(input_key_ids))
                batch_segment_key_ids.append(torch.tensor(token_type_key_ids))
                batch_attention_key_mask.append(torch.tensor(attention_key_mask))

                labels = np.zeros((len(ent2id), max_len, max_len), dtype=np.int)
                for start, end, label in label:
                    if start in start_mapping and end in end_mapping:
                        start = start_mapping[start]
                        end = end_mapping[end]
                        labels[label, start, end] = 1
                batch_labels.append(labels[:, :len(input_ids), :len(input_ids)])
                # mappings.append(mapping)

            batch_input_ids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
            batch_segment_ids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
            batch_attention_mask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()

            batch_input_key_ids = torch.tensor(self.sequence_padding(batch_input_key_ids)).long()
            batch_segment_key_ids = torch.tensor(self.sequence_padding(batch_segment_key_ids)).long()
            batch_attention_key_mask = torch.tensor(self.sequence_padding(batch_attention_key_mask)).float()

            batch_labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=3)).long()

            return raw_text_list, batch_input_ids, batch_attention_mask, batch_segment_ids, \
                   batch_labels, batch_input_key_ids, batch_segment_key_ids, batch_attention_key_mask
        else:
            raw_text_list, batch_input_ids, batch_attention_mask, batch_segment_ids, mappings = [], [], [], [], []
            batch_input_key_ids, batch_segment_key_ids, batch_attention_key_mask = [], [], []
            for item in examples:
                text, input_ids, attention_mask, token_type_ids, mapping, \
                input_key_ids, token_type_key_ids, attention_key_mask = item

                raw_text_list.append(text)
                batch_input_ids.append(input_ids)
                batch_segment_ids.append(token_type_ids)
                batch_attention_mask.append(attention_mask)
                mappings.append(mapping)

                batch_input_key_ids.append(torch.tensor(input_key_ids))
                batch_segment_key_ids.append(torch.tensor(token_type_key_ids))
                batch_attention_key_mask.append(torch.tensor(attention_key_mask))


            batch_input_ids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
            batch_segment_ids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
            batch_attention_mask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()

            batch_input_key_ids = torch.tensor(self.sequence_padding(batch_input_key_ids)).long()
            batch_segment_key_ids = torch.tensor(self.sequence_padding(batch_segment_key_ids)).long()
            batch_attention_key_mask = torch.tensor(self.sequence_padding(batch_attention_key_mask)).float()

            return raw_text_list, batch_input_ids, batch_attention_mask, batch_segment_ids, mappings, \
                   batch_input_key_ids, batch_segment_key_ids, batch_attention_key_mask

    def __getitem__(self, index):
        item = self.features[index]
        return item

# 汉字空格问题
class EntDataset3(Dataset):
    def __init__(self, data, tokenizer, type='train'):
        self.data = data
        self.tokenizer = tokenizer
        self.type = type
        self.features = self.encoder()

    def __len__(self):
        return len(self.data)

    def _is_control(self, ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    def lowercase_and_normalize(self, text):
        """转小写，并进行简单的标准化
        """

        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
        return text

    def stem(self, token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    def _is_special(self, ch):
        """判断是不是有特殊含义的符号
        """
        special = ['[CLS]', '[SEP]', '[PAD]']
        # special = ['[CLS]', '[SEP]', '[PAD]', '[UNK]']
        if ch in special:
            return True
        else:
            False

    def get_token_mapping(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系"""

        text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            ch = self.lowercase_and_normalize(ch)
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            elif token == '[unused1]' or token == '[UNK]':
                start = offset
                end = offset + 1
                token_mapping.append(char_mapping[start:end])
                offset = end
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping

    def encoder(self):
        features = []
        if self.type != 'test':
            for (ex_index, item) in enumerate(self.data):
                if ex_index % 10000 == 0:
                    logger.info("Writing %s example %d of %d", self.type, ex_index, len(self.data))
                text = item[0]
                
                tokens = []
                for t in text.split():
                    tokens += self.tokenizer.tokenize(t)
                    tokens += ['[unused1]']
                    # tokens += ['-']
                tokens = tokens[:-1]
                # text_ = ''.join(tokens)
                tokens_ = copy.deepcopy(tokens)
                tokens_ = ['[CLS]'] + tokens_ + ['[SEP]']
                
                

                mapping = self.get_token_mapping(text, tokens_)
                start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
                end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}

                # 将raw_text的下标 与 token的start和end下标对应
                encoder_txt = self.tokenizer.encode_plus(tokens, max_length=max_len, truncation=True)
                input_ids = encoder_txt["input_ids"]
                token_type_ids = encoder_txt["token_type_ids"]
                attention_mask = encoder_txt["attention_mask"]
                features.append([text, input_ids, attention_mask, token_type_ids, start_mapping, end_mapping, item[1:], mapping])
               
        else:
            # TODO 测试
            for (ex_index, item) in enumerate(self.data):
                if ex_index % 10000 == 0:
                    print("Writing %s example %d of %d", self.type, ex_index, len(self.data))
                text = item[0]
                # 将raw_text的下标 与 token的start和end下标对应
                tokens = []
                for t in text.split():
                    tokens += self.tokenizer.tokenize(t)
                    tokens += ['[unused1]']
                    # tokens += ['-']
                tokens = tokens[:-1]
                tokens_ = copy.deepcopy(tokens)
                tokens_ = ['[CLS]'] + tokens_ + ['[SEP]']

                mapping = self.get_token_mapping(text, tokens_)

                encoder_txt = self.tokenizer.encode_plus(tokens, max_length=max_len, truncation=True)
                input_ids = encoder_txt["input_ids"]
                token_type_ids = encoder_txt["token_type_ids"]
                attention_mask = encoder_txt["attention_mask"]
                features.append((text, input_ids, attention_mask, token_type_ids, mapping))
               
        return features

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        """Numpy函数，将序列padding到同一长度
        """
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    def collate(self, examples):

        if self.type != 'test':

            raw_text_list, batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids = [], [], [], [], []
            #mappings = []
            for item in examples:
                text, input_ids, attention_mask, token_type_ids, start_mapping, end_mapping, label, mapping = item

                raw_text_list.append(text)
                batch_input_ids.append(torch.tensor(input_ids))
                batch_segment_ids.append(torch.tensor(token_type_ids))
                batch_attention_mask.append(torch.tensor(attention_mask))

                labels = np.zeros((len(ent2id), max_len, max_len), dtype=np.int)
                for start, end, label in label:
                    if start in start_mapping and end in end_mapping:
                        start = start_mapping[start]
                        end = end_mapping[end]
                        labels[label, start, end] = 1
                batch_labels.append(labels[:, :len(input_ids), :len(input_ids)])
                #mappings.append(mapping)
            batch_input_ids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
            batch_segment_ids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
            batch_attention_mask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()
            batch_labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=3)).long()

            return raw_text_list, batch_input_ids, batch_attention_mask, batch_segment_ids, batch_labels
        else:
            raw_text_list, batch_input_ids, batch_attention_mask, batch_segment_ids, mappings = [], [], [], [], []
            for item in examples:
                text, input_ids, attention_mask, token_type_ids, mapping = item
                raw_text_list.append(text)
                batch_input_ids.append(input_ids)
                batch_segment_ids.append(token_type_ids)
                batch_attention_mask.append(attention_mask)
                mappings.append(mapping)
            batch_input_ids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
            batch_segment_ids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
            batch_attention_mask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()

            return raw_text_list, batch_input_ids, batch_attention_mask, batch_segment_ids, mappings

    def __getitem__(self, index):
        item = self.features[index]
        return item


# 区分伪标
class EntDataset4(Dataset):
    def __init__(self, data, tokenizer, type='train', pseudo_num=0):
        self.data = data
        self.tokenizer = tokenizer
        self.type = type
        self.pseudo_num = pseudo_num
        self.features = self.encoder()


    def __len__(self):
        return len(self.data)

    def _is_control(self, ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    def lowercase_and_normalize(self, text):
        """转小写，并进行简单的标准化
        """

        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
        return text

    def stem(self, token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    def _is_special(self, ch):
        """判断是不是有特殊含义的符号
        """
        special = ['[CLS]', '[SEP]', '[PAD]']
        # special = ['[CLS]', '[SEP]', '[PAD]', '[UNK]']
        if ch in special:
            return True
        else:
            False

    def get_token_mapping(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系"""

        text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            ch = self.lowercase_and_normalize(ch)
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            elif token == '[unused1]' or token == '[UNK]':
                start = offset
                end = offset + 1
                token_mapping.append(char_mapping[start:end])
                offset = end
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping

    def encoder(self):
        features = []
        if self.type != 'test':
            self.pseudos = []
            for (ex_index, item) in enumerate(self.data):
                if ex_index < self.pseudo_num:
                    self.pseudos.append(1)
                else:
                    self.pseudos.append(0)

                if ex_index % 10000 == 0:
                    logger.info("Writing %s example %d of %d", self.type, ex_index, len(self.data))
                text = item[0]

                tokens = []
                for t in text.split():
                    tokens += self.tokenizer.tokenize(t)
                    tokens += ['[unused1]']
                    # tokens += ['-']
                tokens = tokens[:-1]
                # text_ = ''.join(tokens)
                tokens_ = copy.deepcopy(tokens)
                tokens_ = ['[CLS]'] + tokens_ + ['[SEP]']

                mapping = self.get_token_mapping(text, tokens_)
                start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
                end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}

                # 将raw_text的下标 与 token的start和end下标对应
                encoder_txt = self.tokenizer.encode_plus(tokens, max_length=max_len, truncation=True)
                input_ids = encoder_txt["input_ids"]
                token_type_ids = encoder_txt["token_type_ids"]
                attention_mask = encoder_txt["attention_mask"]
                features.append(
                    [text, input_ids, attention_mask, token_type_ids, start_mapping,
                     end_mapping, item[1:], mapping]
                )

        else:
            # TODO 测试
            for (ex_index, item) in enumerate(self.data):
                if ex_index % 10000 == 0:
                    logger.info("Writing %s example %d of %d", self.type, ex_index, len(self.data))
                text = item[0]
                # 将raw_text的下标 与 token的start和end下标对应
                tokens = []
                for t in text.split():
                    tokens += self.tokenizer.tokenize(t)
                    tokens += ['[unused1]']
                    # tokens += ['-']
                tokens = tokens[:-1]
                tokens_ = copy.deepcopy(tokens)
                tokens_ = ['[CLS]'] + tokens_ + ['[SEP]']

                mapping = self.get_token_mapping(text, tokens_)

                encoder_txt = self.tokenizer.encode_plus(tokens, max_length=max_len, truncation=True)
                input_ids = encoder_txt["input_ids"]
                token_type_ids = encoder_txt["token_type_ids"]
                attention_mask = encoder_txt["attention_mask"]
                features.append((text, input_ids, attention_mask, token_type_ids, mapping))

        return features

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        """Numpy函数，将序列padding到同一长度
        """
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    def collate(self, examples):

        if self.type != 'test':

            raw_text_list, batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids = [], [], [], [], []
            pseudos = []
            # mappings = []
            for item in examples:
                text, input_ids, attention_mask, token_type_ids, start_mapping, end_mapping, \
                label, mapping, pseudo = item
                pseudos.append(pseudo)

                raw_text_list.append(text)
                batch_input_ids.append(torch.tensor(input_ids))
                batch_segment_ids.append(torch.tensor(token_type_ids))
                batch_attention_mask.append(torch.tensor(attention_mask))

                labels = np.zeros((len(ent2id), max_len, max_len), dtype=np.int)
                for start, end, label in label:
                    if start in start_mapping and end in end_mapping:
                        start = start_mapping[start]
                        end = end_mapping[end]
                        labels[label, start, end] = 1
                batch_labels.append(labels[:, :len(input_ids), :len(input_ids)])
                # mappings.append(mapping)
            batch_input_ids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
            batch_segment_ids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
            batch_attention_mask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()
            batch_labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=3)).long()

            return raw_text_list, batch_input_ids, batch_attention_mask, batch_segment_ids, batch_labels, pseudos
        else:
            raw_text_list, batch_input_ids, batch_attention_mask, batch_segment_ids, mappings = [], [], [], [], []
            for item in examples:
                text, input_ids, attention_mask, token_type_ids, mapping = item
                raw_text_list.append(text)
                batch_input_ids.append(input_ids)
                batch_segment_ids.append(token_type_ids)
                batch_attention_mask.append(attention_mask)
                mappings.append(mapping)
            batch_input_ids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
            batch_segment_ids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
            batch_attention_mask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()

            return raw_text_list, batch_input_ids, batch_attention_mask, batch_segment_ids, mappings

    def __getitem__(self, index):
        item = self.features[index] + [self.pseudos[index]]
        return item




if __name__ == '__main__':
    pass