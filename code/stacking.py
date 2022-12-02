# encoding=utf-8

import json
import os
import sys
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append('/home/mw/project/code')
from models.GlobalPointer import GlobalPointer1
from utils.data_loader import load_data, EntDataset3
from tools.finetune_args import args
from train1 import build_transformer_model
import numpy as np
import random


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


device = torch.device("cuda:0")
seed = 1998
same_seeds(seed)
ent2id = json.load(open(args.ent2id_path, encoding="utf-8"))
ent_type_size = len(ent2id)
id2ent = {}
for k, v in ent2id.items():
    id2ent[v] = k


def load_unlabeled_data(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    lines = lines[start:unlabeled_num]
    print(f"开始位置:{start}, 结束位置：{unlabeled_num}， 总数量：{unlabeled_num - start}, 伪标数量:{len(lines)}")
    D = []
    for line in lines[start:unlabeled_num]:
        if len(line) > args.max_length:
            continue
        D.append([line])
        
    return D


def vote(entities_list, threshold=0.9):
    """
    实体级别的投票方式  (entity_type, entity_start, entity_end, entity_text)
    :param entities_list: 所有模型预测出的一个文件的实体
    :param threshold:大于70%模型预测出来的实体才能被选中
    :return:[{type:[(start, end), (start, end)]}, {type:[(start, end), (start, end)]}]
    """
    threshold_nums = int(len(entities_list) * threshold)
    entities_dict = defaultdict(int)
    entities = defaultdict(list)

    for _entities in entities_list:
        for _type in _entities:
            for _ent in _entities[_type]:
                entities_dict[(_type, _ent[0], _ent[1])] += 1

    for key in entities_dict:
        if entities_dict[key] >= threshold_nums:
            entities[key[0]].append((key[1], key[2]))

    return entities

def vote1(entities_list, threshold=0.9):
    """
    实体级别的投票方式  (entity_type, entity_start, entity_end, entity_text)
    :param entities_list: 所有模型预测出的一个文件的实体[{type1:[(), (), ...], type2:[(), ()]},
    {type1:[(), (), ...], type2:[(), ()]}, {type1:[(), (), ...], type2:[(), ()]},
    {type1:[(), (), ...], type2:[(), ()]}, {type1:[(), (), ...], type2:[(), ()]}]
    :param threshold:大于70%模型预测出来的实体才能被选中
    :return:[{type:[(start, end), (start, end)]}, {type:[(start, end), (start, end)]}]
    """
    threshold_nums = int(len(entities_list) * threshold)
    entities_dict = defaultdict(int)
    entities = defaultdict(list)

    # 第i个模型预测的结果
    for _entities in entities_list:
        # 遍历实体
        for _type in _entities:
            for _ent in _entities[_type]:
                entities_dict[(_type, _ent[0], _ent[1])] += 1

    # 是否保留改样本
    for key in entities_dict:
        if entities_dict[key] >= threshold_nums:
            entities[key[0]].append((key[1], key[2]))
        else:
            # 出现实体少于3个模型预测的，该条数据去掉
            entities = []
            break
    return entities

def get_entity_bio(seq, id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0, 1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

from functools import cmp_to_key

def cmp(x,y):
    if int(x[1])<int(y[1]):
        return -1
    else:
        return 1
    
def merge(ner_loader, model_list, model_all, save_path):
    print("merge")
    datalist = []
    remove_count, save_count = 0, 0
    with torch.no_grad():
        for batch in tqdm(ner_loader):
            text, input_ids, attention_mask, segment_ids, mapping = batch
            input_ids, attention_mask, segment_ids = input_ids.to(device), attention_mask.to(
                device), segment_ids.to(device)

            scores_list = []
            for idx, model in enumerate(model_list):
                model.eval()

                scores = model(input_ids, attention_mask, segment_ids)
                scores = scores.data.cpu().numpy()
                scores_list.append(scores)
            model_all.eval()
            scores_all = model_all(input_ids, attention_mask, segment_ids)
            scores_all = scores_all.data.cpu().numpy()
            # 解码
            # print("开始解码", len(text))
            for i in range(len(text)):
                # text, input_ids, attention_mask, segment_ids, mapping = data
                # print(len(text), len(mapping), i)
                # print(text.shape)
                text_1 = text[i]
                mapping_1 = mapping[i]
                scores = 0
                for score in scores_list:
                    scores += score[i] / kfold
                    
                scores = scores * 0.7 + scores_all[i] * 0.3
                scores[:, [0, -1]] -= np.inf
                scores[:, :, [0, -1]] -= np.inf
                # print(scores.shape)
                entities = []
                for l, start, end in zip(*np.where(scores > 0)):
                    entities.append([mapping_1[start][0], mapping_1[end][-1], id2ent[l],scores[l][start][end]])


                entities.sort(key=cmp_to_key(cmp))
                unnested_entities=[]
                for idx,ent in enumerate(entities):
                    if idx==0:
                        candidate_ent=ent
                        continue
                    if candidate_ent[1]>=ent[1]:
                        if ent[3]>candidate_ent[3]:
                            candidate_ent=ent
                    else:
                        unnested_entities.append(candidate_ent)
                        candidate_ent=ent
                unnested_entities.append(candidate_ent)
                entities=unnested_entities

                # 实体修正
                labels = ['O'] * len(text_1)
                
                try:
                    for ent in entities:
                        start = ent[0]
                        end = ent[1]
                        type = ent[2]
                        # if 'I' in labels[start]:
                        #     continue
                        # if 'B' in labels[start] and 'O' not in labels[end]:
                        #     continue
                        # if 'O' in labels[start] and 'B' in labels[end]:
                        #     continue
                        labels[start] = 'B-' + type
                        for i in range(start + 1, end + 1):
                            labels[i] = 'I-' + type
                except:
                    continue
                
                entities = []
                count = 0
                for _type, _start_idx, _end_idx in get_entity_bio(labels, id2label=None):
                    entities.append([_start_idx, _end_idx, _type])     
                if len(entities) > 0:
                    datalist.append({
                        'text': text_1,
                        'entity_list': entities
                    })
                    save_count += 1
                else:
                    remove_count += 1
                if save_count % 10000 == 0:
                    print(f"{save_count}/{len(ner_loader) *128 }")
            if save_count > max_num:
                break
    print(f"移除样本：{remove_count}, 保留样本:{save_count}")
    # return datalist
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(datalist, ensure_ascii=False, indent=4))

def predict(ner_loader, model_list, save_path):
    datalist = []
    remove_count, save_count = 0, 0
    with torch.no_grad():
        for batch in tqdm(ner_loader):
            text, input_ids, attention_mask, segment_ids, mapping = batch
            input_ids, attention_mask, segment_ids = input_ids.to(device), attention_mask.to(
                device), segment_ids.to(device)

            scores_list = []
            for idx, model in enumerate(model_list):
                model.eval()

                scores = model(input_ids, attention_mask, segment_ids).data.cpu().numpy()
                scores_list.append(scores)

            # 解码
            # print("开始解码", len(text))
            for i in range(len(text)):
                text_1 = text[i]
                mapping_1 = mapping[i]
                entities_ls = []
                for scores in scores_list:
                    scores = scores[i]
                    scores[:, [0, -1]] -= np.inf
                    scores[:, :, [0, -1]] -= np.inf
                    # print(scores.shape)
                    predict_entities = {}
                    for l, start, end in zip(*np.where(scores > 0)):
                        if id2ent[l] not in predict_entities:
                            predict_entities[id2ent[l]] = [(mapping_1[start][0], mapping_1[end][-1])]
                        else:
                            predict_entities[id2ent[l]].append((mapping_1[start][0], mapping_1[end][-1]))

                    entities_ls.append(predict_entities)

                if type == 'unlabeled':
                    entities = vote1(entities_ls, 0.8)
                else:
                    entities = vote(entities_ls, 0.6)
                if len(entities) != 0:
                    tmp = []
                    for key in entities:
                        for ent in entities[key]:
                            tmp.append(
                                [ent[0], ent[-1], key]
                            )
                    datalist.append({
                        'text': text_1,
                        'entity_list': tmp
                    })
                    save_count += 1
                else:
                    remove_count += 1
                if save_count % 10000 == 0:
                    print(f"{save_count}/{len(ner_loader)*128}")

            if save_count > max_num:
                break
    print(f"移除样本：{remove_count}, 保留样本:{save_count}")
    # return datalist
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(datalist, ensure_ascii=False, indent=4))


def predict_unlabeled_data(datalist, model_dirs, save_path):
    # encoder, config, tokenizer = build_transformer_model(args.bert_model_path, model_type='nezha')

    # model_list = torch.nn.ModuleList()
    model_list = []
    for i in range(kfold):
        print(f"----------------------------------------------------{i+1}----------------------------------------------")
        print(model_dirs[i])
        encoder, config, tokenizer = build_transformer_model(args.bert_model_path, model_type='nezha')
        model = GlobalPointer1(encoder, ent_type_size, 64).to(device)
        model.load_state_dict(
            torch.load(model_dirs[i], map_location='cuda:0')
        )
        model_list.append(model)
        
  
    encoder, config, tokenizer = build_transformer_model(args.bert_model_path, model_type='nezha')
    model_all = GlobalPointer1(encoder, ent_type_size, 64).to(device)
    model_all.load_state_dict(
        torch.load('/home/mw/temp/model_data/gp_output/model_all_1/checkpoint-saw-100000/model.pt',
                   map_location='cuda:0')
    )

    ner_train = EntDataset3(
        datalist,
        tokenizer=tokenizer,
        type='test'
    )
    ner_loader_train = DataLoader(ner_train, batch_size=128, collate_fn=ner_train.collate,
                                  shuffle=False, num_workers=6, drop_last=False, pin_memory=True)

    # predict(ner_loader_train, model_list, save_path)
    merge(ner_loader_train, model_list, model_all, save_path)
    print("生成完成")


if __name__ == '__main__':

    # type = args.type
    start = 0
    # unlabeled_num = 300000
    unlabeled_num = 10000
    max_num = unlabeled_num - start

    # start = 300000
    # unlabeled_num = 1000000
    # max_num = unlabeled_num - start
    kfold = 10
    type = 'unlabeled'

    # model_dirs = [f'/home/mw/input/10fold_all_202205141290/model_{i+1}.pt' for i in range(kfold)]
    # model_dirs = [f'/home/mw/temp/model_data/gp_output/model_10fold//model_{i+1}.pt' for i in range(kfold)]
    model_dirs = [f'/home/mw/temp/model_data/gp_output/model_10fold/fold-{i + 1}/checkpoint-saw-100000/model.pt' for i in range(kfold)]

    os.makedirs('/home/mw/project/data/pesudo_data', exist_ok=True)
    print('unlabeled')
    # start = 0
    datalist = load_unlabeled_data('/home/mw/input/track2_contest_5713/train_data/train_data/unlabeled_train_data.txt')
    print(len(datalist), '----')
    predict_unlabeled_data(
        datalist,
        model_dirs,
        f'/home/mw/project/data/pesudo_data/unlabeled_pesudo_10-fold_best-model_30w.json'
    )


