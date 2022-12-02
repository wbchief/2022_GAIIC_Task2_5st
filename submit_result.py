import copy
import os.path
import sys

from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig, BertTokenizer
import os

base_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(base_dir, 'code/models'))
from GlobalPointer import GlobalPointer1
import json
import torch
import numpy as np
from tqdm import tqdm

# sys.path.append(os.path.join(base_dir, 'code/tools'))
# from finetune_args import args
sys.path.append(os.path.join(base_dir, 'code/utils'))
from test_data_loader import EntDataset3

# base_dir = os.path.dirname(__file__)
# sys.path.append(os.path.join(base_dir, './'))
sys.path.append(os.path.join(base_dir, 'code/pre_model'))
from configuration_nezha import NeZhaConfig
from modeling_nezha import NeZhaModel

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


same_seeds(1998)
bert_model_path = '/home/mw/input/pretrain_model_5637'
# bert_model_path = args.bert_model_path
model_type = 'nezha'
model_ttt = 'gp'

ent2id_path = '/home/mw/project/data/ent2id.json'
# ent2id_path = args.ent2id_path
device = torch.device("cuda:0")

ent2id = json.load(open(ent2id_path, encoding="utf-8"))
ent_type_size = len(ent2id)
id2ent = {}
for k, v in ent2id.items():
    id2ent[v] = k

def build_transformer_model(bert_model_path, model_type='bert'):
    tokenizer = BertTokenizer.from_pretrained(bert_model_path, do_lower_case=True)
    if model_type == 'bert':
        config = BertConfig.from_pretrained(bert_model_path, num_labels=ent_type_size)
        encoder = BertModel.from_pretrained(bert_model_path, config=config)
    if model_type == 'nezha':
        config = NeZhaConfig.from_pretrained(bert_model_path)
        config.output_hidden_states = True
        encoder = NeZhaModel(config=config)

    return encoder, config, tokenizer

from functools import cmp_to_key

def cmp(x,y):
    if int(x['end_idx'])<int(y['end_idx']):
        return -1
    else:
        return 1

def predict(ner_loader, tokenizer, model):
    # 新加
    # low_frequency_type = ['51', '33', '42', '24', '53', '35', '26']
    # 结束
    datalist = []
    id=0
    for batch in tqdm(ner_loader):
        text, input_ids, attention_mask, segment_ids, mapping = batch
        input_ids, attention_mask, segment_ids = input_ids.to(device), attention_mask.to(
            device), segment_ids.to(device)

        scores = model(input_ids, attention_mask, segment_ids)
        scores = scores[0].data.cpu().numpy()
        
        text = text[0]
        mapping = mapping[0]
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        entities = []
        for l, start, end in zip(*np.where(scores > 0)):
            entities.append(
                {"start_idx": mapping[start][0], "end_idx": mapping[end][-1], "type": id2ent[l],'score':scores[l][start][end]}
            )
        entities.sort(key=cmp_to_key(cmp))
        unnested_entities=[]
        for idx,ent in enumerate(entities):
            if idx==0:
                candidate_ent=ent
                continue
            if candidate_ent['end_idx']>=ent['start_idx']:
                if ent['score']>candidate_ent['score']:
                    candidate_ent=ent
            else:
                unnested_entities.append(candidate_ent)
                candidate_ent=ent
        unnested_entities.append(candidate_ent)

        datalist.append({
            'text': text,
            'entities': unnested_entities
        })
        id+=1
    return datalist

def save_submit(datalist, save_path):
    
    with open(save_path, 'w', encoding='utf-8') as f:
        for data in datalist:
            text = data['text']
            # label = 'OPPO闪充充电器 X9070 X9077 R5 快充头通用手机数据线 套餐【2.4充电头+数据线 】 安卓 1.5m'
            labels = ['O'] * len(text)
            entities = data['entities']
            for ent in entities:
                start = ent['start_idx']
                end = ent['end_idx']
                type = ent['type']

                #将单类型但预测实体与entity_type_dict不符的替换掉
                #if text[start:end+1] in entity_type_dict and type!=entity_type_dict[text[start:end+1]]:
                    # print(text[start:end+1],type," --> ",entity_type_dict[text[start:end+1]])
                #    type=entity_type_dict[text[start:end+1]]
                # if 'I' in labels[start]:
                #     continue
                # if 'B' in labels[start] and 'O' not in labels[end]:
                #     continue
                # if 'O' in labels[start] and 'B' in labels[end]:
                #     continue
                labels[start] = 'B-' + type
                for i in range(start + 1, end + 1):
                    labels[i] = 'I-' + type

            for word, label in zip(text, labels):
                f.write(word + ' ' + label + '\n')
            f.write('\n')


def load_test_data1(path):
    D = []
    for d in json.load(open(path, encoding='utf-8')):
        D.append([d['text']])
    return D


def load_test_data(path):
    datalist = []
    # print("开始处理数据")
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines.append('\n')

        text = []
        for line in lines:
            if line == '\n':
                text = ''.join(text)
                if text == '':
                    continue

                datalist.append({
                    "id": count,
                    'text': text,
                })
                count += 1
                text = []

            elif line == '  \n':
                text.append(' ')
            else:
                line = line.strip('\n')
                term = line[0]
                text.append(term)
    D = []
    for d in datalist:
        D.append([d['text']])
    return D

def post_processing(train_path,predict_path,save_path):
    '''
    :param file_path:
    :return:
    '''
    num=0
    ans=[]
    trains_aloneB=[]
    #train里的单字符实体
    with open(train_path,'r',encoding='utf-8') as f:
        lines=f.readlines()
        for i in range(len(lines)):
            now_line=lines[i]
            if len(now_line)>3 and now_line[2]=='B':
                if(i==len(lines)-1) or (len(lines[i+1])<3) or lines[i+1][2]!='I' or  lines[i+1][3:]!=now_line[3:]:
                    trains_aloneB.append(now_line)


    with open(predict_path,'r',encoding='utf-8') as f:
        lines=f.readlines()
        for i in range(len(lines)):
            now_line=lines[i]
            if i==0 or i==len(lines)-1:
                continue
            pre_line=lines[i-1]
            nex_line=lines[i+1]
            if now_line[0] in [' ','/','+'] and now_line[2]=='I':
                if len(pre_line)>3 and len(nex_line)>3 and pre_line[2]=='I' and nex_line[2]=='B' and \
                        pre_line[3:]==nex_line[3:] and now_line[3:]==pre_line[3:]:
                    num+=1
                    lines[i]=now_line[0]+" O\n"

        #处理单字符实体
        for i in range(len(lines)):
            now_line=lines[i]
            if len(now_line)>3 and now_line[2]=='B':
                if(i==len(lines)-1) or (len(lines[i+1])<3) or lines[i+1][2]!='I' or  lines[i+1][3:]!=now_line[3:]:
                    if now_line not in trains_aloneB:
                        lines[i]=now_line[0]+' O\n'
                        num+=1
            ans.append(lines[i])

    with open(save_path,'w',encoding='utf-8') as f:
        for line in ans:
            f.write(line)

def predict_to_file(file_path, model_path, save_path,batch_size):
    encoder, config, tokenizer = build_transformer_model(bert_model_path, model_type=model_type)
    model = GlobalPointer1(encoder, ent_type_size, 64).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location='cuda:0'), False
    )
    model.eval()

    ner_train = EntDataset3(
        load_test_data(file_path),
        tokenizer=tokenizer,
        type='test'
    )
    ner_loader_train = DataLoader(ner_train, batch_size=batch_size, collate_fn=ner_train.collate,
                                  shuffle=False, num_workers=0, drop_last=False)
    datalist = predict(ner_loader_train, tokenizer, model)
    save_submit(datalist, save_path)


def pred_BIO(path_word:str, path_sample:str, batch_size:int):
    model_path = '/home/mw/project/best_model/model.pt'
    print(model_path)
    predict_to_file(path_word, model_path, '/home/mw/project/submission/results.txt', batch_size)
    post_processing('/home/mw/project/data/train.txt', '/home/mw/project/submission/results.txt','/home/mw/project/submission/results.txt')


if __name__ == '__main__':
    pass

