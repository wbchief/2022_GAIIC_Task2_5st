# coding:utf-8
# !/usr/bin/python

import gc
import os
import copy
import json
import time
import torch
import random
import numpy as np
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertModel, RobertaModel, RobertaConfig, BertConfig, BertTokenizer
import sys
sys.path.append('/home/mw/project/code')

from models.loss import loss_fun
from tools.finetune_args import args
from utils.functions_utils import swa
from tools.common import init_logger, logger
from callback.adversarial import FGM, PGD, EMA
from callback.optimizater.lookahead import Lookahead
from pre_model.modeling_nezha import NeZhaModel
from pre_model.configuration_nezha import NeZhaConfig
from utils.data_loader import load_data, EntDataset3
from models.GlobalPointer import GlobalPointer, MetricsCalculator, GlobalPointerBert

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
        # logger.info("nezha")
        config = NeZhaConfig.from_pretrained(bert_model_path)
        config.output_hidden_states = True
        encoder = NeZhaModel.from_pretrained(bert_model_path, config=config)
    if model_type == 'roberta':
        config = RobertaConfig.from_pretrained(bert_model_path, num_labels=ent_type_size)
        encoder = RobertaModel.from_pretrained(bert_model_path, config=config)
    return encoder, config, tokenizer


def save_model(model, global_step):
    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
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
            logits = model(input_ids, attention_mask, segment_ids)

        f1, p, r = metrics.get_evaluate_fpr(logits, labels)
        total_f1_ += f1
        total_precision_ += p
        total_recall_ += r

    avg_f1 = total_f1_ / (len(ner_loader_evl))
    avg_precision = total_precision_ / (len(ner_loader_evl))
    avg_recall = total_recall_ / (len(ner_loader_evl))

    eval_metric['f1'], eval_metric['precision'], eval_metric['recall'] = avg_f1, avg_precision, avg_recall

    return eval_metric

def get_model_path_list(base_dir):
    """
    从文件夹中获取 model.pt 的路径
    """
    model_lists = []

    for root, dirs, files in os.walk(base_dir):

        for _file in files:
            if 'model.pt' == _file and 'saw' not in root:
                model_lists.append(os.path.join(root, _file).replace("\\", '/'))

    model_lists = sorted(model_lists,
                         key=lambda x: (x.split('/')[-3], int(x.split('/')[-2].split('-')[-1])))

    return model_lists



def swa(model, model_dir, swa_start=1):
    """
    swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
    """
    model_path_list = get_model_path_list(model_dir)
    print(model_path_list)

    assert 1 <= swa_start < len(model_path_list) - 1, \
        f'Using swa, swa start should smaller than {len(model_path_list) - 1} and bigger than 0'

    swa_model = copy.deepcopy(model)
    swa_n = 0.

    with torch.no_grad():
        for _ckpt in model_path_list[swa_start:-2]:
            print(_ckpt)
            # logger.info(f'Load model from {_ckpt}')
            model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu')))
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)

            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

            swa_n += 1

    # use 100000 to represent swa to avoid clash
    swa_model_dir = os.path.join(model_dir, f'checkpoint-saw2-6')
    if not os.path.exists(swa_model_dir):
        os.mkdir(swa_model_dir)

    # logger.info(f'Save swa model in: {swa_model_dir}')

    swa_model_path = os.path.join(swa_model_dir, 'model.pt')

    torch.save(swa_model.state_dict(), swa_model_path)

    return swa_model


def train_step():
    # output_dir = '/home/baiph/GAIIC2022_prelim/data/model_data/gp_output/20220431/checkpoint-331250_fgm_pgd-new'
    output_dir = '/home/mw/project/data/gp_output/20220512/checkpoint-331250_mlm'
    swa_start = 2
    encoder, config, tokenizer = build_transformer_model(args.bert_model_path, args.model_type)

    datalist = load_data(args.train_file)

    # logger.info(f"数量{len(datalist)}")
    # logger.info(f"训练数量:{len(datalist)}")

    ner_eval = EntDataset3(datalist[-4000:], tokenizer=tokenizer)
    ner_loader_eval = DataLoader(ner_eval, batch_size=args.batch_size * 8, num_workers=args.num_workers,
                                 collate_fn=ner_eval.collate, shuffle=False, pin_memory=True)

    # default set to gp
    if args.model == 'gp_bert':
        # logger.info("GlobalPointerBert")
        model = GlobalPointerBert(encoder, ent_type_size, 64).to(device)
    else:
        # logger.info("GlobalPointer")
        model = GlobalPointer(encoder, ent_type_size, 64).to(device)

    swa_raw_model = copy.deepcopy(model)

    metrics = MetricsCalculator()

    swa(swa_raw_model, output_dir, swa_start=swa_start)

    model.load_state_dict(torch.load(os.path.join(output_dir, 'checkpoint-saw2-6/model.pt'),
                                     map_location='cuda:0'))

    metric = evaluate(model, ner_loader_eval, metrics)

    f1_score = metric['f1']
    recall = metric['recall']
    precision = metric['precision']

    print("\n >> Average: \t Eval f1 : {}\t Precision : {}\t Recall : {}\t ".
                format(f1_score, recall, precision))
                
                



def main():
    same_seeds(args.seed)
    train_step()


if __name__ == '__main__':
    main()
    # get_model_path_list(r'E:\code\GAIIC\JDNER\GlobalPointer_out\nezha-cn-base_base')
