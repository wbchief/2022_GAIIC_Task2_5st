# coding:utf-8

import os
import argparse

data_path = '/home/mw/temp'
outputs_path = '/home/mw/temp'
pretrain_path = '/home/mw/temp'

def finetune(params=None):

    print("\n Start fine-tuning")

    parser = argparse.ArgumentParser()
    # Required parameters

    parser.add_argument("--model_type", type=str, default='nezha')

    parser.add_argument("--train_file", type=str,
                        default='/home/mw/project/data/all_train.json')
    parser.add_argument("--ent2id_path", type=str,
                        default='/home/mw/project/data/ent2id.json')

    parser.add_argument("--bert_model_path", type=str,
                        default='/home/mw/project/data/pretrain_model_mlm')

    parser.add_argument('--output_dir', type=str,
                         default='/home/mw/temp/model_data/gp_output/model_all_1')

    parser.add_argument('--output_dir_best', type=str,
                        default=os.path.join(outputs_path, 'model_data/gp_output/eval_best'))

    parser.add_argument('--num_workers', type=int, default=6)

    parser.add_argument("--epoch", type=int, default=8)
    # 16
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=110)

    parser.add_argument('--swa_start', type=int, default=2)
    # 4e-4
    parser.add_argument('--lr', type=float, default=4e-5)

    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)

    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--do_eval', type=bool, default=True)

    parser.add_argument('--use_fgm', type=bool, default=True)
    parser.add_argument('--use_pgd', type=bool, default=True)
    parser.add_argument('--adv_k', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--epsilon', type=float, default=0.5)

    parser.add_argument('--model', type=str, default='gp')

    parser.add_argument('--use_ema', type=bool, default=False)
    parser.add_argument('--use_lookahead', type=bool, default=False)
    parser.add_argument('--ema_decay', type=float, default=0.995)

    parser.add_argument('--logging_steps', type=int, default=2250)  # 2250

    parser.add_argument('--seed', type=int, default=1998)  # 1998

    parser.add_argument('--type', type=str, default='testB')
    
    parser.add_argument('--kfold', type=bool, default=False)
    parser.add_argument('--n_splits', type=int, default=10)
    parser.add_argument('--pseudos', type=bool, default=False)
    print(params)
    args = parser.parse_args(args=params)

    return args

args = finetune()

