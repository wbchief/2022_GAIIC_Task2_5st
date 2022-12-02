# coding:utf-8


import os
import json
from tqdm import tqdm
from argparse import ArgumentParser


def format_data(args):

    """
        process train data
    """
    lines = []
    with open(args.train_path, 'r', encoding='utf-8') as f:
        words, labels = [], []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    lines.append({"words": words, "labels": labels})
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    labels.append("O")
        if words:
            lines.append({"words": words, "labels": labels})

    # print('total train num : ', len(lines))

    with open(args.out_train_path, 'w', encoding='utf-8') as f:
        for line_id, text in enumerate(lines):
            f.write(json.dumps(text) + '\n')

    # """
    # process test data
    # """
    test_lines = []
    with open(args.test_A_path, 'r', encoding='utf-8') as f:
        words = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    test_lines.append({"words": words})
                    words = []
            else:
                word = line.strip()
                words.append(word)
        if words:
            test_lines.append({"words": words})

    with open(args.test_B_path, 'r', encoding='utf-8') as f:
        words = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    test_lines.append({"words": words})
                    words = []
            else:
                word = line.strip()
                words.append(word)
        if words:
            test_lines.append({"words": words})

    print('total test num : ', len(test_lines))

    with open(args.out_test_path, 'w', encoding='utf-8') as f:
        for line_id, text in enumerate(test_lines):
            f.write(json.dumps(text) + '\n')

    unsup_lines = []
    with open(args.unlabeled_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in tqdm(enumerate(lines), desc='reading from unsup data', total=len(lines)):
            line = line.strip()
            words = [w for w in line]
            unsup_lines.append({'words': words})

    print('total test num : ', len(unsup_lines))

    with open(args.out_unlabeled_path, 'w', encoding='utf-8') as f:
        for line_id, text in enumerate(unsup_lines):
            f.write(json.dumps(text) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    data_path =  '/home/mw/input/track2_contest_5713'
    save_path = '/home/mw/temp/tmp_data'
    parser.add_argument('--train_path', type=str, default=os.path.join(data_path, 'train_data/train_data/train.txt'))
    parser.add_argument('--test_A_path', type=str, default=os.path.join(data_path, 'preliminary/word_per_line_preliminary_A.txt'))
    parser.add_argument('--test_B_path', type=str, default=os.path.join(data_path, 'preliminary/word_per_line_preliminary_B.txt'))
    parser.add_argument('--unlabeled_path', type=str, default=os.path.join(data_path, 'train_data/train_data/unlabeled_train_data.txt'))

    parser.add_argument('--out_train_path', type=str, default=os.path.join(save_path, 'processed_data/train.json'))
    parser.add_argument('--out_test_path', type=str, default=os.path.join(save_path, 'processed_data/test.json'))
    parser.add_argument('--out_unlabeled_path', type=str, default=os.path.join(save_path, 'processed_data/unlabeled.json'))

    args = parser.parse_args()

    os.makedirs(os.path.join(save_path, 'processed_data'), exist_ok=True)

    format_data(args)

    with open(args.out_train_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            data = json.loads(line)
            words, labels = data['words'], data['labels']
            print(words)
            print(labels)
            break

    with open(args.out_unlabeled_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            data = json.loads(line)
            words = data['words']
            print(words)
            break
