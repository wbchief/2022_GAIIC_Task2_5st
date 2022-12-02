# encoding=utf-8
import json
import os.path
import sys
import os

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


def convert_json(train_path, save_path):
    datalist = []
    # jieba增加词表
    # with open('./datasets/words.txt', 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         if line != '\n':
    #             jieba.add_word(line.strip())

    print("开始处理数据")
    with open(os.path.join(train_path, 'train.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines.append('\n')

        text = []
        labels = []
        label_set = set()
        for line in lines:
            # 标题结束
            if line == '\n':
                text = ''.join(text)
                entity_labels = []
                for _type, _start_idx, _end_idx in get_entity_bio(labels, id2label=None):
                    entity_labels.append([_start_idx, _end_idx, _type])
                if text == '':
                    continue
                # words = jieba.cut(text)
                # words = [w if len(w) != 0 else '[unused1]' for w in words]
                datalist.append({
                    'text': text,
                    'entity_list': entity_labels,
                })

                text = []
                labels = []

            elif line == '  O\n':
                text.append(' ')
                labels.append('O')
            else:
                line = line.strip('\n').split()
                if len(line) == 1:
                    term = ' '
                    label = line[0]
                else:
                    term, label = line

                text.append(term)
                label_set.add(label.split('-')[-1])
                labels.append(label)

    label_set.remove("O")
    label_set = sorted(label_set)

    label_dic = {}
    for i, label in enumerate(label_set):
        label_dic[label] = i

    with open(os.path.join(save_path, 'ent2id.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(label_dic, ensure_ascii=False, indent=4))

    with open(os.path.join(save_path, 'all_train.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(datalist, ensure_ascii=False, indent=4))


def convert_json_test(data_path, save_path):
    datalist = []
    print("开始处理数据")
    count = 0
    with open(data_path, 'r', encoding='utf-8') as f:
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

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(datalist, ensure_ascii=False, indent=4))


if __name__ == '__main__':
    type = sys.argv[1]
    train_path = '/home/mw/input/track2_contest_5713/train_data/train_data'
    save_path = '/home/mw/project/data'
    if type == 'train':
        #os.makedirs(os.path.join(save_path, 'tmp_data/dataset'), exist_ok=True)
        os.makedirs(save_path, exist_ok=True)
        convert_json(
            train_path=train_path,
            save_path=save_path
        )
    elif type == 'testB':
        convert_json_test(
            './data/contest_data/preliminary_test_b/word_per_line_preliminary_B.txt',
            './data/tmp_data/dataset/test_B.json'
        )
