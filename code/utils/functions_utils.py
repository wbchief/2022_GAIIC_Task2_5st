import copy
import os

import torch
import sys

base_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(base_dir, '../'))


def get_model_path_list(base_dir):
    """
    从文件夹中获取 model.pt 的路径
    """
    model_lists = []

    for root, dirs, files in os.walk(base_dir):
        for _file in files:
            if 'model.pt' == _file:
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
        for _ckpt in model_path_list[swa_start:]:
            print(_ckpt)
            # logger.info(f'Load model from {_ckpt}')
            model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu')))
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)

            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

            swa_n += 1

    # use 100000 to represent swa to avoid clash
    swa_model_dir = os.path.join(model_dir, f'checkpoint-saw-100000')
    if not os.path.exists(swa_model_dir):
        os.mkdir(swa_model_dir)

    # logger.info(f'Save swa model in: {swa_model_dir}')

    swa_model_path = os.path.join(swa_model_dir, 'model.pt')

    torch.save(swa_model.state_dict(), swa_model_path)

    return swa_model


if __name__ == '__main__':
    # # get_model_path_list('/home/baiph/JDNER/gp_out/nezha-cn-base-chage-kong')
    #
    #
    # # tokenizer
    # encoder, config, tokenizer = build_transformer_model(args.bert_model_path, args.model_type)
    # # train_data and val_data
    # # datalist = load_data(args.train_file)
    #
    # # GP MODEL
    # model = GlobalPointer(encoder, 52, 64).to(torch.device("cuda:0"))
    # swa(model, args.output_dir, swa_start=3)
    pass