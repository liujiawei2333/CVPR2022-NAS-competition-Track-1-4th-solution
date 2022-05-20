from collections import OrderedDict
import numpy as np
import logging
import torch
import json
import sys
import os

from modules.sampler import make_divisible
from .config import setup

def get_logger(save_dir):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    return logging


def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(
        group_weight_decay) + len(group_no_weight_decay)
    groups = [dict(params=group_weight_decay), dict(
        params=group_no_weight_decay, weight_decay=0.)]
    return groups


def get_model_code(max_channel_code=7, min_block_nums=2, max_block_nums1=5, max_block_nums2=8, stage_nums=4):
    def _uniform_sample(min_value, max_value):
        value = int(np.random.randint(low=min_value, high=max_value + 1, size=1))
        return value

    output_list = list()
    stem_channel_scale_ratio_code = _uniform_sample(1, max_channel_code)

    block_nums_list = list()

    for stage_index in range(stage_nums):
        max_repeat_nums = max_block_nums2 if stage_index == 2 else max_block_nums1
        repeat_nums = _uniform_sample(min_block_nums, max_repeat_nums)
        block_nums_list.append(repeat_nums)

        channel_scale_ratio_code_list = list()
        for block_index in range(max_repeat_nums):
            if block_index >= repeat_nums:
                channel_scale_ratio_code_list.append(0)
                channel_scale_ratio_code_list.append(0)
            else:
                channel_scale_ratio_code_list.append(_uniform_sample(1, max_channel_code))
                channel_scale_ratio_code_list.append(_uniform_sample(1, max_channel_code))
        output_list.append(channel_scale_ratio_code_list)

    output_list.insert(0, [stem_channel_scale_ratio_code])
    output_list.insert(0, block_nums_list)

    return output_list


def load_model(model, model_save_path):
    print("loading model from {}".format(model_save_path))
    modeldict = model.state_dict()
    new_state_dict = OrderedDict()
    for k, v in torch.load(model_save_path, map_location="cpu" if not torch.cuda.is_available() else "cuda").items():
        name = k.replace("module.", "") if "module." in k else k
        if name in modeldict:
            new_state_dict[name] = v
        else:
            print('model skipped: %s' % k)
    modeldict.update(new_state_dict)
    model.load_state_dict(modeldict)
    return model


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def load_json(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


def dict2json(dict_obj, save_path):
    with open(save_path, "w") as f:
        json.dump(dict_obj, f)


def code2list(code_str, stem_base_channel, block_base_channels, expand_ratio_list):
    if len(code_str) > 51:
        code_str = code_str[1:]
    width = list()
    depth = list()

    code_index = 0
    for stage_index in range(len(block_base_channels)):
        cur_stage_depth = code_str[stage_index]
        depth.append(int(cur_stage_depth))
        code_index += 1
    stem_channel_ratio = expand_ratio_list[int(code_str[code_index]) - 1]
    stem_channel = make_divisible(round(stem_base_channel * stem_channel_ratio))
    width.append([stem_channel])
    code_index += 1

    for stage_index in range(len(block_base_channels)):
        max_stage_depth = 8 if stage_index == 2 else 5
        cur_stage_depth = depth[stage_index]

        stage_basic_channel = block_base_channels[stage_index]
        width.append([])

        for index in range(max_stage_depth):
            if index < cur_stage_depth:
                cur_middle_ratio = expand_ratio_list[int(code_str[code_index]) - 1]  # code index start at 1
                cur_out_ratio = expand_ratio_list[int(code_str[code_index + 1]) - 1]  # code index start at 1
                middle_channel = make_divisible(round(stage_basic_channel * cur_middle_ratio))
                out_channel = make_divisible(round(stage_basic_channel * cur_out_ratio))
                width[-1].append(middle_channel)
                width[-1].append(out_channel)
            code_index += 2
    return width, depth


def build_args_and_env(console_args):
    assert console_args.train_config_file and os.path.isfile(console_args.train_config_file), \
        'cannot locate config file'
    assert console_args.common_config_file and os.path.isfile(
        console_args.common_config_file), 'cannot locate common config file'
    args = setup(console_args.train_config_file)
    common_args = setup(console_args.common_config_file)
    args.config_file = console_args.train_config_file
    common_args.common_config_file = console_args.common_config_file

    return args, common_args