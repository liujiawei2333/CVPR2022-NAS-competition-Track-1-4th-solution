import numpy as np
import random

'''Fair sampling from FairNAS:https://arxiv.org/abs/1907.01845.pdf'''

"""
the list's formation of width:
[
    [], # stem channel
    [c1, c2, c3, ...], # stage 1 channel
    [c1, c2, c3, ...], # stage 2 channel
    [c1, c2, c3, ...], # stage 3 channel
    [c1, c2, c3, ...], # stage 4 channel 
]

the list's formation of depth: [d1, d2, d3, d4]
"""

def make_divisible(v, divisor=8, min_value=1):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def code2channel(stem_basic_channel, stage_basic_channels, expand_ratio_list, block_nums, channel_code_index_list):
    width = list()
    channel_index = 0

    stem_channel_code_index = channel_code_index_list[channel_index]
    stem_channel = make_divisible(round(stem_basic_channel * expand_ratio_list[stem_channel_code_index]))
    width.append([stem_channel])
    channel_index += 1

    for block_index, stage_nums in enumerate(block_nums):
        basic_channel = stage_basic_channels[block_index]
        for stage_index in range(stage_nums):
            if stage_index == 0:
                width.append([])

            for _ in range(2):
                channel_code_index = channel_code_index_list[channel_index]
                channel = make_divisible(round(basic_channel * expand_ratio_list[channel_code_index]))
                width[-1].append(channel)
                channel_index += 1
    return width


def get_fair_width_group(stem_basic_channel, stage_basic_channels, expand_ratio_list, block_nums):
    op_nums = len(expand_ratio_list)
    layer_nums = sum(block_nums) * 2 + 1  # stem layer, each layer has 2 conv ops in every stage
    width_group = list()
    channel_code_index_list = [list(np.random.choice(a=op_nums, size=op_nums, replace=False)) for _ in range(layer_nums)]

    for ratio_index in range(len(expand_ratio_list)):
        cur_channel_code_index_list = [channel_code_index_list[channel_index][ratio_index]
                                       for channel_index in range(len(channel_code_index_list))]
        width_group.append(code2channel(stem_basic_channel, stage_basic_channels,
                                        expand_ratio_list, block_nums, cur_channel_code_index_list))
    return width_group


def get_fair_depth_group(block_nums):
    """
    Propose: fair sample depth strictly based the FairNas.

    However, because the number of the depth candidate in every stage is not same,
    this func can not sample depth strict fairly.

    The solution of cvpr-workshop2021-trace1-3 expand the number of channel to the max number of channel candidate,
    to apply the fair sample on different number of channel candidate.

    This func implement the ways of FairNas like the cvpr-workshop2021-trace1-3,
    in which expand the number of depth to the max number of depth candidate
    """
    depth_per_stage_group = list()
    depth_group = list()

    max_layer_nums = max(block_nums) - 2 + 1
    for block_index, stage_nums in enumerate(block_nums):
        cur_layer_nums = stage_nums - 2 + 1
        layer_num_range = list(range(2, stage_nums + 1)) * int(round(max_layer_nums / cur_layer_nums))
        depth = list(np.random.choice(layer_num_range, max_layer_nums, replace=False))
        depth_per_stage_group.append(depth)

    for op_nums_index in range(max_layer_nums):
        depth_group.append([])
        for stage_index in range(len(block_nums)):
            depth_group[-1].append(depth_per_stage_group[stage_index][op_nums_index])

    return depth_group


def sample_width_group(stem_basic_channel, stage_basic_channels, expand_ratio_list, group_nums,
                       block_nums, max_channel=False, min_channel=False):
    assert (max_channel and min_channel) is False, "flag of max_channel and min_channel can not be True the same time"

    def _get_channel(basic_channel):
        if max_channel:
            expand_ratio = max(expand_ratio_list)
        elif min_channel:
            expand_ratio = min(expand_ratio_list)
        else:
            expand_ratio = random.choice(expand_ratio_list)

        result = make_divisible(round(basic_channel * expand_ratio))
        return result

    width_group = list()

    for group_index in range(group_nums):
        width = list()
        width.append([_get_channel(stem_basic_channel)])

        for idx, channel in enumerate(stage_basic_channels):
            width.append([])
            for _ in range(block_nums[idx]):
                width[-1].append(_get_channel(channel))
                width[-1].append(_get_channel(channel))
        width_group.append(width)

    return width_group


def sample_depth_group(block_nums, group_nums, max_depth=False, min_depth=False):
    assert (max_depth and min_depth) is False, "flag of max_depth and min_depth can not be True the same time"

    depth_group = list()

    min_stage_nums = 2
    for group_index in range(group_nums):
        depth = list()
        for stage_nums in block_nums:
            if min_depth:
                depth.append(min_stage_nums)
            elif max_depth:
                depth.append(stage_nums)
            else:
                depth.append(random.randint(min_stage_nums, stage_nums))

        depth_group.append(depth)

    return depth_group


def get_width_and_depth_group(stem_basic_channel, stage_basic_channels, expand_ratio_list, block_nums, group_nums=1,
                            fair_width=False, fair_depth=False, min_c=False, max_c=False, min_d=False, max_d=False):

    if fair_width or fair_depth:
        assert len(expand_ratio_list) == group_nums, "if sample fairly, group number must be " \
                                                    "the same with the number of candidate channels"

    if fair_width:
        assert min_c is False and max_c is False, "if sample channel fairly, the number of channel can not be setting"
        width_group = get_fair_width_group(stem_basic_channel, stage_basic_channels, expand_ratio_list, block_nums)
    else:
        width_group = sample_width_group(stem_basic_channel, stage_basic_channels, expand_ratio_list, group_nums,
                                        block_nums, max_channel=max_c, min_channel=min_c)

    if fair_depth:
        assert min_d is False and max_d is False, "if sample depth fairly, the number of depth can not be setting"
        depth_group = get_fair_depth_group(block_nums)
    else:
        depth_group = sample_depth_group(block_nums, group_nums, max_depth=max_d, min_depth=min_d)

    assert(len(width_group) == len(depth_group)), "the group number of channel and depth must be the same"

    new_width_group = list()

    #  crop channel list according to the depth
    for group_index in range(len(width_group)):
        cur_width_list = width_group[group_index]
        cur_depth_list = depth_group[group_index]

        tmp_width_list = list()
        tmp_width_list.append(cur_width_list[0])
        for depth_idx, cur_depth in enumerate(cur_depth_list):
            tmp_width_list.append(cur_width_list[depth_idx + 1][:cur_depth * 2])

        new_width_group.append(tmp_width_list)

    if len(new_width_group) == 1:
        new_width_group = new_width_group[0]
    if len(depth_group) == 1:
        depth_group = depth_group[0]

    return new_width_group, depth_group
