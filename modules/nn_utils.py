import random

from .activations import *


def int2list(val, repeat_time=1):
    if isinstance(val, list):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == 'h_swish':
        return Hswish(inplace=inplace)
    elif act_func == 'h_sigmoid':
        return Hsigmoid(inplace=inplace)
    elif act_func == 'swish':
        return MemoryEfficientSwish()
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


def make_divisible(v, divisor=8, min_value=1):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def drop_connect(inputs, p, training):
    """Drop connect.
        Args:
            input (tensor: BCWH): Input of this structure.
            p (float: 0.0~1.0): Probability of drop connection.
            training (bool): The running mode.
        Returns:
            output: Output after drop connection.
    """
    assert 0 <= p <= 1, 'p must be in range of [0,1]'
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1.0 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output

def copy_bn(target_bn, src_bn):
    feature_dim = target_bn.num_features

    target_bn.weight.data.copy_(src_bn.weight.data[:feature_dim])
    target_bn.bias.data.copy_(src_bn.bias.data[:feature_dim])
    target_bn.running_mean.data.copy_(src_bn.running_mean.data[:feature_dim])
    target_bn.running_var.data.copy_(src_bn.running_var.data[:feature_dim])


def get_net_device(net):
    return net.parameters().__next__().device


def get_random_width_and_depth(stem_basic_channel, stage_basic_channels, expand_ratio_list, depth_min=False, depth_max=False):
    width = list()

    stem_channel = make_divisible(round(stem_basic_channel * random.choice(expand_ratio_list)))
    width.append([stem_channel])

    depth = list()
    for idx in range(len(stage_basic_channels)):
        max_value = 8 if idx == 2 else 5

        if depth_min:
            depth.append(2)
        elif depth_max:
            depth.append(max_value)
        else:
            depth.append(random.randint(2, max_value))

    for idx, channel in enumerate(stage_basic_channels):
        width.append([])
        fixed_channel = stem_channel if idx == 0 else \
            make_divisible(round(channel * random.choice(expand_ratio_list)))
        for _ in range(depth[idx]):
            width[-1].append(make_divisible(round(channel * random.choice(expand_ratio_list))))
            width[-1].append(fixed_channel)
    return width, depth


def get_max_channel_per_stage(group_nums):
    assert 1 <= group_nums <= 7

    if group_nums == 1:
        return [[64],
                [128],
                [256],
                [512]]

    if group_nums == 2:
        return [[56, 64],
                [120, 128],
                [240, 256],
                [488, 512]]

    if group_nums == 3:
        return [[48, 56, 64],
                [112, 120, 128],
                [232, 240, 256],
                [464, 488, 512]]

    if group_nums == 4:
        return [[48, 56, 64],
                [104, 112, 120, 128],
                [216, 232, 240, 256],
                [432, 464, 488, 512]]

    if group_nums == 5:
        return [[48, 56, 64],
                [96, 104, 112, 120, 128],
                [208, 216, 232, 240, 256],
                [408, 432, 464, 488, 512]]

    if group_nums == 6:
        return [[48, 56, 64],
                [88, 96, 104, 112, 120, 128],
                [192, 208, 216, 232, 240, 256],
                [384, 408, 432, 464, 488, 512]]

    if group_nums == 7:
        return [[48, 56, 64],
                [88, 96, 104, 112, 120, 128],
                [176, 192, 208, 216, 232, 240, 256],
                [360, 384, 408, 432, 464, 488, 512]]