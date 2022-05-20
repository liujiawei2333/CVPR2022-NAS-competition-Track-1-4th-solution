from torch import nn
import torch
import random

from .dynamic_layers import DynamicBasicBlock, DynamicConvBnActLayer, DynamicLinearLayer
from .nas_static_model import StaticSuperNet
from .nn_utils import get_random_width_and_depth
from .nn_base import MyNetwork

class SuperNet(MyNetwork):
    def __init__(self, n_class, mode, dynamic_resolution, act_func="relu", bn_param=(0., 1e-5)):
        super(SuperNet, self).__init__()
        self.n_class = n_class
        self.mode = mode
        self.divisor = 8
        self.stem_basic_channel = 64
        self.stage_basic_channels = [64, 128, 256, 512]
        self.expand_ratio_list = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
        self.block_nums = [5, 5, 8, 5]
        self.act_func = act_func

        if dynamic_resolution:
            self.resolution_list = [[128,128], [160,160], [192,192],[224,224]]
        elif not dynamic_resolution:
            self.resolution_list = [[224,224]]

        self.first_conv = DynamicConvBnActLayer(in_channel_list=3,
                                                out_channel_list=self.stem_basic_channel,
                                                kernel_size=7, stride=2, act_func=act_func)

        self.first_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block_group_info = []
        blocks = []
        _block_index = 0
        input_channel = self.stem_basic_channel

        for stage_index, repeat_num in enumerate(self.block_nums):

            stage_basic_channel = self.stage_basic_channels[stage_index]
            output_channel = stage_basic_channel

            self.block_group_info.append([_block_index + i for i in range(repeat_num)])
            _block_index += repeat_num

            for block_index in range(repeat_num):
                stride = 2 if (block_index == 0 and stage_index != 0) else 1

                blocks.append(
                    DynamicBasicBlock(in_channel_list=input_channel, out_channel_list=output_channel, stride=stride,
                                      act_func=self.act_func, expand_ratio_list=self.expand_ratio_list)
                )
                input_channel = output_channel

        self.blocks = nn.ModuleList(blocks)

        self.last_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = DynamicLinearLayer(
            in_features_list=input_channel, out_features=self.n_class, bias=True
        )

        if self.mode == "train_supernet":
            self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

        if self.mode == "train_supernet":
            self.zero_residual_block_bn_weights()

        self.active_dropout_rate = 0
        self.active_drop_connect_rate = 0
        self.active_resolution = [224, 224]

    def zero_residual_block_bn_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, DynamicBasicBlock):
                    m.bn2.bn.weight.zero_()
    

    def forward(self, x):
        if x.size(-1) != self.active_resolution[1] or x.size(-2) != self.active_resolution[0]:
            x = torch.nn.functional.interpolate(x, size=self.active_resolution, mode='bicubic', align_corners=True)

        x = self.first_conv(x)
        x = self.first_pool(x)

        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x)

        x = self.last_pool(x)

        if self.active_dropout_rate > 0 and self.training:
            x = torch.nn.functional.dropout(x, p=self.active_dropout_rate)

        x = x.contiguous().view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        # todo
        pass

    @property
    def config(self):
        # todo
        pass

    @staticmethod
    def build_from_config(config):
        # todo
        pass

    def zero_last_gamma(self):
        # todo
        pass

    def set_dropout_rate(self, dropout=0, drop_connect=0, drop_connect_only_last_two_stages=True):
        self.active_dropout_rate = dropout
        for idx, block in enumerate(self.blocks):
            if drop_connect_only_last_two_stages:#走
                if idx not in self.block_group_info[-1] + self.block_group_info[-2]:
                    continue
            this_drop_connect_rate = drop_connect * float(idx) / len(self.blocks)
            block.drop_connect_rate = this_drop_connect_rate

    def set_active_subnet(self, resolution=224, width=None, depth=None):
        assert len(width) - 1 == len(depth)

        self.active_resolution = resolution

        # first conv
        self.first_conv.active_out_channel = width[0][0]

        for stage_id, (c, d) in enumerate(zip(width[1:], depth)):
            start_idx = min(self.block_group_info[stage_id])
            tmp_idx = 0
            for block_id in range(start_idx, start_idx+d):
                block = self.blocks[block_id]

                block.active_middle_channel = c[2 * tmp_idx]
                block.active_out_channel = c[2 * tmp_idx + 1]

                tmp_idx += 1

        # Resnet Blocks repeated times
        for i, d in enumerate(depth):
            self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

    def sample_min_subnet(self):
        return self.sample_active_subnet(min_net=True)


    def sample_max_subnet(self):
        return self.sample_active_subnet(max_net=True)


    def sample_active_subnet(self, min_net=False, max_net=False):
        sample_cfg_for_list = lambda candidates, sample_min, sample_max: \
            candidates[0] if sample_min else (candidates[-1] if sample_max else random.choice(candidates))

        resolution = sample_cfg_for_list(self.resolution_list, min_net, max_net)

        stem_basic_channel = self.stem_basic_channel
        stage_basic_channels = self.stage_basic_channels
        expand_ratio_list = self.expand_ratio_list
        depth_min = False
        depth_max = False
        if min_net:
            expand_ratio_list = [min(expand_ratio_list)] * len(expand_ratio_list)
            depth_min = True

        elif max_net:
            expand_ratio_list = [max(expand_ratio_list)] * len(expand_ratio_list)
            depth_max = True

        width, depth = get_random_width_and_depth(stem_basic_channel, stage_basic_channels,
                                                expand_ratio_list, depth_min, depth_max)

        self.set_active_subnet(resolution=resolution, width=width, depth=depth)


    def get_active_subnet(self, preserve_weight=True):
        with torch.no_grad():
            first_conv = self.first_conv.get_active_subnet(3, preserve_weight)

            blocks = []
            input_channel = first_conv.out_channels
            # blocks
            for stage_id, block_idx in enumerate(self.block_group_info):
                depth = self.runtime_depth[stage_id]
                active_idx = block_idx[:depth]
                stage_blocks = []
                for idx in active_idx:
                    stage_blocks.append(
                        self.blocks[idx].get_active_subnet(input_channel, preserve_weight)
                    )
                    input_channel = self.blocks[idx].active_out_channel
                blocks += stage_blocks

            classifier = self.classifier.get_active_subnet(input_channel, preserve_weight)

            _subnet = StaticSuperNet(first_conv, blocks, classifier, self.active_resolution)
            _subnet.set_bn_param(**self.get_bn_param())
            return _subnet

    def load_weights_from_pretrained_models(self, models_path):
        with open(models_path, 'rb') as f:
            # checkpoint = torch.load(f, map_location='cpu')
            checkpoint = torch.load(f)
        assert isinstance(checkpoint, dict)
        pretrained_state_dicts = checkpoint
        for k, v in self.state_dict().items():
            name = 'module.' + k if not k.startswith('module') else k
            v.copy_(pretrained_state_dicts[name])
