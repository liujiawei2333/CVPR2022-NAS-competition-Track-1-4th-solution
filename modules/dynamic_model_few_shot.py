from collections import OrderedDict
from torch import nn
import torch
import random

from .nn_utils import get_random_width_and_depth, get_max_channel_per_stage
from .dynamic_layers_few_shot import DynamicBasicBlock, DynamicConvBnActLayer, DynamicLinearLayer
from .nas_static_model import StaticSuperNet
from .nn_base import MyNetwork


class SuperNetDependTotal(MyNetwork):
    def __init__(self, n_class, mode, dynamic_resolution, split_group=2, act_func="relu", bn_param=(0., 1e-5)):
        super(SuperNetDependTotal, self).__init__()
        self.n_class = n_class
        self.mode = mode
        self.divisor = 8
        self.stem_basic_channel = 64
        self.stage_basic_channels = [64, 128, 256, 512]
        self.expand_ratio_list = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
        self.block_nums = [5, 5, 8, 5]
        self.stage_idx_layer_idx = {0: [0, 1, 2, 3, 4],
                                    1: [5, 6, 7, 8, 9],
                                    2: [10, 11, 12, 13, 14, 15, 16, 17],
                                    3: [18, 19, 20, 21, 22]
                                    }
        self.act_func = act_func

        if dynamic_resolution:
            self.resolution_list = [(128, 128), (160, 160), (192, 192), (224, 224)]
        elif not dynamic_resolution:
            self.resolution_list = [(224,224)]

        self.split_group = split_group
        self.choice_max_channel = get_max_channel_per_stage(self.split_group)

        self.first_conv = DynamicConvBnActLayer(in_channel_list=3,
                                                out_channel_list=self.stem_basic_channel,
                                                kernel_size=7, stride=2, act_func=act_func)

        self.first_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block_group_info = []
        self.blocks = torch.nn.ModuleList()
        self.active_stage_group_index = [0] * len(self.block_nums)

        # _block_index = 0
        input_channel = self.stem_basic_channel

        for stage_index, repeat_num in enumerate(self.block_nums):

            self.blocks.append(torch.nn.ModuleList())

            base_channel = self.stage_basic_channels[stage_index]
            output_channel = self.stage_basic_channels[stage_index]

            cur_stage_max_channel_per_group = self.choice_max_channel[stage_index]

            for max_channel in cur_stage_max_channel_per_group:
                cur_blocks = self.make_fix_channel_block(stage_index, input_channel, base_channel,
                                                        max_channel, repeat_num)
                self.blocks[-1].append(cur_blocks)
            input_channel = output_channel

            self.block_group_info.append([i for i in range(repeat_num)])

            # self.block_group_info.append([_block_index + i for i in range(repeat_num)])
            # _block_index += repeat_num

        self.last_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier_groups = torch.nn.ModuleList()
        for fc_input_max_channel in self.choice_max_channel[-1]:
            self.classifier_groups.append(DynamicLinearLayer(
                in_features_list=fc_input_max_channel, out_features=self.n_class, bias=True))

        if self.mode == "train_supernet":
            self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

        if self.mode == "train_supernet":
            self.zero_residual_block_bn_weights()

        self.active_dropout_rate = 0
        self.active_drop_connect_rate = 0
        self.active_resolution = (224, 224)

        # self.block_group_info:[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4]]


    def make_fix_channel_block(self, stage_index, input_channel, middle_channel, output_channel, repeat_num):
        blocks = torch.nn.ModuleList()

        for block_index in range(repeat_num):
            stride = 2 if (block_index == 0 and stage_index != 0) else 1

            blocks.append(
                DynamicBasicBlock(in_channel_list=input_channel, middle_channel_list=middle_channel,
                                out_channel_list=output_channel, stride=stride,
                                act_func=self.act_func, expand_ratio_list=self.expand_ratio_list))
            input_channel = output_channel

        return blocks


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

        cur_stage_group_index = None
        for stage_id, block_idx in enumerate(self.block_group_info):

            cur_stage_group_index = self.active_stage_group_index[stage_id]

            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[stage_id][cur_stage_group_index][idx](x)

        x = self.last_pool(x)
        if self.active_dropout_rate > 0 and self.training:
            x = torch.nn.functional.dropout(x, p=self.active_dropout_rate)

        x = x.contiguous().view(x.shape[0], -1)
        cur_fc_group_index = cur_stage_group_index
        x = self.classifier_groups[cur_fc_group_index](x)
        return x


    def set_dropout_rate(self, dropout=0, drop_connect=0, drop_connect_only_last_two_stages=True):
        self.active_dropout_rate = dropout

        if drop_connect_only_last_two_stages:
            stage_idx_list = [-2, -1]
        else:
            stage_idx_list = list(range(len(self.block_nums)))

        for stage_idx in stage_idx_list:
            stage_block = self.blocks[stage_idx]
            start_idx = sum(self.block_nums[:stage_idx])
            for group_index, group_block in enumerate(stage_block):
                for cur_layer_index, cur_layer in enumerate(group_block):
                    idx = cur_layer_index + start_idx
                    this_drop_connect_rate = drop_connect * float(idx) / sum(self.block_nums)
                    cur_layer.drop_connect_rate = this_drop_connect_rate


    def set_active_subnet(self, resolution=(224, 224), width=None, depth=None):
        assert len(width) - 1 == len(depth)

        self.active_resolution = resolution

        # first conv
        self.first_conv.active_out_channel = width[0][0]

        for stage_id, (c, d) in enumerate(zip(width[1:], depth)):
            start_idx = min(self.block_group_info[stage_id])
            tmp_idx = 0

            cur_stage_output_channel = c[-1]

            cur_max_channel_list = self.choice_max_channel[stage_id]
            cur_stage_group_index = None
            for group_index, cur_max_channel in enumerate(cur_max_channel_list):
                if cur_stage_output_channel <= cur_max_channel:
                    cur_stage_group_index = group_index
                    self.active_stage_group_index[stage_id] = group_index
                    break

            for block_id in range(start_idx, start_idx + d):
                block = self.blocks[stage_id][cur_stage_group_index][block_id]

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


    def sample_active_subnet(self, min_net=False, max_net=False, result_flag=False):
        def _sample_resolution(candidates, sample_min, sample_max):
            if sample_min:
                return candidates[0]
            if sample_max:
                return candidates[-1]
            return random.choice(candidates)

        resolution = _sample_resolution(self.resolution_list, min_net, max_net)

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
        if result_flag:
            return width, depth


    def get_active_subnet(self, preserve_weight=True):

        with torch.no_grad():
            first_conv = self.first_conv.get_active_subnet(3, preserve_weight)

            blocks = []
            input_channel = first_conv.out_channels

            cur_stage_group_index = None

            # blocks
            for stage_id, block_idx in enumerate(self.block_group_info):

                cur_stage_group_index = self.active_stage_group_index[stage_id]
                depth = self.runtime_depth[stage_id]
                active_idx = block_idx[:depth]
                stage_blocks = []
                for idx in active_idx:
                    stage_blocks.append(
                        self.blocks[stage_id][cur_stage_group_index][idx].get_active_subnet(input_channel,
                                                                                            preserve_weight)
                    )
                    input_channel = self.blocks[stage_id][cur_stage_group_index][idx].active_out_channel
                blocks += stage_blocks

            cur_fc_group_index = cur_stage_group_index
            classifier = self.classifier_groups[cur_fc_group_index]

            _subnet = StaticSuperNet(first_conv, blocks, classifier, self.active_resolution)
            _subnet.set_bn_param(**self.get_bn_param())
            return _subnet


    def init_weights_from_one(self, model_path):
        update_dict = self._get_valid_state_dict(model_path)

        for k, v in update_dict.items():
            name = k.replace("module.", "") if "module." in k else k
        
            if "first_conv" in name:
                name_list = name.split(".")
                if "conv" == name_list[1]:
                    self.first_conv.conv.conv.weight.data.copy_(v)
                else:
                    bn_val_name = name_list[-1]
                    first_conv_bn_obj = getattr(self.first_conv.bn.bn, bn_val_name)
                    first_conv_bn_obj.data.copy_(v)
                    setattr(self.first_conv.bn.bn, bn_val_name, first_conv_bn_obj)

            elif "blocks" in name:
                name_list = name.split(".")
                layer_idx = int(name_list[1])
                stage_idx = None
                for key, value in self.stage_idx_layer_idx.items():
                    if layer_idx in value:
                        stage_idx = key
                        break

                idx = self.stage_idx_layer_idx[stage_idx].index(layer_idx)
                for group_index, max_channel in enumerate(self.choice_max_channel[stage_idx]):
                    if "conv" in name_list:
                        conv_name = name_list[2]
                        conv_layer = getattr(self.blocks[stage_idx][group_index][idx], conv_name)
                        if idx == 0:
                            if "conv2" in name or "short_conv" in name:
                                conv_layer.conv.weight.data.copy_(v[:max_channel, ...])
                            else:
                                conv_layer.conv.weight.data.copy_(v)
                        else:
                            if "conv1" in name:
                                conv_layer.conv.weight.data.copy_(v[:, :max_channel, ...])
                            elif "conv2" in name:
                                conv_layer.conv.weight.data.copy_(v[:max_channel, ...])
                            else:
                                conv_layer.conv.weight.data.copy_(v[:max_channel, :max_channel, ...])

                        setattr(self.blocks[stage_idx][group_index][idx], conv_name, conv_layer)

                    else:

                        bn_name = name_list[2]  # bn1, bn2, short_bn
                        bn_val_name = name_list[-1]  # weight, bias, running_mean, running_var

                        bn_obj = getattr(self.blocks[stage_idx][group_index][idx], bn_name)
                        bn_layer = getattr(bn_obj.bn, bn_val_name)

                        if "bn2" in name or "short_bn" in name:
                            bn_layer.data.copy_(v[:max_channel])
                        else:
                            bn_layer.data.copy_(v)

                        setattr(bn_obj.bn, bn_val_name, bn_layer)
                        setattr(self.blocks[stage_idx][group_index][idx], bn_name, bn_obj)

            elif "classifier" in name:

                for fc_group_index, fc_in_dim in enumerate(self.choice_max_channel[-1]):
                    if "weight" in name:
                        self.classifier_groups[fc_group_index].linear.linear.weight.data.copy_(v[:, :fc_in_dim])
                    else:
                        self.classifier_groups[fc_group_index].linear.linear.bias.data.copy_(v)


    def init_weights(self, model_path, root_split_groups):
        assert root_split_groups < self.split_group, \
            "the groups of source model must smaller than ones of the current model!"
        update_dict = self._get_valid_state_dict(model_path)

        target_start_group_idx_per_stage = [0] * len(self.block_nums)
        root_choice_max_channel = get_max_channel_per_stage(root_split_groups)
        cur_block_group_idx = 0
        target_start_group_idx = 0

        target_start_group_idx_in_fc = 0
        cur_target_start_group_idx_in_fc = 0
        cur_fc_group_idx = 0

        for k, v in update_dict.items():
            name = k.replace("module.", "") if "module." in k else k

            if "first_conv" in name:
                name_list = name.split(".")
                if "conv" == name_list[1]:
                    self.first_conv.conv.conv.weight.data.copy_(v)
                else:
                    bn_val_name = name_list[-1]
                    first_conv_bn_obj = getattr(self.first_conv.bn.bn, bn_val_name)
                    first_conv_bn_obj.data.copy_(v)
                    setattr(self.first_conv.bn.bn, bn_val_name, first_conv_bn_obj)

            elif "blocks" in name:
                name_list = name.split(".")

                source_stage_idx = int(name_list[1])
                source_group_idx = int(name_list[2])
                source_layer_idx = int(name_list[3])

                target_stage_idx = source_stage_idx
                target_layer_idx = source_layer_idx
                target_max_channel_list = self.choice_max_channel[target_stage_idx]

                if cur_block_group_idx != source_group_idx:
                    cur_block_group_idx = source_group_idx
                    target_start_group_idx = target_start_group_idx_per_stage[target_stage_idx]

                for target_group_index in range(target_start_group_idx, len(target_max_channel_list)):
                    max_channel = target_max_channel_list[target_group_index]
                    if root_choice_max_channel[source_stage_idx][source_group_idx] < max_channel:
                        target_start_group_idx_per_stage[target_stage_idx] = target_group_index
                        break

                    # layer name: blocks.0.0.0.conv1.conv.weight
                    if "conv" in name_list:
                        conv_name = name_list[4]
                        conv_layer = getattr(self.blocks[target_stage_idx][target_group_index][target_layer_idx],
                                            conv_name)
                        if target_layer_idx == 0:
                            if "conv2" in name or "short_conv" in name:
                                conv_layer.conv.weight.data.copy_(v[:max_channel, ...])
                            else:
                                conv_layer.conv.weight.data.copy_(v)
                        else:
                            if "conv1" in name:
                                conv_layer.conv.weight.data.copy_(v[:, :max_channel, ...])
                            elif "conv2" in name:
                                conv_layer.conv.weight.data.copy_(v[:max_channel, ...])
                            else:
                                conv_layer.conv.weight.data.copy_(v[:max_channel, :max_channel, ...])

                        setattr(self.blocks[target_stage_idx][target_group_index][target_layer_idx],
                                conv_name, conv_layer)

                    else:
                        # layer name: blocks.0.0.0.bn1.bn.weight
                        bn_name = name_list[4]  # bn1, bn2, short_bn
                        bn_val_name = name_list[-1]  # weight, bias, running_mean, running_var

                        bn_obj = getattr(self.blocks[target_stage_idx][target_group_index][target_layer_idx], bn_name)
                        bn_layer = getattr(bn_obj.bn, bn_val_name)

                        if "bn2" in name or "short_bn" in name:
                            bn_layer.data.copy_(v[:max_channel])
                        else:
                            bn_layer.data.copy_(v)

                        setattr(bn_obj.bn, bn_val_name, bn_layer)
                        setattr(self.blocks[target_stage_idx][target_group_index][target_layer_idx], bn_name, bn_obj)

            elif "classifier" in name:
                #  layer name: classifier_groups.1.linear.linear.weight
                name_list = name.split(".")
                source_group_index = int(name_list[1])
                source_fc_in_dim_list = root_choice_max_channel[-1]
                target_fc_in_dim_list = self.choice_max_channel[-1]

                if cur_fc_group_idx != source_group_index:
                    cur_fc_group_idx = source_group_index
                    target_start_group_idx_in_fc = cur_target_start_group_idx_in_fc

                for fc_group_index in range(target_start_group_idx_in_fc, len(target_fc_in_dim_list)):
                    if source_fc_in_dim_list[source_group_index] < target_fc_in_dim_list[fc_group_index]:
                        cur_target_start_group_idx_in_fc = fc_group_index
                        break

                    fc_in_dim = target_fc_in_dim_list[fc_group_index]

                    if "weight" in name:
                        self.classifier_groups[fc_group_index].linear.linear.weight.data.copy_(v[:, :fc_in_dim])
                    else:
                        self.classifier_groups[fc_group_index].linear.linear.bias.data.copy_(v)


    def _get_valid_state_dict(self, model_path):

        new_dict = torch.load(model_path, map_location="cpu" if not torch.cuda.is_available() else "cuda")
        update_dict = OrderedDict()

        for key, value in new_dict.items():
            if "num_batches_tracked" in key or "bn_tracking" in key:
                continue
            # str2txt(key, "model_name.txt")
            update_dict[key] = value

        return update_dict

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
