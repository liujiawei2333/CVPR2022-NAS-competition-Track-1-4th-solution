from .nn_utils import int2list, build_activation, copy_bn, get_net_device
from .dynamic_ops import DynamicPointConv2d, DynamicBatchNorm2d, DynamicLinear
from .static_layers import ConvBnActLayer, BasicBlock, LinearLayer
from .nn_base import MyModule


class DynamicBasicBlockFc(MyModule):

    def __init__(self, in_channel_list, out_channel_list, kernel_size_list=[3], stride=1,
                 act_func='relu6', expand_ratio_list=[1]):
        super(DynamicBasicBlockFc, self).__init__()

        self.in_channel_list = int2list(in_channel_list)
        self.out_channel_list = int2list(out_channel_list)

        self.kernel_size_list = int2list(kernel_size_list, 1)
        self.expand_ratio_list = expand_ratio_list

        self.stride = stride
        self.act_func = act_func

        max_out_channel = round(max(self.out_channel_list) * max(self.expand_ratio_list))
        max_middle_channel = max_out_channel

        self.conv1 = DynamicPointConv2d(max(self.in_channel_list), max_middle_channel, stride=self.stride)
        self.bn1 = DynamicBatchNorm2d(max_middle_channel)
        self.act = build_activation(self.act_func, inplace=True)

        self.conv2 = DynamicPointConv2d(max_middle_channel, max_out_channel)
        self.bn2 = DynamicBatchNorm2d(max_out_channel)

        self.short_conv = DynamicPointConv2d(max(self.in_channel_list), max_out_channel, stride=self.stride, kernel_size=1)
        self.short_bn = DynamicBatchNorm2d(max_out_channel)

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)

        self.active_middle_channel = max_middle_channel
        self.active_out_channel = max_out_channel

    def forward(self, x):

        self.conv1.active_out_channel = self.active_middle_channel
        self.conv2.active_out_channel = self.active_out_channel
        self.short_conv.active_out_channel = self.active_out_channel

        identity_x = x
        in_channel = x.size(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if in_channel != self.active_out_channel or self.stride != 1:
            identity_x = self.short_conv(identity_x)
            identity_x = self.short_bn(identity_x)

        x += identity_x

        x = self.act(x)
        return x

    @property
    def module_str(self):
        export_str = "middle channel: {}, out channel:{}".format(self.active_middle_channel,
                                                                                self.active_out_channel)
        return export_str

    @property
    def config(self):
        return {
            'name': DynamicBasicBlock.__name__,
            'in_channel_list': self.in_channel_list,
            'out_channel_list': self.out_channel_list,
            'kernel_size_list': self.kernel_size_list,
            'expand_ratio_list': self.expand_ratio_list,
            'stride': self.stride,
            'act_func': self.act_func,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicBasicBlock(**config)

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):

        # build the new layer
        sub_layer = BasicBlock(in_channel, self.active_middle_channel, self.active_out_channel, self.stride, self.act_func)

        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        sub_layer.conv1.weight.data.copy_(self.conv1.conv.weight.data[:self.active_middle_channel, :in_channel, ...])
        copy_bn(sub_layer.bn1, self.bn1.bn)

        sub_layer.conv2.weight.data.copy_(self.conv2.conv.weight.data[:self.active_out_channel, :self.active_middle_channel, ...])
        copy_bn(sub_layer.bn2, self.bn2.bn)

        if in_channel != self.active_out_channel or self.stride != 1:
            sub_layer.short_conv.weight.data.copy_(self.short_conv.conv.weight.data[:self.active_out_channel, :in_channel, ...])
            copy_bn(sub_layer.short_bn, self.short_bn.bn)
        return sub_layer

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        raise NotImplementedError


class DynamicBasicBlock(MyModule):

    def __init__(self, in_channel_list, middle_channel_list, out_channel_list, kernel_size_list=[3], stride=1,
                act_func='relu6', expand_ratio_list=[1]):
        super(DynamicBasicBlock, self).__init__()

        self.in_channel_list = int2list(in_channel_list)
        self.middle_channel_list = int2list(middle_channel_list)
        self.out_channel_list = int2list(out_channel_list)

        self.kernel_size_list = int2list(kernel_size_list, 1)
        self.expand_ratio_list = expand_ratio_list

        self.stride = stride
        self.act_func = act_func

        max_middle_channel = round(max(self.middle_channel_list) * max(self.expand_ratio_list))
        max_out_channel = round(max(self.out_channel_list) * max(self.expand_ratio_list))

        self.conv1 = DynamicPointConv2d(max(self.in_channel_list), max_middle_channel, stride=self.stride)
        self.bn1 = DynamicBatchNorm2d(max_middle_channel)
        self.act = build_activation(self.act_func, inplace=True)

        self.conv2 = DynamicPointConv2d(max_middle_channel, max_out_channel)
        self.bn2 = DynamicBatchNorm2d(max_out_channel)

        self.short_conv = DynamicPointConv2d(max(self.in_channel_list), max_out_channel, stride=self.stride, kernel_size=1)
        self.short_bn = DynamicBatchNorm2d(max_out_channel)

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)

        self.active_middle_channel = max_middle_channel
        self.active_out_channel = max_out_channel


    def forward(self, x):
        self.conv1.active_out_channel = self.active_middle_channel
        self.conv2.active_out_channel = self.active_out_channel
        self.short_conv.active_out_channel = self.active_out_channel

        identity_x = x
        in_channel = x.size(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if in_channel != self.active_out_channel or self.stride != 1:
            identity_x = self.short_conv(identity_x)
            identity_x = self.short_bn(identity_x)

        x += identity_x
        x = self.act(x)
        return x

    @property
    def module_str(self):
        export_str = "middle channel: {}, out channel:{}".format(self.active_middle_channel,
                                                                                self.active_out_channel)
        return export_str

    @property
    def config(self):
        return {
            'name': DynamicBasicBlock.__name__,
            'in_channel_list': self.in_channel_list,
            'out_channel_list': self.out_channel_list,
            'kernel_size_list': self.kernel_size_list,
            'expand_ratio_list': self.expand_ratio_list,
            'stride': self.stride,
            'act_func': self.act_func,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicBasicBlock(**config)

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):

        # build the new layer
        sub_layer = BasicBlock(in_channel, self.active_middle_channel, self.active_out_channel, self.stride, self.act_func)

        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        sub_layer.conv1.weight.data.copy_(self.conv1.conv.weight.data[:self.active_middle_channel, :in_channel, ...])
        copy_bn(sub_layer.bn1, self.bn1.bn)

        sub_layer.conv2.weight.data.copy_(self.conv2.conv.weight.data[:self.active_out_channel, :self.active_middle_channel, ...])
        copy_bn(sub_layer.bn2, self.bn2.bn)

        if in_channel != self.active_out_channel or self.stride != 1:
            sub_layer.short_conv.weight.data.copy_(self.short_conv.conv.weight.data[:self.active_out_channel, :in_channel, ...])
            copy_bn(sub_layer.short_bn, self.short_bn.bn)
        return sub_layer

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        raise NotImplementedError


class DynamicConvBnActLayer(MyModule):

    def __init__(self, in_channel_list, out_channel_list, kernel_size=3, stride=1, dilation=1,
                use_bn=True, act_func='relu6'):
        super(DynamicConvBnActLayer, self).__init__()

        self.in_channel_list = int2list(in_channel_list)
        self.out_channel_list = int2list(out_channel_list)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func

        self.conv = DynamicPointConv2d(
            max_in_channels=max(self.in_channel_list), max_out_channels=max(self.out_channel_list),
            kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation,
        )
        if self.use_bn:
            self.bn = DynamicBatchNorm2d(max(self.out_channel_list))

        if self.act_func is not None:
            self.act = build_activation(self.act_func, inplace=True)

        self.active_out_channel = max(self.out_channel_list)


    def forward(self, x):
        self.conv.active_out_channel = self.active_out_channel

        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)

        if self.act_func is not None:
            x = self.act(x)
        return x

    @property
    def module_str(self):
        return 'DyConv(O%d, K%d, S%d)' % (self.active_out_channel, self.kernel_size, self.stride)

    @property
    def config(self):
        return {
            'name': DynamicConvBnActLayer.__name__,
            'in_channel_list': self.in_channel_list,
            'out_channel_list': self.out_channel_list,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicConvBnActLayer(**config)

    def get_active_subnet(self, in_channel, preserve_weight=True):
        sub_layer = ConvBnActLayer(
            in_channel, self.active_out_channel, self.kernel_size, self.stride, self.dilation,
            use_bn=self.use_bn, act_func=self.act_func
        )
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        sub_layer.conv.weight.data.copy_(self.conv.conv.weight.data[:self.active_out_channel, :in_channel, :, :])
        if self.use_bn:
            copy_bn(sub_layer.bn, self.bn.bn)

        return sub_layer


class DynamicLinearLayer(MyModule):

    def __init__(self, in_features_list, out_features, bias=True):
        super(DynamicLinearLayer, self).__init__()

        self.in_features_list = int2list(in_features_list)
        self.out_features = out_features
        self.bias = bias
        self.linear = DynamicLinear(
            max_in_features=max(self.in_features_list), max_out_features=self.out_features, bias=self.bias
        )

    def forward(self, x):
        return self.linear(x)

    @property
    def module_str(self):
        return 'DyLinear(%d)' % self.out_features

    @property
    def config(self):
        return {
            'name': DynamicLinear.__name__,
            'in_features_list': self.in_features_list,
            'out_features': self.out_features,
            'bias': self.bias
        }

    @staticmethod
    def build_from_config(config):
        return DynamicLinearLayer(**config)

    def get_active_subnet(self, in_features, preserve_weight=True):
        # sub_layer = LinearLayer(in_features, self.out_features, self.bias, dropout_rate=self.dropout_rate)
        sub_layer = LinearLayer(in_features, self.out_features, self.bias)
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        sub_layer.linear.weight.data.copy_(self.linear.linear.weight.data[:self.out_features, :in_features])
        if self.bias:
            sub_layer.linear.bias.data.copy_(self.linear.linear.bias.data[:self.out_features])
        return sub_layer
