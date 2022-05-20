import torch.nn as nn
import torch

from .nn_base import MyNetwork


class StaticSuperNet(MyNetwork):

    def __init__(self, first_conv, blocks, classifier, resolution):
        super(StaticSuperNet, self).__init__()

        self.first_conv = first_conv
        self.first_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.blocks = nn.ModuleList(blocks)
        self.last_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = classifier
        self.resolution = resolution

    def forward(self, x):

        if isinstance(self.resolution,list):
            if x.size(-1) != self.resolution[1] or x.size(-2) != self.resolution[0]:
                x = torch.nn.functional.interpolate(x, size=self.resolution, mode='bicubic',align_corners=True)

        x = self.first_conv(x)
        x = self.first_pool(x)
        for block in self.blocks:
            x = block(x)
        x = self.last_pool(x)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        for block in self.blocks:
            _str += block.module_str + '\n'
        # _str += self.last_conv.module_str + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': StaticSuperNet.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            # 'last_conv': self.last_conv.config,
            'classifier': self.classifier.config,
            'resolution': self.resolution
        }

    def weight_initialization(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def reset_running_stats_for_calibration(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.SyncBatchNorm):
                m.training = True
                m.momentum = None  # cumulative moving average
                m.reset_running_stats()

    def zero_last_gamma(self):
        raise NotImplementedError