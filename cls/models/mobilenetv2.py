#!/usr/bin/env python

"""
MobileNetV2. https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
"""

import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch import nn
from .utils import load_state_dict_from_url


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        # super(ConvBNReLU, self).__init__(
        #     nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
        #     norm_layer(out_planes),
        #     nn.ReLU6(inplace=True)
        # )
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(*[
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
            ]
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1).conv)
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim).conv,
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            x = x + self.conv(x)
        else:
            x = self.conv(x)
        return x


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16,  1, 1],
                [6, 24,  2, 2],
                [6, 32,  3, 2],
                [6, 64,  4, 1], # NOTE: we switch the stride number of this line and next line setting
                [6, 96,  3, 2],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]

        # [feat1, feat2, feat3, feat4]
        intermediate_features = []

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

    def get_bn_before_relu(self):
        bn1 = self.features[4].conv[1][1]
        bn2 = self.features[7].conv[1][1]
        bn3 = self.features[11].conv[1][1]
        bn4 = self.features[17].conv[1][1]
        return [bn1, bn2, bn3, bn4]

    def get_channel_num(self):
        return [144, 192, 384, 960]

    def extract_feature(self, x, preReLU=False):
        input = x

        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), x.size(1))
        out = self.classifier(x)

        feat1 = self.features[4].conv[1][0:2](
                self.features[4].conv[0](
                self.features[0:4](input)))

        feat2 = self.features[7].conv[1][0:2](
                self.features[7].conv[0](
                self.features[0:7](input)))

        feat3 = self.features[11].conv[1][0:2](
                self.features[11].conv[0](
                self.features[0:11](input)))

        feat4 = self.features[17].conv[1][0:2](
                self.features[17].conv[0](
                self.features[0:17](input)))

        if not preReLU:
            feat1 = F.relu(feat1)
            feat2 = F.relu(feat2)
            feat3 = F.relu(feat3)
            feat4 = F.relu(feat4)

        # we do not use feat2
        return [feat1, feat2, feat3, feat4], out

def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls['mobilenet_v2'],
            progress=progress
        )
        _pretrained_dict = OrderedDict()
        for idx, (k, v) in enumerate(state_dict.items()):
            splitted_k = k.split('.')

            # 0-5, 306-311
            if idx in list(range(0, 6)):
                splitted_k.insert(2, 'conv')

            if idx in list(range(306, 312)):
                splitted_k.insert(2, 'conv')

            if 'classifier' in splitted_k:
                splitted_k[1] = '0'

            _pretrained_dict['.'.join(splitted_k)] = v

        model.load_state_dict(_pretrained_dict)
    return model

