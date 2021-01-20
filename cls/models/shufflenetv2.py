#!/usr/bin/env python

"""
ShuffleNetV2. https://github.com/pytorch/vision/blob/master/torchvision/models/shufflenetv2.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from collections import OrderedDict

__all__ = [
    'ShuffleNetV2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
]

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, pre_act=True):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            if pre_act:
                self.branch1 = nn.Sequential(
                    nn.ReLU(inplace=False),
                    self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                    nn.BatchNorm2d(inp),
                    nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(branch_features),
                )
            else:
                self.branch1 = nn.Sequential(
                    self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                    nn.BatchNorm2d(inp),
                    nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(branch_features),
                )
        else:
            self.branch1 = nn.Sequential()

        if pre_act:
            self.branch2 = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(inp if (self.stride > 1) else branch_features,
                          branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
            )
        else:
            self.branch2 = nn.Sequential(
                nn.Conv2d(inp if (self.stride > 1) else branch_features,
                          branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
            )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000, inverted_residual=InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            pre_act = False if name == 'stage2' else True
            seq = [inverted_residual(input_channels, output_channels, 2, pre_act=pre_act)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)  # 56 x 56
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.stage2(x) # 28 x 28
        x = self.stage3(x) # 14 x 14
        x = self.stage4(x) # 7 x 7
        x = self.conv5(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), x.size(1))

        x = self.fc(x)
        return x

    def get_bn_before_relu(self):
        bn1 = self.conv1[-1]
        bn2 = self.stage2[-1].branch2[-1]
        bn3 = self.stage3[-1].branch2[-1]
        bn4 = self.stage4[-1].branch2[-1]
        return [bn1, bn2, bn3, bn4]

    def get_channel_num(self):
        return [24, 116, 232, 464]

    def extract_feature(self, x, preReLU=False):
        x = self.conv1(x)
        x = F.relu(x)

        feat1 = self.maxpool(x)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)

        x = self.conv5(feat4)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.squeeze(3).squeeze(2)
        out = self.fc(x)

        if not preReLU:
            feat1 = F.relu(feat1)
            feat2 = F.relu(feat2)
            feat3 = F.relu(feat3)
            feat4 = F.relu(feat4)

        return [feat1, feat2, feat3, feat4], out


def _shufflenetv2(arch, pretrained, progress, *args, **kwargs):
    model = ShuffleNetV2(*args, **kwargs)

    if pretrained:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            _pretrained_dict = OrderedDict()
            for idx, (k, v) in enumerate(state_dict.items()):
                splitted_k = k.split('.')
                # special for 1.0x
                if 29 < idx < 280:
                    splitted_k[-2] = str(int(splitted_k[-2])+1)
                    _pretrained_dict['.'.join(splitted_k)] = v
                else:
                    _pretrained_dict[k] = v

            model.load_state_dict(_pretrained_dict)
            # release
            del _pretrained_dict
            del state_dict
    return model

def shufflenet_v2_x0_5(pretrained=False, progress=True, **kwargs):
    return _shufflenetv2('shufflenetv2_x0.5', pretrained, progress,
                         [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)

def shufflenet_v2_x1_0(pretrained=False, progress=True, **kwargs):
    return _shufflenetv2('shufflenetv2_x1.0', pretrained, progress,
                         [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)

if __name__ == '__main__':
    net = shufflenet_v2_x1_0(pretrained=True)
    print(net)
    print(net.get_bn_before_relu())
    input = torch.Tensor(1, 3, 224, 224)
    out = net(input)
    # grad check
    torch.autograd.backward(out.sum(), grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None)

