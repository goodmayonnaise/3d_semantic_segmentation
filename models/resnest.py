import math
from fvcore.common.registry import Registry
from modules import Concatenate
from models.segnext import SpatialAttention

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Module, ReLU


RESNEST_MODELS_REGISTRY = Registry('RESNEST_MODELS')

def get_model(model_name):
    return RESNEST_MODELS_REGISTRY.get(model_name)

class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel//self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited) 
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel//self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


__all__ = ['ResNet', 'Bottleneck']

_url_format = 'https://s3.us-west-1.wasabisys.com/resnest/torch/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ]}


def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)

class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False,
                 norm_layer=None, dropblock_prob=0.0, last_gamma=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False) # 128 --> 256
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, rectify=rectified_conv,
                rectify_avg=rectify_avg,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob)
        elif rectified_conv:
            from rfconv import RFConv2d
            self.conv2 = RFConv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False,
                average_mode=rectify_avg)
            self.bn2 = norm_layer(group_width)
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeSt(nn.Module):
    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64,
                 num_classes=1000, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 rectified_conv=False, rectify_avg=False,
                 avd=False, avd_first=False,
                 final_drop=0.0, dropblock_prob=0,
                 last_gamma=False, norm_layer=nn.BatchNorm2d):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        super(ResNeSt, self).__init__()
        self.rectified_conv = rectified_conv

        self.rectify_avg = rectify_avg
        if rectified_conv:
            from rfconv import RFConv2d
            conv_layer = RFConv2d
        else:
            conv_layer = nn.Conv2d
        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}
        if deep_stem:
            self.conv1 = nn.Sequential(
                # SpatialAttention(d_model=3),
                conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False, **conv_kwargs),
                # SpatialAttention(d_model=stem_width),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                # SpatialAttention(d_model=stem_width),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
                # SpatialAttention(d_model=stem_width),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                # SpatialAttention(d_model=stem_width),
                conv_layer(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs)
                # SpatialAttention(d_model=stem_width*2)
            )
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False, **conv_kwargs)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        elif dilation==2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        self.avgpool = GlobalAvgPool2d()
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                # if planes == 256:
                #     down_layers.append(SpatialAttention(d_model=self.inplanes))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
                # if planes == 64:
                #     down_layers.append(SpatialAttention(d_model=planes * block.expansion))
            else:
                # if planes == 256:
                #     down_layers.append(SpatialAttention(d_model=self.inplanes))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
                # if planes == 64:
                #     down_layers.append(SpatialAttention(d_model=planes * block.expansion))
                                             
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=2, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x1 = self.relu(x)

        # x2 = self.maxpool(x1)
        # x2 = self.layer1(x2)

        # x3 = self.layer2(x2)

        # x4 = self.layer3(x3)

        # x5 = self.layer4(x4)

        return 

class Fusion_ResNeSt(nn.Module):
    # pylint: disable=unused-variable
    def __init__(self, block, layers, fusion_lev, radix=1, groups=1, bottleneck_width=64, 
                 num_classes=20, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 rectified_conv=False, rectify_avg=False,
                 avd=False, avd_first=False,
                 final_drop=0.0, dropblock_prob=0,
                 last_gamma=False, norm_layer=nn.BatchNorm2d):
        
        self.fusion_lev = fusion_lev

        self.deep_stem = deep_stem
        self.stem_width = stem_width
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        self.norm_layer = norm_layer

        self.cardinality = groups
        self.bottleneck_width = bottleneck_width

        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg

        super(Fusion_ResNeSt, self).__init__()
        self.concat = Concatenate(axis=1)
        if rectified_conv:
            from rfconv import RFConv2d
            conv_layer = RFConv2d
        else:
            conv_layer = nn.Conv2d

        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}
        
        if deep_stem:
            if self.fusion_lev == "none":
                self.conv1 = nn.Sequential(
                                            conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False, **conv_kwargs),
                                            norm_layer(stem_width),
                                            nn.ReLU(inplace=True),
                                            conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
                                            norm_layer(stem_width),
                                            nn.ReLU(inplace=True),
                                            conv_layer(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
                                          )                
            elif self.fusion_lev == "early":
                self.conv1_fusion = nn.Sequential(
                                                    conv_layer(4, stem_width, kernel_size=3, stride=2, padding=1, bias=False, **conv_kwargs),
                                                    norm_layer(stem_width),
                                                    nn.ReLU(inplace=True),
                                                    conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
                                                    norm_layer(stem_width),
                                                    nn.ReLU(inplace=True),
                                                    conv_layer(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
                                                )
            else:
                self.conv1_rgb = nn.Sequential(
                                                conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False, **conv_kwargs),
                                                norm_layer(stem_width),
                                                nn.ReLU(inplace=True),
                                                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
                                                norm_layer(stem_width),
                                                nn.ReLU(inplace=True),
                                                conv_layer(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
                                            )
                self.conv1_rem = nn.Sequential(
                                                conv_layer(1, stem_width, kernel_size=3, stride=2, padding=1, bias=False, **conv_kwargs),
                                                norm_layer(stem_width),
                                                nn.ReLU(inplace=True),
                                                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
                                                norm_layer(stem_width),
                                                nn.ReLU(inplace=True),
                                                conv_layer(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
                                            )
        else:
            if self.fusion_lev == "early":
                self.conv1_fusion = conv_layer(4, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False, **conv_kwargs)
            elif self.fusion_lev == "none":
                self.conv1_fusion = conv_layer(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False, **conv_kwargs)
            else:
                self.conv1_rgb = conv_layer(3, 64, kernel_size=7, stride=2, padding=3,
                                            bias=False, **conv_kwargs)
                self.conv1_rem = conv_layer(1, 64, kernel_size=7, stride=2, padding=3,
                                            bias=False, **conv_kwargs)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).cuda()

        if self.fusion_lev == "mid_stage1":
            self.inplanes *= 2
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)

        # if self.fusion_lev == "mid_stage2":
        #     self.inplanes *= 2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        # self.layer2 = self._make_layer(block, 256, layers[1], stride=2, norm_layer=norm_layer)

        if self.fusion_lev == "mid_stage3":
            self.inplanes *= 2
        
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        elif dilation==2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        self.avgpool = GlobalAvgPool2d()
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)
        layers = []
        if dilation == 1 or dilation == 2: # self.inplanes 128  inplanes 128
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=2, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))

        return nn.Sequential(*layers)

    def forward(self, rgb, rem): # ------------------------------------------------------ origin ------------------------------------------------------------------------
        if self.fusion_lev == "none":
            x = self.conv1(x)
            x = self.bn1(x)
            x0 = self.relu(x)

            x1 = self.maxpool(x0)
            x1 = self.layer1(x1)

            x2 = self.maxpool(x1)

            return x0, x1, x2

        elif self.fusion_lev == "early": 
                                                       #   C,   W,   H    size
            x = self.concat(rgb, rem)                  #   4,  384, 1248

            x = self.conv1_fusion(x)                   #  64,  192, 624      
            x = self.bn1(x)                            #  64,  192, 624  
            x0 = self.relu(x)                          #  64,  192, 624   1/2

            x1 = self.maxpool(x0)                      #  64,   96, 312   

            # stage 1~4 features                              
            x1 = self.layer1(x1)                       # 256,   96, 312   1/4
            x2 = self.layer2(x1)                       # 512,   48, 156   1/8
            # x3 = self.layer3(x2)                       # 1024,  24, 78    1/16
            # x4 = self.layer4(x3)                       # 2048,  12, 39    1/32

            return x0, x1, x2

        elif self.fusion_lev == "mid_stage1": # size 1/2 output fusion 
                                                       #    C    W  H       size
            # For RGB                                        3  384 1248    1
            x_rgb = self.conv1_rgb(rgb)                #    64  192 624
            x_rgb = self.bn1(x_rgb)                    #    64  192 624
            x0_rgb = self.relu(x_rgb)                  #    64  192 624     1/2 

            # For Remission 
            x_rem = self.conv1_rem(rem)
            x_rem = self.bn1(x_rem)
            x0_rem = self.relu(x_rem)
            
            # mid stage 1 (size 1/2) fusion
            x0 = self.concat(x0_rgb, x0_rem)          # 128, 192, 624       1/2
            
            # stage 1~4
            x1 = self.maxpool(x0)                     # 128 192 624
            x1 = self.layer1(x1)                      # 256, 96,  312    1/4
            x2 = self.layer2(x1)                      # 512, 48,  156    1/8
            # x3 = self.layer3(x2)                     
            # x4 = self.layer4(x3)                     
            return x0, x1, x2

        # size 1/4 output fusion 
        elif self.fusion_lev == "mid_stage2":           # C     W       H       size
            # For RGB                                   # 3     96      320     1
            x_rgb = self.conv1_rgb(rgb)                 # 64    48      160
            x_rgb = self.bn1(x_rgb)                     # 64    48      160
            x1_rgb = self.relu(x_rgb)                   # 64    48      160     1/2
            x2_rgb = self.maxpool(x1_rgb)               # 64    24      80
            x2_rgb = self.layer1(x2_rgb)                # 256   24      80      1/4
            x3_rgb = self.layer2(x2_rgb)                # 512   12      40      1/8
            x4_rgb = self.layer3(x3_rgb)                # 1024  6       20      1/16

            # For Remission 
            x_rem = self.conv1_rem(rem)
            x_rem = self.bn1(x_rem)
            x1_rem = self.relu(x_rem)        
            x2_rem = self.maxpool(x1_rem)
            x2_rem = self.layer1(x2_rem)    
            x3_rem = self.layer2(x2_rem)
            x4_rem = self.layer3(x3_rem)


            x1 = self.concat(x1_rgb, x1_rem)          # 128     48      160     1/2
            x2 = self.concat(x2_rgb, x2_rem)          # 512     24      80      1/4
            x3 = self.concat(x3_rgb, x3_rem)          # 1024    12      40      1/8
            x4 = self.concat(x4_rgb, x4_rem)          # 2048    6       20      1/16
            return x1, x2, x3, x4


            # # stage3
            # x3 = self.layer3(x2)                     # 1024, 24,  78    1/16    
            # # # stage4
            # x4 = self.layer4(x3)                     # 2048, 12,  39    1/32
            

        elif self.fusion_lev == "mid_stage3": # size 1/8 output fusion 
                                                       # C    W     H           size
            # For RGB                                      3    384    1248     1
            x_rgb = self.conv1_rgb(rgb)                 # 64    192     624
            x_rgb = self.bn1(x_rgb)                     # 64    192     624
            x0_rgb = self.relu(x_rgb)                   # 64    192     624     1/2
            x1_rgb = self.maxpool(x0_rgb)               # 64    96      312
            x1_rgb = self.layer1(x1_rgb)                # 256   96      312     1/4 
            x2_rgb = self.layer2(x1_rgb)                # 512   48      156     1/8

            # For Remission 
            x_rem = self.conv1_rem(rem)
            x_rem = self.bn1(x_rem)
            x0_rem = self.relu(x_rem)        
            x1_rem = self.maxpool(x0_rem)
            x1_rem = self.layer1(x1_rem)    
            x2_rem = self.layer2(x1_rem)


            x0 = self.concat(x0_rgb, x0_rem)            # 128   192     624     1/2
            x1 = self.concat(x1_rgb, x1_rem)            # 512   96      312     1/4
            x2 = self.concat(x2_rgb, x2_rem)            #1024   48      156     1/8

            # # stage3
            # x3 = self.layer3(x2)                     # 1024, 24,  78    1/16    
            # # # stage4
            # x4 = self.layer4(x3)                     # 2048, 12,  39    1/32
            return x0, x1, x2

        elif self.fusion_lev == "mid_stage4": # size 1/8 output fusion 
                                                       # C    W     H           size
            # For RGB                                      3    384    1248     1
            x_rgb = self.conv1_rgb(rgb)                 # 64    192     624
            x_rgb = self.bn1(x_rgb)                     # 64    192     624
            x0_rgb = self.relu(x_rgb)                   # 64    192     624     1/2
            x1_rgb = self.maxpool(x0_rgb)               # 64    96      312
            x1_rgb = self.layer1(x1_rgb)                # 256   96      312     1/4 
            x2_rgb = self.layer2(x1_rgb)                # 512   48      156     1/8

            # For Remission 
            x_rem = self.conv1_rem(rem)
            x_rem = self.bn1(x_rem)
            x0_rem = self.relu(x_rem)        
            x1_rem = self.maxpool(x0_rem)
            x1_rem = self.layer1(x1_rem)    
            x2_rem = self.layer2(x1_rem)


            x0 = self.concat(x0_rgb, x0_rem)            # 128   192     624     1/2
            x1 = self.concat(x1_rgb, x1_rem)            # 512   96      312     1/4
            x2 = self.concat(x2_rgb, x2_rem)            #1024   48      156     1/8

            return x0, x1, x2, x2_rgb, x2_rem # 1/2, 1/4, 1/8

        elif self.fusion_lev == "late": # size 1/8 output fusion 
                                                       # C    W     H           size
            # For RGB                                      3    384    1248     1
            x_rgb = self.conv1_rgb(rgb)                 # 64    192     624
            x_rgb = self.bn1(x_rgb)                     # 64    192     624
            x0_rgb = self.relu(x_rgb)                   # 64    192     624     1/2
            x1_rgb = self.maxpool(x0_rgb)               # 64    96      312
            x1_rgb = self.layer1(x1_rgb)                # 256   96      312     1/4 
            x2_rgb = self.layer2(x1_rgb)                # 512   48      156     1/8

            # For Remission 
            x_rem = self.conv1_rem(rem)
            x_rem = self.bn1(x_rem)
            x0_rem = self.relu(x_rem)        
            x1_rem = self.maxpool(x0_rem)
            x1_rem = self.layer1(x1_rem)    
            x2_rem = self.layer2(x1_rem)

            return x0_rgb, x0_rem, x1_rgb, x1_rem, x2_rgb, x2_rem # 1/2, 1/4, 1/8


