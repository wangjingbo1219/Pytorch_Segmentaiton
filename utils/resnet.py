import numpy as np
import torch.nn as nn
import os
import sys
import math
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from resnet_helper import *
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=21,bn_momentum=0.9):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64,momentum=bn_momentum,affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],stride=1,dilation=1,bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,dilation=1,bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,dilation=1,bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,dilation=1,bn_momentum=bn_momentum)

        self.fconv = nn.Conv2d(512 * block.expansion, num_classes,kernel_size=1,stride=1,bias = False)
        self.upsamlpe = nn.UpsamplingBilinear2d(scale_factor=32)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,dilation=1,bn_momentum=0.9):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,momentum=bn_momentum,affine=True),
            )

        layers = []
        if stride !=1:
            layers.append(block(self.inplanes, planes, stride=stride,dilation=dilation,padding=1,bn_momentum=bn_momentum,downsample=downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes,dilation = dilation,bn_momentum = bn_momentum))
        elif stride == 1:
            if dilation == 2:
                padding = 2
            else:
                padding =1
            layers.append(block(self.inplanes, planes, stride=stride,dilation=dilation,padding=padding,bn_momentum=bn_momentum,downsample=downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes,dilation = dilation,padding = padding,bn_momentum = bn_momentum))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.fconv(x)
        x = self.upsamlpe(x)
        return x