import numpy as np
import torch.nn as nn
import os
import sys
import math
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,padding=1,dilation=1,bn_momentum=0.9, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,stride=stride,padding=0,dilation=1,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features= planes,momentum=bn_momentum,affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=padding, dilation=dilation,bias=False)
        self.bn2 = nn.BatchNorm2d(planes,momentum=bn_momentum,affine=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1,padding=0,dilation=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4,momentum=bn_momentum,affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
