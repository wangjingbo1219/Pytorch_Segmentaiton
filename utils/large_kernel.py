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
from large_kernel_helper import *


class Large_Kernel_ResNet(nn.Module):

    def __init__(self, block,gcn,br,layers,bn_momentum=0.9):
        self.inplanes = 64
        super(Large_Kernel_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64,momentum=bn_momentum,affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],stride=1,dilation=1,bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,dilation=1,bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,dilation=1,bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,dilation=1,bn_momentum=bn_momentum)

        self.gcn1 = self._make_gcn_layer(gcn,2048,21,stride=1,padding=6,kernel=13,bn_momentum=bn_momentum)
        self.gcn2 = self._make_gcn_layer(gcn,1024,21,stride=1,padding=6,kernel=13,bn_momentum=bn_momentum)
        self.gcn3 = self._make_gcn_layer(gcn,512,21,stride=1,padding=6,kernel=13,bn_momentum=bn_momentum)
        self.gcn4 = self._make_gcn_layer(gcn,256,21,stride=1,padding=6,kernel=13,bn_momentum=bn_momentum)

        self.br1 = self._make_br_layer(br, 21, 21, stride=1, padding=1, kernel=3, bn_momentum=bn_momentum)
        self.br2 = self._make_br_layer(br, 21, 21, stride=1, padding=1, kernel=3, bn_momentum=bn_momentum)
        self.br3 = self._make_br_layer(br, 21, 21, stride=1, padding=1, kernel=3, bn_momentum=bn_momentum)
        self.br4 = self._make_br_layer(br, 21, 21, stride=1, padding=1, kernel=3, bn_momentum=bn_momentum)
        self.br5 = self._make_br_layer(br, 21, 21, stride=1, padding=1, kernel=3, bn_momentum=bn_momentum)
        self.br6 = self._make_br_layer(br, 21, 21, stride=1, padding=1, kernel=3, bn_momentum=bn_momentum)
        self.br7 = self._make_br_layer(br, 21, 21, stride=1, padding=1, kernel=3, bn_momentum=bn_momentum)
        self.br8 = self._make_br_layer(br, 21, 21, stride=1, padding=1, kernel=3, bn_momentum=bn_momentum)
        self.br9 = self._make_br_layer(br, 21, 21, stride=1, padding=1, kernel=3, bn_momentum=bn_momentum)



        self.upsamlpe1 = nn.UpsamplingBilinear2d(scale_factor=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,dilation=1,bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
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



    def _make_gcn_layer(self,gcn,inplanes,planes,stride,padding,kernel,bn_momentum):

        return gcn(inplanes,planes,stride,padding,kernel,bn_momentum)




    def _make_br_layer(self,br,inplanes,planes,stride,padding,kernel,bn_momentum):

        return br(inplanes,planes,stride,padding,kernel,bn_momentum)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
       
        x4 = self.gcn1(x4)
       
        x4 = self.br1(x4)
        x4 = self.upsamlpe1(x4)

        x3 = self.gcn2(x3)
        x3 = self.br2(x3)

        x3 = x3 +x4
        x3 = self.br3(x3)
        x3 = self.upsamlpe1(x3)

        x2 = self.gcn3(x2)
        x2 = self.br4(x2)
        x2 = x2 + x3
        x2 = self.br5(x2)
        x2 = self.upsamlpe1(x2)


        x1 = self.gcn4(x1)
        x1 = self.br6(x1)
        x1 = x1+x2
        x1 = self.br7(x1)
        x1 = self.upsamlpe1(x1)

        x1 = self.br8(x1)
        x1 = self.upsamlpe1(x1)
        x1 = self.br9(x1)




        return x1
