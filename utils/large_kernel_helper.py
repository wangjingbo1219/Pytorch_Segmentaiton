import numpy as np
import torch.nn as nn
import os
import sys
import math
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self,inplanes,planes,stride,padding,kernel,bn_momentum=0.95):
        super(GCN,self).__init__()
        #left side
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=(kernel,1),padding=(padding,0),stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=planes,momentum=bn_momentum,affine=True)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=(1,kernel),padding=(0,padding),stride=stride)
        self.bn2 = nn.BatchNorm2d(num_features=planes,momentum=bn_momentum,affine=True)


        #right side
        self.conv3 = nn.Conv2d(inplanes,planes,kernel_size=(1,kernel),padding=(0,padding),stride=stride)
        self.bn3 = nn.BatchNorm2d(num_features=planes,momentum=bn_momentum,affine=True)
        self.conv4 = nn.Conv2d(planes,planes,kernel_size=(kernel,1),padding=(padding,0),stride=stride)
        self.bn4 = nn.BatchNorm2d(num_features=planes,momentum=bn_momentum,affine=True)

        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)

        x2 = self.conv3(x)
        x2 = self.bn3(x2)
        x2 = self.relu(x2)
        x2 = self.conv4(x2)
        x2 = self.bn4(x2)
        x2 = self.relu(x2)

        output = x1+x2
        return output




class BR(nn.Module):
    def __init__(self,inplanes,planes,stride=1,padding=1,kernel=3,bn_momentum=0.95):

        super(BR,self).__init__()

        #right side

        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=kernel,padding=padding,stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=planes,momentum=bn_momentum,affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes,planes,kernel_size=kernel,padding=padding,stride=stride)
        self.bn2 = nn.BatchNorm2d(num_features=planes,momentum=bn_momentum,affine=True)

    def forward(self,x):
        res = x
        out = self.conv1(res)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += x

        return out

