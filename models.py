import numpy as np
import torch.nn as nn
import os
import sys
import math
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from utils.resnet_helper import *
from utils.large_kernel_helper import *
from utils.resnet import *
from utils.large_kernel import *
import torch.nn.parameter as Parameter
def resnet50(blocks = [3,4,6,3],nb_classes = 21,bn_momentum=0.95):
    model = ResNet(Bottleneck, blocks,nb_classes,bn_momentum=bn_momentum)

    model_dict = model.state_dict()
    pretrained_dict = torch.load('/home/cis/PyTorch/initmodel/resnet50.pth')
    #model.load_state_dict(pretrained_dict)
    #print pretrained_dict
    #print model.state_dict()

   
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)


    model.load_state_dict(model_dict)

 

    #print model
    return model

def large_kernel_resnet50(blocks = [3,4,6,3],bn_momentum=0.95):
    model = Large_Kernel_ResNet(Bottleneck,GCN,BR,blocks,bn_momentum=bn_momentum)
    #model_dict = model.state_dict()
    pretrained_dict = torch.load('/home/cis/PyTorch/initmodel/resnet50.pth')
    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict)
    print model
    return model

if __name__ == '__main__':
    resnet50()