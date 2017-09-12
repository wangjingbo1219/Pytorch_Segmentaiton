from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def Pixel_Accuracy(output,target):
    n,c,h,w = output.size()
    output = output.permute(0,2,3,1).contiguous()
    target = target.permute(0,2,3,1).contiguous()

    output = output[target.repeat(1,1,1,c) >= 0].view(-1,c)
    output,predict = torch.max(output,dim = 1)

    predict = predict.view(-1)
    target = target[target >= 0].view(-1)

    r = (predict == target)
    num = r.float().sum().data[0]
    all = len(target)


    return num/all*100



#def Mean_IoU(output,target):
    



