from __future__ import division
import torch

import torch.nn as nn
import torch.nn.functional as F


def Sparse_Cross_Entropy(input,target,weight = None,size_average = True):
    #prepare data
    n,c,h,w = input.size()
    input = input.permute(0,2,3,1).contiguous()
    target = target.permute(0,2,3,1).contiguous()
    input = input[target.repeat(1,1,1,c) >=0 ].view(-1,c)

    temp,prediction = torch.max(input,dim=1)

    prediction = prediction.view(-1).contiguous()

    target = target[target >= 0].view(-1).contiguous()
    #calculate loss
    loss = F.cross_entropy(input,target,weight=weight,size_average=size_average)

    #calculate acc
    
    prediction = prediction.view(-1)
    #target = target[target >= 0].view(-1)
    r = (prediction == target)
    num = r.float().sum().data[0]
    all = len(target)
    acc = num/all*100

    return loss,acc
'''

def Sparse_Cross_Entropy(output,target):
    mask = (target.view(-1) >=0 )
    target = target.view(-1)[mask]

    C = output.size(1)
    output = output.permute(0,2,3,1)

    output = output.contiguous().view(-1,C)
    mask2d = mask.unsqueeze(1).expand(mask.size(0),C).contiguous().view(-1)

    output = output[mask2d].view(-1,C)
    loss = F.cross_entropy(output,target,size_average=True)
    return loss

'''

