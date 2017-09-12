import torch
import torch.nn as nn

import torch.nn.functional as F

def output_data_processing(output,target):
    n,c,h,w = output.size()

    output = output.permute(0,2,3,1).contiguous()
    target = target.permute(0,2,3,1).contiguous()

    output = output[target.repeat(1,1,1,c) >=0 ].view(-1,c)

    output,prediction = torch.max(output,dim = 1 )
    prediction = prediction.view(-1)

    target = target[target >= 0].view(-1)
    prediction = prediction.data.cpu().numpy()
    target = target.data.cpu().numpy()
    return prediction,target