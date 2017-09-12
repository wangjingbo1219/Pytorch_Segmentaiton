import torch
import torch.nn as nn
import os
import numpy as np
from torch.autograd import Variable
from utils.data_transfrom import *
from PIL import  Image


def inference(model,image_size,image_list,data_dir,return_results = True,save_dir = None):



    results = []
    total = 0
    for img_num in image_list:
        img_num = img_num.strip('\n')
        total += 1
        print '#%d: %s' % (total, img_num)
        image = Image.open('%s/%s.jpg' % (data_dir, img_num))
        image = img_to_array(image)
        img_h ,img_w = image.shape[1:3]
        pad_h = (32 - img_h % 32) % 32
        pad_w = (32 - img_w % 32) % 32


        '''
        pad_h = max(image_size[0]-img_h,0)
        pad_w = max(image_size[1]-img_w,0)
        image = np.lib.pad(image,((0,0),(pad_h/2,pad_h-pad_h/2),(pad_w/2,pad_w-pad_w/2)),'constant',constant_values=0.)
        '''

        image = np.lib.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), 'constant', constant_values=0.)
        image = standardlize(image)
        image = np.expand_dims(image,axis=0)
        image = np.ascontiguousarray(image)

        image = torch.from_numpy(image).float()
        image = image.cuda(0)
        image = Variable(image)
        output = model(image)
        output = output.permute(0,2,3,1)
        result = output.data[0].cpu()
        result = result.numpy()

        result = np.argmax(np.squeeze(result),axis=-1).astype(np.uint8)
        
        result_img = Image.fromarray(result, mode='P')
        #result_img = result_img.crop((pad_w / 2, pad_h / 2, pad_w / 2 + img_w, pad_h / 2 + img_h))
        result_img = result_img.crop((0,0,img_w,img_h))
        if return_results:
            results.append(result_img)
        if save_dir:
            result_img.save(os.path.join(save_dir, img_num + '.png'))
    return results


