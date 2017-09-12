import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

from PIL import  Image

from models import *
from utils.data_transfrom import *
from inference import inference


def calculate_iou(nb_classes, res_dir, label_dir, image_list):
    conf_m = np.zeros((nb_classes,nb_classes), dtype=float)
    total = 0

    for img_num in image_list:
        img_num = img_num.strip('\n')
        total += 1
        print '#%d: %s'%(total,img_num)
        pred = img_to_array(Image.open('%s/%s.png'%(res_dir, img_num))).astype(int)
        label = img_to_array(Image.open('%s/%s.png'%(label_dir, img_num))).astype(int)
        flat_pred = np.ravel(pred)
        flat_label = np.ravel(label)
        #acc = 0.
        for p,l in zip(flat_pred,flat_label):
            if l==255:
                continue
            conf_m[l,p] += 1
    I = np.diag(conf_m)
    U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    IOU = I/U
    meanIOU = np.mean(IOU)
    return conf_m, IOU, meanIOU
def evaluate(model, weight_file, image_size, nb_classes, val_file_path, data_dir, label_dir):

    #init model
    current_dir = '/home/alex/PyTorch'
    save_path = os.path.join(current_dir,'initmodel')
    weight_file = os.path.join(save_path,weight_file)
    print weight_file
    pretrian_model = torch.load(weight_file)
    model.load_state_dict(pretrian_model)
    model.cuda(0)
    model.eval()

    #save temp data
    save_dir = os.path.join(current_dir,'save_data')
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)


    #create data list
    fp = open(val_file_path)
    image_list = fp.readlines()
    fp.close()

    start_time = time.time()

    inference(model,image_size,image_list,data_dir,return_results=False,save_dir=save_dir)
    duration = time.time()-start_time
    print '{}s used to make predictions.\n'.format(duration)

    start_time = time.time()
    conf_m, IOU, meanIOU = calculate_iou(nb_classes, save_dir, label_dir, image_list)
    print 'IOU: '
    print IOU
    print 'meanIOU: %f'%meanIOU
    print 'pixel acc: %f'%(np.sum(np.diag(conf_m))/np.sum(conf_m))
    duration = time.time() - start_time
    print '{}s used to calculate IOU.\n'.format(duration)

if __name__ == '__main__':
    model = resnet50()
    weight_file = 'resnet50_weight.pth'
    image_size = (512,512)
    nb_classes = 21
    val_file_path = '/home/alex/Data/VOClarge/VOC2012/ImageSets/Segmentation/val.txt'
    data_dir = '/home/alex/Data/VOClarge/VOC2012/JPEGImages'
    label_dir = '/home/alex/Data/VOClarge/VOC2012/SegmentationClass'
    evaluate(model,weight_file,image_size,nb_classes,val_file_path,data_dir,label_dir)