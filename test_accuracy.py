from __future__ import division
import torch
import numpy as np
import time
from PIL import Image
from torch.optim import SGD

from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.accuracy import *
from utils.resnet import *

from utils.output_precess import *
from utils.dataset import *
from utils.loss import *
from models import *
from utils.accuracy import *

val_data_file = '/home/alex/Data/VOClarge/VOC2012/ImageSets/Segmentation/val.txt'
data_dir = '/home/alex/Data/VOClarge/VOC2012/JPEGImages'
label_dir = '/home/alex/Data/VOClarge/VOC2012/SegmentationClass'

val_data_loader = DataLoader(VOC2012Val(val_data_file, data_dir, label_dir, label_cval=255,
                                        ignore_label=255, nb_class=21),
                             num_workers=8, batch_size=1, shuffle=False)
model = resnet50()
pretrained_dict = torch.load('/home/alex/PyTorch/initmodel/resnet50_weight.pth')
model.load_state_dict(pretrained_dict)
model.eval()
model.cuda(0)

acc = []
print "Begin Test"
for step, (images, labels) in enumerate(val_data_loader):
    images = images.cuda(0)
    labels = labels.cuda(0)
    inputs = Variable(images)
    labels = Variable(labels)
    outputs = model(inputs)
    pixel_acc = Pixel_Accuracy(outputs,labels)
    print "End----PixelAccuracy:%f" % (pixel_acc)
    acc.append(pixel_acc)
print np.mean(acc)