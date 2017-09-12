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
from utils.newdataset import *
from utils.loss import *
from models import *


def main(model,train_data_file,val_data_file,data_dir,label_dir,base_lr = 0.01,power = 0.9,num_epochs= 144):
    #init model
    model.train()
    model.cuda(0)

    #load data
    train_data_loader = DataLoader(VOC2012(train_data_file,data_dir,label_dir,
            crop_size=448,zoom_range=[0.5,1.5],rotation=0.,channel_shift = 20.,
            mirror=True,label_cval=255,ignore_label = 255,nb_class =21,task = 'Train'),
            num_workers=12,batch_size=16,shuffle=True)

    val_data_loader = DataLoader(VOC2012Val(val_data_file,data_dir,label_dir,label_cval=255,
                                         ignore_label = 255,nb_class=21),
                                num_workers=8,batch_size=1,shuffle=False )

    #split param
    #fcov_params = list(map(id,model.fconv.parameters()))
    #base_params = filter(lambda p:id(p) not in fcov_params,model.parameters())


    #compile model
    optimizer = SGD(model.parameters(),lr=base_lr,momentum=0.99,weight_decay=5e-4,nesterov=True)
    #optimizer = SGD([{'params':base_params},
                     #{'params':model.fconv.parameters(),'lr':base_lr*10}],lr=base_lr,momentum=0.9,
                    #weight_decay=1e-4,nesterov=True)
                    

    #poly
    def lr_poly(base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))
    print '----Begin Training----'
    print '---Total: %d Epochs---'%(num_epochs)
    print '---Base lr:%f---Momentum:%f---Weight Decay:%f---'%(base_lr,0.99,1e-4)
    for epoch in range(num_epochs):

        # train
        model.train()
        for step,(images,labels) in enumerate(train_data_loader):
            t1=time.clock()
            optimizer.zero_grad()
            images = images.cuda(0)
            labels = labels.cuda(0)
            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs)
            loss,acc= Sparse_Cross_Entropy(outputs,targets)
            loss.backward()
            optimizer.step()
            lr_ = lr_poly(base_lr=base_lr, iter=epoch*661+step+1, max_iter=num_epochs*661, power=power)
            optimizer = SGD(model.parameters(),lr=lr_,momentum=0.99,weight_decay=5e-4,nesterov=True)
            #optimizer = SGD([{'params': base_params},
                             #{'params': model.fconv.parameters(), 'lr': lr_*10}], lr=lr_, momentum=0.9,
                            #weight_decay=1e-4, nesterov=True)
            t2=time.clock()
            print "-----------Epoch:%d-----Iter:%d-----------" %(epoch,step)
            print 'Time:%fs(%f iter/s)----lr:%f-----Loss:%f' %(t2-t1,1/(t2-t1),lr_,loss.data[0])
            print 'Accuracy:',acc,'%'
        '''
        # val
        if epoch%10 == 0 or epoch == num_epochs-1:
        '''
        start_time = time.time()

        print "--------Test mean IoU----------"
        print "----------Epoch:%d-----------"%(epoch)
        model.eval()
        conf_m = np.zeros((21, 21))
        n=0
        for step,(images,labels) in enumerate(val_data_loader):
            images = images.cuda(0)
            labels = labels.cuda(0)
            inputs = Variable(images)
            labels = Variable(labels)
            outputs = model(inputs)
            outputs,labels = output_data_processing(outputs,labels)
            outputs = np.ravel(outputs).astype(int)
            labels = np.ravel(labels).astype(int)
            for p,l in zip(outputs,labels):
                conf_m[l,p] += 1
            I = np.diag(conf_m)
            U = np.sum(conf_m,axis=0)+np.sum(conf_m,axis=1) - I
            IoU = I/U
            mean_IoU = np.mean(IoU)
            pixel_acc = np.sum(np.diag(conf_m))/np.sum(conf_m)
            if (step%100 == 0):
                print "Step:%d-------PixelAccuracy:%f"%(step,pixel_acc)
        end_time = time.time()
        print "---------Time:%f--------MeanIoU:%f----PixelAccuracy:%f-------"%(end_time-start_time,mean_IoU,pixel_acc)
        '''
        else:
            print "-------Only Test pixel acc-------"
            start_time = time.time()
            model.eval()
            acc = []
            for step, (images, labels) in enumerate(val_data_loader):
                images = images.cuda(0)
                labels = labels.cuda(0)
                inputs = Variable(images)
                labels = Variable(labels)
                outputs = model(inputs)
                pixel_acc = Pixel_Accuracy(outputs, labels)
                acc.append(pixel_acc)
            mean_acc = np.mean(acc)
            end_time = time.time()
            print "-------Time:%f-----Epoch:%d-------Acc:%f------"%(end_time-start_time,epoch,mean_acc)
        '''
        torch.save(model.state_dict(),'/home/cis/PyTorch/initmodel/resnet50_weight_trainval.pth')
    print '----End Training----'
if __name__ =='__main__':
    model = large_kernel_resnet50(bn_momentum=0.99)
    train_data_file = '/home/cis/Data/VOClarge/VOC2012/ImageSets/Segmentation/train.txt'
    val_data_file = '/home/cis/Data/VOClarge/VOC2012/ImageSets/Segmentation/val.txt'
    data_dir = '/home/cis/Data/VOClarge/VOC2012/JPEGImages'
    label_dir = '/home/cis/Data/VOClarge/VOC2012/SegmentationClass'
    base_lr = 0.01
    power = 0.9
    num_epochs = 144
    main(model,train_data_file,val_data_file,data_dir,label_dir,base_lr=base_lr,num_epochs=num_epochs)
