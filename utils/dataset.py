#from __future__ import  division
import numpy as np
import os
import torch
import scipy.ndimage as ndi


import torchvision
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.data_transfrom import *

from scipy import linalg

from six.moves import range





class VOC2012(Dataset):
    def __init__(self,data_file,data_dir,label_dir,
                 target_size=512,crop_size=320,zoom_range=[0.5,2],rotation=None,channel_shift = 0.,
                 mirror=True,label_cval=255,ignore_label = 255,nb_class =21,task = 'Train'):

        self.target_size = target_size
        self.crop_size = crop_size
        self.zoom_range = zoom_range
        self.rotation = rotation

        self.channel_shift = channel_shift
        self.mirror = mirror
        self.label_cval = label_cval
        self.ignore_label = ignore_label
        self.nb_class = nb_class
        self.task = task
        self.data_file = data_file
        self.data_dir = data_dir
        self.label_dir = label_dir
        file = open(self.data_file)
        self.data_list = file.readlines()
        file.close


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        data_index = self.data_list[index].strip('\n')
        img = Image.open(os.path.join(self.data_dir,data_index)+'.jpg')
        label = Image.open(os.path.join(self.label_dir,data_index)+'.png')
        img,label = self.transform(img,label)
        return img,label


    def transform(self,img,label):
        img_w, img_h = img.size
        img = img_to_array(img)
        label = img_to_array(label).astype(int)


        pad_w = max(self.target_size-img_w,0)
        pad_h = max(self.target_size-img_h,0)

        img = np.pad(img,((0,0),(pad_h/2,pad_h-pad_h/2),(pad_w/2,pad_w-pad_w/2)),'constant',constant_values=0.)
        label = np.pad(label,((0,0),(pad_h/2,pad_h-pad_h/2),(pad_w/2,pad_w-pad_w/2)),'constant',constant_values=self.label_cval).astype(int)


        if self.rotation:
            theta = np.pi/180*np.random.uniform(-self.rotation,self.rotation)
        else:
            theta = 0

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])


        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zy = zx
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(rotation_matrix,zoom_matrix)
        h = img.shape[1]
        w = img.shape[2]

        transform_matrix = transform_matrix_offset_center(transform_matrix,h,w)
        img = apply_transform(img,transform_matrix,0,fill_mode='constant',cval=0)
        label = apply_transform(label,transform_matrix,0,fill_mode='constant',cval=self.label_cval)
        if self.mirror:
            if np.random.random() < 0.5:
                img = flip_axis(img,2)
                label = flip_axis(label,2)

        if self.channel_shift != 0:
            img = random_channel_shift(img,self.channel_shift,0)


        #h = img.shape[1]
        #w = img.shape[2]
        #pad_w = max(self.crop_size-w,0)+np.random.randint(low=1,high=5)
        #pad_h = max(self.crop_size-h,0)+np.random.randint(low=1,high=5)
        #img = np.pad(img,((0,0),(pad_h/2,pad_h-pad_h/2),(pad_w/2,pad_w-pad_w/2)),'constant',constant_values=0.)
        #label = np.pad(label,((0,0),(pad_h/2,pad_h-pad_h/2),(pad_w/2,pad_w-pad_w/2)),'constant',constant_values=self.label_cval).astype(int)
        img,label = pair_random_crop(img,label,self.crop_size)
        img = standardlize(img)


        if self.ignore_label:
            label[np.where(label==self.ignore_label)] = -1

        img = np.ascontiguousarray(img)
        label = np.ascontiguousarray(label)
        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).long()
        img = img.float().div(256)
        img[0,:,:] = img[0,:,:].div(0.229)
        img[1,:,:] = img[1,:,:].div(0.224)
        img[2,:,:] = img[2,:,:].div(0.225)
        if self.task == 'Train':
            return img,label
        elif self.task == 'Test':
            return img
        else:
            raise ('task must in Train or Test')


class VOC2012Val(Dataset):
    def __init__(self,data_file,data_dir,label_dir,label_cval = 255,ignore_label = 255,nb_class = 21):
        self.data_file = data_file
        self.data_dir = data_dir
        self.label_dir = label_dir

        self.label_cval = label_cval
        self.ignore_label = ignore_label
        self.nb_class = nb_class

        file = open(self.data_file)
        self.data_list = file.readlines()
        file.close

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_index = self.data_list[index].strip('\n')
        img = Image.open(os.path.join(self.data_dir,data_index)+'.jpg')
        label = Image.open(os.path.join(self.label_dir,data_index)+'.png')

        img_w, img_h = img.size
        img = img_to_array(img)
        label = img_to_array(label).astype(int)

        pad_h = (32 - img_h % 32) % 32
        pad_w = (32 - img_w % 32) % 32

        img = np.lib.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), 'constant', constant_values=0.)
        label = np.lib.pad(label,((0,0),(0,pad_h),(0,pad_w)),'constant',constant_values=self.label_cval)

        img = standardlize(img)
        if self.ignore_label:
            label[np.where(label == self.ignore_label)] = -1
        img = np.ascontiguousarray(img)
        label = np.ascontiguousarray(label)
        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).long()
        img = img.float().div(256)
        img[0,:,:] = img[0,:,:].div(0.229)
        img[1,:,:] = img[1,:,:].div(0.224)
        img[2,:,:] = img[2,:,:].div(0.225)
        return img,label


