import sys
import os
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random 
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg

class ReadDataset(data.Dataset):
    """VOC Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
    def __init__(self, root,annotxt):
        self.imgdir = root
        self.annotations = []
        self.loadtxt(annotxt)
        self.total_num = self.__len__()
        self.shulf_num = list(range(self.total_num))
        random.shuffle(self.shulf_num)
        # self.rgb_mean = np.array([123.,117.,104.])[np.newaxis, np.newaxis,:].astype('float32')
        self.rgb_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis,:].astype('float32')
        self.rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        # self.rgb_mean = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis,:].astype('float32')
        # self.rgb_std = np.array([0.225, 0.225, 0.225])[np.newaxis, np.newaxis,:].astype('float32')

    def loadtxt(self,annotxt):
        self.data_r = open(annotxt,'r')
        voc_annotations = self.data_r.readlines()
        for tmp in voc_annotations:
            tmp_splits = tmp.strip().split(',')
            img_path = os.path.join(self.imgdir,tmp_splits[0])
            # img_name = tmp_splits[0].split('/')[-1][:-4] if len(tmp_splits[0].split('/')) >0 else tmp_splits[0][:-4]
            label = int(tmp_splits[1])
            labels = [img_path,label]
            self.annotations.append(labels)

    def __getitem__(self, index):
        im, gt = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.annotations)

    def pull_item(self, index):
        idx = self.shulf_num[index]
        tmp_annotation = self.annotations[idx]
        tmp_path = tmp_annotation[0]
        img_data = cv2.imread(tmp_path)
        gt_label = tmp_annotation[1]
        img = cv2.cvtColor(img_data,cv2.COLOR_BGR2RGB)
        img = self.prepro(img)
        # return torch.from_numpy(img).permute(0,3,1,2), torch.from_numpy(target)
        return torch.from_numpy(img).permute(2,0,1), gt_label

    def prepro(self,img,):
        img = self.mirror(img)
        img = self.resize_subtract_mean(img)
        # img,gt = transform_crop(img,gt)
        return img

    def mirror(self,image):
        if random.randrange(2):
            image = image[:, ::-1,:]
        return image

    def resize_subtract_mean(self,image):
        h,w,_ = image.shape
        if h != cfg.InputSize_h or w != cfg.InputSize_w:
            image = cv2.resize(image,(int(cfg.InputSize_w),int(cfg.InputSize_h)))
        image = image.astype(np.float32)
        image = image / 255.0
        image -= self.rgb_mean
        image = image / self.rgb_std
        return image
    
if __name__=='__main__':
    train_dataset = ReadDataset(cfg.shanghai_dir,image_sets=[('part_B_final', 'train_data')])