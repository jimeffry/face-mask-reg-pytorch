#-*- coding:utf-8 -*-
import os
import sys
import numpy as np
from load_image import ReadDataset
import torch
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg


def dataset_factory(dataset,mode='train'):
    train_dataset = None
    val_dataset = None
    # if dataset == 'mafa' :
    train_dataset = ReadDataset(cfg.Imgdir,cfg.train_file)
    val_dataset = ReadDataset(cfg.Imgdir,cfg.test_file)
    # else:
        # print("please input right dataset")
    return train_dataset, val_dataset

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    targets = np.array(targets,dtype=np.float32)
    return torch.stack(imgs, 0), torch.from_numpy(targets.copy()).long()
    # return batch[0][0],batch[0][1]