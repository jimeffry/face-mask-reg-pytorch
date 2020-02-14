#-*- coding:utf-8 -*-
import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg


class MultiLoss(nn.Module):
    """
    calculate the L2 and Num_loss
    """
    def __init__(self,alpha=0.25,gamma=1.0):
        super(MultiLoss, self).__init__()
        # self.loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        """Multi Loss
        Args:
            predictions (tuple): network output density map,
            targets (tensor): Ground truth  labels for a batch,
                shape: [batch_size,imgh,imgw] 
        """
        pred_fg = predictions[:,1]
        pred_bg = predictions[:,0]
        alpha_factor_fg = torch.ones(targets.shape).cuda() * self.alpha
        alpha_factor_bg =  1. - alpha_factor_fg
        # focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
        # focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
        # bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
        # cls_loss = focal_weight * bce
        focal_weight_fg = alpha_factor_fg * torch.pow(1-pred_fg,self.gamma)
        focal_weight_bg = alpha_factor_bg * torch.pow(1-pred_bg,self.gamma)
        loss_c = -(focal_weight_fg * targets * torch.log(pred_fg) + focal_weight_bg * (1-targets)* torch.log(pred_bg))
        # loss_c = self.loss(predictions, targets)
        # print(loss_l.size())
        return loss_c.mean()