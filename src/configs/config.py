# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/12/20 17:09
#project: face anti spoofing
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face anti spoofing
####################################################
from easydict import EasyDict 

cfg = EasyDict()

cfg.InputSize_w = 112
cfg.InputSize_h = 112
# shanghai dataset dir
cfg.Imgdir = '/mnt/data/LXY.data/img_celeba/img_detected' #'/mnt/data/LXY.data/'
# training set
cfg.EPOCHES = 600
cfg.LR_STEPS = [100000,200000,300000,400000]
cfg.MAX_STEPS = 500000
cfg.train_file = './mafa_celeba_train.txt'
cfg.test_file = './mafa_celeba_test.txt'
# -------------------------------------------- test model
cfg.threshold = [0.5,0.7,0.9]
cfg.ShowImg = 0
cfg.debug = 0
cfg.display_model = 0
cfg.batch_use = 0
cfg.time = 0
cfg.x_y = 1
cfg.box_widen = 1