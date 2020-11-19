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

cfg.InputSize_w = 224 #224
cfg.InputSize_h = 224 #224
# shanghai dataset dir
cfg.Imgdir = '/data/detect/zhatu_car' #'/mnt/data/LXY.data/img_celeba/img_detected' #
# training set
cfg.EPOCHES = 600
cfg.LR_STEPS = [100000,200000,300000,400000]
cfg.MAX_STEPS = 500000
cfg.train_file = '../data/rubbish_train.txt'
cfg.test_file = '../data/rubbish_val.txt'
# -------------------------------------------- test model
cfg.threshold = [0.5,0.6,0.8]
cfg.ShowImg = 0
cfg.debug = 0
cfg.display_model = 0
cfg.batch_use = 0
cfg.time = 0
cfg.x_y = 1
cfg.box_widen = 1
#-------------------------------------face attribute
cfg.CLS_NUM = 2 #21 #inlcude background:0, mobile:1  tv:2 remote-control:3
cfg.FaceProperty = ['normal','unsell'] # ['No_Beard','Mustache','Goatee','5_o_Clock_Shadow','Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair','Bangs','Bald', \
        # 'Male','Wearing_Hat','Wearing_Earrings','Wearing_Necklace','Wearing_Necktie',\
        # 'Eyeglasses','Young','Smiling','Arched_Eyebrows','Bushy_Eyebrows','Blurry']
        #['normal','unsell']