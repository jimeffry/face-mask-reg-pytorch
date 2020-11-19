#-*- coding:utf-8 -*-
###############################################
#created by :  lxy
#Time:  2020/1/7 10:09
#project: head detect
#rversion: 0.1
#tool:   python 3.6
#modified:
#description  histogram
####################################################
from tensorflow.python.platform import gfile
import tensorflow as tf
import sys
import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import cv2
import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
from resnet import resnet50,resnet18
from mobilenet import mobilenet_v2
from shufflenet import shufflenet_v2_x1_0
from resnet_cbam import resnet50_cbam

def parms():
    parser = argparse.ArgumentParser(description='CSRnet demo')
    parser.add_argument('--save_dir', type=str, default='tmp/',
                        help='Directory for detect result')
    parser.add_argument('--breath_modelpath', type=str,
                        default='weights/s3fd.pth', help='trained model')
    parser.add_argument('--threshold', default=0.65, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--use_cuda', default=False, type=bool,
                        help='gpu run')
    parser.add_argument('--img_dir', type=str, default='tmp/',
                        help='Directory for images')
    parser.add_argument('--file_in', type=str, default='tmp.txt',
                        help='image namesf')
    return parser.parse_args()


class UnsellModel(object):
    def __init__(self,args):
        if args.cuda_use and torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False
        if self.use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        self.loadmodel(args.unsell_modelpath)
        self.real_num = 0

    def loadmodel(self,modelpath):
        if self.use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        # self.net = shufflenet_v2_x1_0(pretrained=False,num_classes=6).to(device)
        self.net = resnet18(pretrained=False,num_classes=2).to(device)
        # self.net = mobilenet_v2(pretrained=False,num_classes=6).to(device)
        # self.net = resnet50_cbam(pretrained=False,num_classes=3).to(device)
        state_dict = torch.load(modelpath,map_location=device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        if self.use_cuda:
            cudnn.benckmark = True
        # torch.save(self.net.state_dict(),'rbcar_best.pth')

    def propress(self,imgs):
        # rgb_mean = np.array([123.,117.,104.])[np.newaxis, np.newaxis,:].astype('float32')
        rgb_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis,:].astype('float32')
        rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        img_out = []
        for img in imgs:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            w,h = img.shape[:2]
            if w != cfg.InputSize_w or h != cfg.InputSize_h:
                img = cv2.resize(img,(cfg.InputSize_w,cfg.InputSize_h))
            img = img.astype('float32')
            img /= 255.0
            img -= rgb_mean
            img /= rgb_std
            # img = np.transpose(img,(2,0,1))
            img_out.append(img)
        return np.array(img_out)

    def inference(self,imglist):
        t1 = time.time()
        imgs = self.propress(imglist)
        bt_img = torch.from_numpy(imgs)
        if self.use_cuda:
            bt_img = bt_img.cuda()
        output = self.net(bt_img)
        # output = F.softmax(output,dim=-1)
        pred_cls = torch.argmax(output,dim=1)
        # scores = [output[idx,pred_cls[idx]].data.cpu().numpy() for idx in range(output.size(0))]
        t2 = time.time()
        # print('consuming:',t2-t1)
        return output.data.cpu().numpy(),pred_cls.data.cpu().numpy()



def args_p():
    parser = argparse.ArgumentParser(description='s3df demo')
    parser.add_argument('--save_dir', type=str, default='tmp/',
                        help='Directory for detect result')
    parser.add_argument('--modelpath', type=str,
                        default='weights/s3fd.pth', help='trained model')
    parser.add_argument('--threshold', default=0.1, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--ctx', default=True, type=bool,
                        help='gpu run')
    parser.add_argument('--img_dir', type=str, default='tmp/',
                        help='Directory for images')
    parser.add_argument('--file_in', type=str, default='tmp.txt',
                        help='image namesf')
    return parser.parse_args()

class UnsellModelTf(object):
    def __init__(self,args):
        self.loadmodel(args.unsell_modelpath)
        
    def loadmodel(self,mpath):
        tf_config = tf.ConfigProto()
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        #tf_config.gpu_options = gpu_options
        tf_config.gpu_options.allow_growth=True  
        tf_config.log_device_placement=False
        self.sess = tf.Session(config=tf_config)
        # self.sess = tf.Session()
        modefile = gfile.FastGFile(mpath, 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(modefile.read())
        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, name='unsell') 
        # tf.train.write_graph(graph_def, './', 'breathtest.pbtxt', as_text=True)
        # print("************begin to print graph*******************")
        # op = self.sess.graph.get_operations()
        # for m in op:
        #     # if 'input' in m.name or 'output' in m.name or 'confidence' in m.name:
        #     print(m.name)#m.values())
        # print("********************end***************")
        self.input_image = self.sess.graph.get_tensor_by_name('unsell/img_input:0') #img_input
        self.conf_out = self.sess.graph.get_tensor_by_name('unsell/softmax_output:0') #softmax_output

    def propress(self,imgs):
        rgb_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis,:].astype('float32')
        rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        img_out = []
        for img in imgs:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            w,h = img.shape[:2]
            if w != cfg.InputSize_w or h != cfg.InputSize_h:
                img = cv2.resize(img,(cfg.InputSize_w,cfg.InputSize_h))
            img = img.astype('float32')
            img /= 255.0
            img -= rgb_mean
            img /= rgb_std
            # img = np.transpose(img,(2,0,1))
            img_out.append(img)
        return np.array(img_out)

        
    def inference(self,imglist):
        t1 = time.time()
        bt_img = self.propress(imglist)
        # output = []
        # for i in range(bt_img.shape[0]):
        #     tmp_output = self.sess.run([self.conf_out],feed_dict={self.input_image:np.expand_dims(bt_img[i],0)})
        #     output.append(tmp_output[0][0])
        output = self.sess.run([self.conf_out],feed_dict={self.input_image:bt_img})
        # t2 = time.time()
        # print("debug*********",np.shape(output))
        output = np.array(output[0])
        pred_cls = np.argmax(output,axis=1)
        t3 = time.time()
        print('consuming:',t3-t1)
        # showimg = self.label_show(bboxes,imgorg)
        return output,pred_cls
  


if __name__ == '__main__':
    # args = parms()
    convert_pbtxt_to_pb('../breathtest.pbtxt')


if __name__ == '__main__':
    args = parms()
    detector = HeadCount(args)
    imgpath = args.file_in
    detector.headcnts(imgpath)