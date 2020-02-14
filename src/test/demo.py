#-*- coding:utf-8 -*-
import sys
import os
import argparse

import cv2
import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

#from breathmask_model import BreathMask
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg
sys.path.append(os.path.join(os.path.dirname(__file__),'../face_detect'))
from Detector import MtcnnDetector
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from util import Img_Pad

from breathmask_model import BreathMask
# from breathmask_model_tf import BreathMask


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
def parms():
    parser = argparse.ArgumentParser(description='CSRnet demo')
    parser.add_argument('--save_dir', type=str, default='tmp/',
                        help='Directory for detect result')
    parser.add_argument('--breath_modelpath', type=str,
                        default='weights/s3fd.pth', help='trained model')
    parser.add_argument('--threshold', default=0.65, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--cuda_use', default=False, type=str2bool,
                        help='gpu run')
    parser.add_argument('--img_dir', type=str, default='tmp/',
                        help='Directory for images')
    parser.add_argument('--file_in', type=str, default=None,
                        help='image namesf')
    parser.add_argument('--minsize',type=int,default=50,\
                        help="scale img size")
    parser.add_argument('--img-dir',type=str,dest='img_dir',default="./",\
                        help="images saved dir")
    parser.add_argument('--save-dir',type=str,dest='save_dir',default="./",\
                        help="images saved dir")
    parser.add_argument('--base-name',type=str,dest='base_name',default="videox",\
                        help="images saved dir")
    parser.add_argument('--save-dir2',type=str,dest='save_dir2',default=None,\
                        help="images saved dir")
    parser.add_argument('--crop-size',type=str,dest='crop_size',default='112,112',\
                        help="images saved size")
    parser.add_argument('--detect-model-dir',type=str,dest='detect_model_dir',default="../../models/",\
                        help="models saved dir")
    return parser.parse_args()


class BreathMastTest(object):
    def __init__(self,args):
        self.Detect_Model = MtcnnDetector(args)
        self.BreathMaskModel = BreathMask(args)
        self.threshold = args.threshold
        self.img_dir = args.img_dir
        self.real_num = 0
    
    def img_crop(self,img,bbox,imgw,imgh):
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        boxw = x2-x1
        boxh = y2-y1
        x1 = int(max(0,int(x1-0.3*boxw)))
        y1 = int(max(0,int(y1-0.3*boxh)))
        x2 = int(min(imgw,int(x2+0.3*boxw)))
        y2 = int(min(imgh,int(y2+0.3*boxh)))
        cropimg = img[y1:y2,x1:x2,:]
        return cropimg
    
    def label_show(self,img,rectangles,scores,pred_id):
        '''
        scores: shape-[batch,cls_nums]
        pred_id: shape-[batch,cls_nums]
        rectangles: shape-[batch,15]
            0-5: x1,y1,x2,y2,score
        '''
        show_labels = ['no_weare','weare_mask']
        colors = [(0,0,255),(255,0,0)]
        for idx,box in enumerate(rectangles):
            tmp_pred = pred_id[idx]
            tmp_score = '%.2f' % scores[idx]
            show_name = show_labels[int(tmp_pred)] +'_'+tmp_score
            color = colors[int(tmp_pred)]
            cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),color,1)
            font=cv2.FONT_HERSHEY_COMPLEX_SMALL
            font_scale = int((box[3]-box[1])*0.01)
            points = (int(box[0]),int(box[1]))
            cv2.putText(img, show_name, points, font, font_scale, color, 2)
        return img
        
    def inference_img(self,img):
        t1 = time.time()
        imgorg = img.copy()
        orgreg = 0
        img_out = img
        pred_ids = []
        if not orgreg:
            rectangles = self.Detect_Model.detectFace(imgorg)
            face_breathmasks = []
            frame_h,frame_w = imgorg.shape[:2]
            img_out = img.copy()
            pred_ids = []
            if len(rectangles)> 0:
                for box in rectangles:
                    img_verify = self.img_crop(imgorg,box,frame_w,frame_h)
                    img_verify = cv2.resize(img_verify,(112,112))
                    # img_verify = Img_Pad(img_verify,[cfg.InputSize_h,cfg.InputSize_w])
                    cv2.imshow('crop',img_verify)
                    face_breathmasks.append(img_verify)
                scores,pred_ids = self.BreathMaskModel.inference(face_breathmasks)
                img_out = self.label_show(img,rectangles,scores,pred_ids)
        else:
            scores,pred_ids = self.BreathMaskModel.inference([img])
            img_out = self.label_show(img,[[10,10,119,119]],scores,pred_ids)
        t2 = time.time()
        #print('consuming:',t2-t1)
        return img_out,pred_ids

    def __call__(self,imgpath):
        if os.path.isdir(imgpath):
            cnts = os.listdir(imgpath)
            for tmp in cnts:
                tmppath = os.path.join(imgpath,tmp.strip())
                img = cv2.imread(tmppath)
                if img is None:
                    continue
                frame,cnt_head = self.inference_img(img)
                # print('heads >> ',cnt_head)
                cv2.imshow('result',frame)
                #savepath = os.path.join(self.save_dir,save_name)
                # cv2.imwrite('test.jpg',frame)
                cv2.waitKey(0) 
        elif os.path.isfile(imgpath) and imgpath.endswith('txt'):
            # if not os.path.exists(self.save_dir):
            #     os.makedirs(self.save_dir)
            f_r = open(imgpath,'r')
            file_cnts = f_r.readlines()
            for j in tqdm(range(len(file_cnts))):
                tmp_file = file_cnts[j].strip()
                tmp_file_s = tmp_file.split('\t')
                if len(tmp_file_s)>0:
                    tmp_file = tmp_file_s[0]
                    self.real_num = int(tmp_file_s[1])
                if not tmp_file.endswith('jpg'):
                    tmp_file = tmp_file +'.jpg'
                # tmp_path = os.path.join(self.img_dir,tmp_file) 
                tmp_path = tmp_file
                if not os.path.exists(tmp_path):
                    print(tmp_path)
                    continue
                img = cv2.imread(tmp_path) 
                if img is None:
                    print('None',tmp)
                    continue
                frame,cnt_head = self.inference_img(img)
                cv2.imshow('result',frame)
                #savepath = os.path.join(self.save_dir,save_name)
                #cv2.imwrite('test.jpg',frame)
                cv2.waitKey(0) 
        elif os.path.isfile(imgpath) and imgpath.endswith(('.mp4','.avi')) :
            cap = cv2.VideoCapture(imgpath)
            if not cap.isOpened():
                print("failed open camera")
                return 0
            else: 
                while cap.isOpened():
                    _,img = cap.read()
                    frame,cnt_head = self.inference_img(img)
                    cv2.imshow('result',frame)
                    q=cv2.waitKey(10) & 0xFF
                    if q == 27 or q ==ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()
        elif os.path.isfile(imgpath):
            img = cv2.imread(imgpath)
            if img is not None:
                # grab next frame
                # update FPS counter
                frame,cnt_head = self.inference_img(img)
                print(cnt_head)
                cv2.imshow('result',frame)
                # cv2.imwrite('test_a1.jpg',frame)
                key = cv2.waitKey(0) 
        elif imgpath=='video':
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("failed open camera")
                return 0
            else: 
                while cap.isOpened():
                    _,img = cap.read()
                    frame,cnt_head = self.inference_img(img)
                    cv2.imshow('result',frame)
                    q=cv2.waitKey(10) & 0xFF
                    if q == 27 or q ==ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()
        else:
            print('please input the right img-path')

if __name__ == '__main__':
    args = parms()
    detector = BreathMastTest(args)
    imgpath = args.file_in
    detector(imgpath)
    # evalu_img(args)