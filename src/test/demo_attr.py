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

from faceattr_model_tf import FaceAttribute

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
def parms():
    parser = argparse.ArgumentParser(description='FaceAtribute demo')
    parser.add_argument('--faceattr_modelpath', type=str,
                        default='../models', help='trained model')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--cuda_use', default=False, type=str2bool,
                        help='gpu run')
    parser.add_argument('--img_dir', type=str, default=None,
                        help='Directory for images')
    parser.add_argument('--file_in', type=str, default=None,
                        help='image namesf')
    parser.add_argument('--minsize',type=int,default=50,\
                        help="scale img size")
    parser.add_argument('--save_dir',type=str,default="./",help="images saved dir")
    parser.add_argument('--base-name',type=str,dest='base_name',default="videox",\
                        help="images saved dir")
    parser.add_argument('--save-dir2',type=str,dest='save_dir2',default=None,\
                        help="images saved dir")
    parser.add_argument('--crop-size',type=str,dest='crop_size',default='112,112',\
                        help="images saved size")
    parser.add_argument('--detect-model-dir',type=str,dest='detect_model_dir',default="../../models/",\
                        help="models saved dir")
    return parser.parse_args()


class FaceAttrTest(object):
    def __init__(self,args):
        self.Detect_Model = MtcnnDetector(args)
        self.FaceAttributeModel = FaceAttribute(args)
        self.threshold = args.threshold
        self.img_dir = args.img_dir
        self.real_num = 0
        self.save_dir = args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
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
        for idx,rectangle in enumerate(rectangles):
            tmp_pred = pred_id[idx]
            tmp_scores = scores[idx]
            labels = np.zeros(6,dtype=np.int32)
            show_scores = np.zeros(6)
            labels = [tmp_pred[0],tmp_pred[8],tmp_pred[10],tmp_pred[11],tmp_pred[15],tmp_pred[17]]
            show_scores = [tmp_scores[0],tmp_scores[8],tmp_scores[10],tmp_scores[11],tmp_scores[15],tmp_scores[17]]
            p_name = ['no_bear','bangs','male','hat','glass','smile']
            n_name = ['bear','no_bangs','female','no_hat','no_glass','no_smile']
            #p_color = [[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]]
            #n_color = [[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]]
            show_name = np.where(labels,p_name,n_name)
            # show_scores = np.where(labels,show_scores,1-show_scores)
            #show_color = np.where(labels,p_color,n_color)
            for t_id,score in enumerate(show_name):
                if labels[t_id]:
                    colors = (255,0,0)
                    tmp_s = show_scores[t_id]
                else:
                    colors = (0,0,255)
                    tmp_s = 1.0-show_scores[t_id]
                score = score+'_'+'%.2f' % tmp_s
                if score=='no_bear':
                    colors = (0,0,255)
                self.draw_label(img,rectangle,score,t_id,colors)
            if labels[3]:
                color_box = (255,0,0)
            else:
                color_box = (0,0,255)
            self.draw_box(img,rectangle,color_box)
        return img

    def draw_label(self,image, point, label,mode=0,color=(255,255,255),font=cv2.FONT_HERSHEY_COMPLEX_SMALL,
               font_scale=2, thickness=2):
        '''
        mode: 0~7
        '''
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = int(point[0]),int(point[1])
        w, h = int(point[2]),int(point[3])
        lb_w = int(size[0])
        lb_h = int(size[1])
        unit = int(mode)
        unit2 = int(mode+1)
        if y-int(unit2*lb_h) <= 0:
            cv2.rectangle(image, (x, y+h+unit*lb_h), (x + lb_w, y+h+unit2*lb_h), color)
            cv2.putText(image, label, (x,y+h+unit2*lb_h), font, font_scale, color, thickness)
        else:
            cv2.rectangle(image, (x, y-unit2*lb_h), (x + lb_w, y-unit*lb_h), color)
            cv2.putText(image, label, (x,y-unit*lb_h), font, font_scale, color, thickness)

    def draw_box(self,img,box,color=(255,0,0)):
        cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),color,1)
        '''
        if len(box)>5:
            if cfgs.x_y:
                for i in range(5,15,2):
                    cv2.circle(img,(int(box[i+0]),int(box[i+1])),2,(0,255,0))
            else:
                box = box[5:]
                for i in range(5):
                    cv2.circle(img,(int(box[i]),int(box[i+5])),2,(0,255,0))
        '''
        
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
                    # img_verify = cv2.resize(img_verify,(112,112))
                    img_verify = Img_Pad(img_verify,[cfg.InputSize_h,cfg.InputSize_w])
                    cv2.imshow('crop',img_verify)
                    face_breathmasks.append(img_verify)
                scores,pred_ids = self.FaceAttributeModel.inference(face_breathmasks)
                img_out = self.label_show(img,rectangles,scores,pred_ids)
        else:
            scores,pred_ids = self.FaceAttributeModel.inference([img])
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
                savename = tmp.strip()
                savepath = os.path.join(self.save_dir,savename)
                cv2.imwrite(savepath,frame)
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
    detector = FaceAttrTest(args)
    imgpath = args.file_in
    detector(imgpath)
    # evalu_img(args)