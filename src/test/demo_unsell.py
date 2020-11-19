#-*- coding:utf-8 -*-
import sys
import os
import argparse
import shutil
import json

import cv2
import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from load_model import UnsellModel,UnsellModelTf
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from util import Img_Pad

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
def parms():
    parser = argparse.ArgumentParser(description='CSRnet demo')
    parser.add_argument('--save_dir', type=str, default='tmp/',
                        help='Directory for detect result')
    parser.add_argument('--unsell_modelpath', type=str,
                        default='weights/s3fd.pth', help='trained model')
    parser.add_argument('--threshold', default=0.65, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--cuda_use', default=False, type=str2bool,
                        help='gpu run')
    parser.add_argument('--img_dir', type=str, default='tmp/',
                        help='Directory for images')
    parser.add_argument('--file_in', type=str, default=None,
                        help='image namesf')
    parser.add_argument('--minsize',type=int,default=24,\
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
    parser.add_argument('--detect_modelpath',type=str,default=None,help='retinateface')
    parser.add_argument('--detect-model-dir',type=str,dest='detect_model_dir',default="../../models/",\
                        help="models saved dir")
    return parser.parse_args()


class UnsellTest(object):
    def __init__(self,args):
        # self.Model = UnsellModel(args)
        self.Model = UnsellModelTf(args)
        self.threshold = args.threshold
        self.img_dir = args.img_dir
        self.real_num = 0
        self.save_dir = args.save_dir
        if self.save_dir is not None :
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
    
    def img_crop(self,img,bbox,imgw,imgh):
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        boxw = x2-x1
        boxh = y2-y1
        cropimg = []
        if boxw > 20:
            x1 = int(max(0,int(x1-0.3*boxw)))
            y1 = int(max(0,int(y1-0.3*boxh)))
            x2 = int(min(imgw,int(x2+0.3*boxw)))
            y2 = int(min(imgh,int(y2+0.3*boxh)))
            cropimg = img[y1:y2,x1:x2,:]
        return cropimg
    
    def label_show(self,img,scores,pred_id):
        '''
        scores: shape-[batch,cls_nums]
        pred_id: shape-[batch,cls_nums]
        rectangles: shape-[batch,15]
            0-5: x1,y1,x2,y2,score
        '''
        show_labels = ['normal','unsell']
        colors = [(0,0,255),(255,0,0)]
        tmp_pred = pred_id
        tmp_score = '%.2f' % scores[int(tmp_pred)]
        show_name = show_labels[int(tmp_pred)] #+'_'+tmp_score
        color = colors[int(tmp_pred)]
        font=cv2.FONT_HERSHEY_COMPLEX_SMALL
        # font_scale = int((box[3]-box[1])*0.01)
        points = (10,10)
        cv2.putText(img, show_name, points, font, 1.0, color, 2)
        return img
        
    def inference_img(self,img):
        t1 = time.time()
        imgorg = img.copy()
        orgreg = 0
        pred_ids = []
        scores,pred_ids = self.Model.inference([img])
        # img_out = self.label_show(img,scores[0],pred_ids[0])
        #print('consuming:',t2-t1)
        return scores[0],pred_ids[0]

    def putrecord(self,datalist):
        file = open('ehualu_test.json','w',encoding='utf-8')
        json.dump(datalist,file,ensure_ascii=False)
        file.close()

    def __call__(self,imgpath):
        if os.path.isdir(imgpath):
            cnts = os.listdir(imgpath)
            for tmp in cnts:
                tmppath = os.path.join(imgpath,tmp.strip())
                img = cv2.imread(tmppath)
                if img is None:
                    continue
                score,pid = self.inference_img(img)
                print("scores and cls_id:",score,pid,tmp.strip())
                cv2.imshow('result',img)
                # save_name = tmp.strip()
                # savepath = os.path.join(self.save_dir,save_name)
                # cv2.imwrite(savepath,frame)
                cv2.waitKey(0) 
        elif os.path.isfile(imgpath) and imgpath.endswith('txt'):
            f_r = open(imgpath,'r')
            file_cnts = f_r.readlines()
            #**********smoking and calling
            label_name_list = ['normal','smoking','calling']
            data_list = []
            #******** for label and to be classed
            bg_path = os.path.join(self.save_dir,'bg_imgs')
            fg_path = os.path.join(self.save_dir,'fg_imgs')
            pathlist = [bg_path,fg_path]
            #
            if not os.path.exists(bg_path):
                os.makedirs(bg_path)
            if not os.path.exists(fg_path):
                os.makedirs(fg_path)
            for j in tqdm(range(len(file_cnts))):
                record_dict = dict()
                tmp_file = file_cnts[j].strip()
                #*************** here is for label file test
                tmp_file_s = tmp_file.split(',')
                if len(tmp_file_s)>0:
                    tmp_file = tmp_file_s[0]
                    real_label = int(tmp_file_s[1])
                # if not tmp_file.endswith('jpg'):
                    # tmp_file = tmp_file +'.jpg'
                tmp_path = os.path.join(self.img_dir,tmp_file) 
                # tmp_path = tmp_file
                if not os.path.exists(tmp_path):
                    print(tmp_path)
                    continue
                img = cv2.imread(tmp_path) 
                if img is None:
                    print('None',tmp)
                    continue
                scores,pred_id = self.inference_img(img)
                #************ smoking and calling test
                # record_dict['image_name'] = tmp_file
                # record_dict['category'] = label_name_list[int(pred_id)]
                # record_dict['score'] = '%.5f' % scores[int(pred_id)]
                # data_list.append(record_dict)
                #************to be classed images
                # dist_path = os.path.join(pathlist[pred_id],tmp_file)
                # shutil.copyfile(tmp_path,dist_path)
                #********* label test files
                if int(pred_id) != real_label:
                    tmp_file_s = tmp_file.split('/')
                    dist_path = os.path.join(pathlist[real_label],tmp_file_s[-1])
                    shutil.copyfile(tmp_path,dist_path)
            # self.putrecord(data_list)
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
                score,pid = self.inference_img(img)
                print("scores and cls_id:",score,pid)
                cv2.imshow('img',img)
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
    detector = UnsellTest(args)
    imgpath = args.file_in
    detector(imgpath)
    # evalu_img(args)