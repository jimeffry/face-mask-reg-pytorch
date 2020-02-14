# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import numpy as np
import tqdm
import argparse
import cv2
import time
import scipy.io as scio
import shutil
from easydict import EasyDict as edict
sys.path.append(os.path.join(os.path.dirname(__file__), '../configs'))
from config import cfg

def process_data(fpath,imgdir,disdir):
    '''
    process mafa data:
    '''
    if not os.path.exists(disdir):
        os.makedirs(disdir)
    fp = scio.loadmat(fpath)
    fdata = fp['label_train']
    record = open('mafa_train.txt','w')
    fg_cnt = 0
    for i in tqdm.tqdm(range(fdata.shape[1])):
        imgname = fdata[0,i][1][0]
        labels = fdata[0,i][2][0,:]
        imagepath = os.path.join(imgdir,imgname)
        dispath = os.path.join(disdir,str(labels[12]),str(labels[13]))
        if not os.path.exists(dispath):
            os.makedirs(dispath)
        dispathname = os.path.join(dispath,imgname)
        # shutil.copyfile(imagepath,dispathname)
        savepath = os.path.join(str(labels[12]),str(labels[13]),imgname)
        bboxs = list(map(str,labels[:4]))
        if labels[12] ==3:
            bboxs.append('0')
        elif labels[12] ==1 :
            if labels[13]==1:
                bboxs.append('0')
            else:
                bboxs.append('1')
                fg_cnt +=1
        elif labels[12]==2:
            if labels[13]==3:
                bboxs.append('1')
                fg_cnt+=1
            else:bboxs.append('0')
        bbox_str = ','.join(bboxs)
        record.write("{},{}\n".format(savepath,bbox_str))
    print('fg cnt:',fg_cnt)
    print('bg_cnt:',fdata.shape[1]-fg_cnt)

def save_imgs(imgdir,txtfile,disdir):
    fr = open(txtfile,'r')
    fcnts = fr.readlines()
    if not os.path.exists(disdir):
        os.makedirs(disdir)
    fgdir = os.path.join(disdir,'fg')
    bgdir = os.path.join(disdir,'bg')
    if not os.path.exists(fgdir):
        os.makedirs(fgdir)
        os.makedirs(bgdir)
    for i in tqdm.tqdm(range(len(fcnts))):
        tmp = fcnts[i]
        tmp_splits = tmp.strip().split(',')
        imgname = tmp_splits[0]
        bbox_label = tmp_splits[1:]
        bbox_label = list(map(int,bbox_label))
        imgpath = os.path.join(imgdir,imgname)
        img = cv2.imread(imgpath)
        imgh,imgw = img.shape[:2]
        x1,y1,w,h = bbox_label[:4]
        xc = x1+w/2
        yc = y1+h/2
        w = w+w/2
        h = h+h/2
        x1 = int(np.clip(xc - w/2,0,imgw-1))
        y1 = int(np.clip(yc - h*3/4,0,imgh-1))
        x2 = int(np.clip(xc + w/2,0,imgw-1))
        y2 = int(np.clip(yc + h/2,0,imgh-1))
        imgcrop = img[y1:y2+1,x1:x2+1,:]
        imgcrop = cv2.resize(imgcrop,(112,112))
        # txt = str(bbox_label[-1])
        # point = (int(x1),int(y1))
        # cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
        # cv2.putText(img,txt,point,cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
        # cv2.imshow('src',img)
        # cv2.waitKey(0)
        sname = imgname.split('/')[-1]
        if bbox_label[-1] ==1:
            savepath = os.path.join(fgdir,sname)
        else:
            savepath = os.path.join(bgdir,sname)
        cv2.imwrite(savepath,imgcrop)

def generate_2_dir(imgdir,outfile):
    fw = open(outfile,'w')
    flist = os.listdir(imgdir)
    for tmp in flist:
        dir2 = os.path.join(imgdir,tmp)
        fpaths = os.listdir(dir2)
        if tmp =='fg':
            label = '1'
        else:
            label = '0'
        label = '1'
        for imgname in fpaths:
            fw.write("{},{}\n".format(tmp+'/'+imgname,label))

def merge2file(txt1,txt2,outfile,testfile):
    f1 = open(txt1,'r')
    f2 = open(txt2,'r')
    fw = open(outfile,'w')
    fw2 = open(testfile,'w')
    fcnts1 = f1.readlines()
    fcnts2 = f2.readlines()
    cnt = 0
    for tmp in fcnts1:
        cnt +=1
        if cnt <=110000:
            fw.write(tmp.strip()+'\n')
        else:
            fw2.write(tmp.strip()+'\n')
        if cnt ==115000:
            break
    cnt = 0
    for tmp in fcnts2:
        cnt +=1
        if cnt <1000:
            fw2.write(tmp.strip()+'\n')
        else:
            fw.write(tmp.strip()+'\n')

def resize_imgs(imgdir,disdir):
    files = os.listdir(imgdir)
    if not os.path.exists(disdir):
        os.makedirs(disdir)
    for tmp in files:
        tmp = tmp.strip()
        imgpath = os.path.join(imgdir,tmp)
        img = cv2.imread(imgpath)
        if img is not None:
            img = cv2.resize(img,(112,112))
            sname = tmp.split('.')[0]
            savepath = os.path.join(disdir,sname+'.jpg')
            cv2.imwrite(savepath,img)

def merge2file2(txt1,txt2,outfile):
    f1 = open(txt1,'r')
    f2 = open(txt2,'r')
    fw = open(outfile,'w')
    fcnts1 = f1.readlines()
    fcnts2 = f2.readlines()
    cnt = 0
    for tmp in fcnts1:
        fw.write(tmp.strip()+'\n')
    for tmp in fcnts2:
        fw.write(tmp.strip()+'\n')

if __name__=='__main__':
    fpath = '/data/detect/MAFA/MAFA_Label_Train/LabelTrainAll.mat'
    imgdir = '/data/detect/MAFA/images'
    disdir = '/data/detect/MAFA/'
    # process_data(fpath,imgdir,disdir)
    # save_imgs(disdir,'mafa_train.txt',disdir)
    disdir = '/data/detect/MAFA/tmp'
    celeba = '/data/Face_Reg/CelebA/img_detected'
    # generate_2_dir(celeba,'celeba_crop.txt')
    # merge2file('celeba_crop.txt','mafa_train_fg.txt','mafa_celeba_train.txt','mafa_celeba_test.txt')
    imgdir = '/home/lxy/Develop/git_prj/BaiduImageSpider/mask2'
    disdir = '/data/detect/breathmask/fg3'
    # resize_imgs(imgdir,disdir)
    # generate_2_dir('/data/detect/breathmask','../breakmask.txt')
    merge2file2('../mafa_celeba_train.txt','../breakmask.txt','../mafa_new_train.txt')
