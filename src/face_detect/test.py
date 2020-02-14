# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/06/12 14:09
#project: Face detect
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face detect testing caffe model
####################################################
import sys
from tqdm import tqdm
sys.path.append('.')
import os
os.environ['GLOG_minloglevel'] = '2'
import cv2
import numpy as np
import argparse
import time
import mxnet as mx
from scipy.spatial import distance
from Detector import MtcnnDetector 
import shutil

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def args():
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
    parser.add_argument('--file_in', type=str, default='tmp.txt',
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


def alignImg_opencv(img,image_size,points):
    '''
    image_size: [h,w]
    points: coordinates of the 5 points(eye,nose,mouse) will be ordered x1,x2,x3,x4,x5,y1,y2,y3,y4,y5
    '''
    dst = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==112:
        dst[:,0] += 8.0
    crop_imgs = []
    #new_dst_points = np.array([[(dst[0,0] + dst[1,0]) / 2, (dst[0,1] + dst[1,1]) / 2], [(dst[3,0] + dst[4,0]) / 2, (dst[3,1] + dst[4,1]) / 2]])
    for p in points:
        src = np.reshape(p,(2,5)).T
        src = src.astype(np.float32)
        #src_points = np.array([[(src[0][0] + src[1][0]) / 2, (src[0][1] + src[1][1]) / 2],[(src[3][0] + src[4][0]) / 2, (src[3][1] + src[4][1]) / 2]])
        similarTransformation = cv2.estimateRigidTransform(src.reshape(1,5,2), dst.reshape(1,5,2), fullAffine=True)
        #if similarTransformation is None:
            #continue
        warped = cv2.warpAffine(img, similarTransformation,(image_size[1],image_size[0]),borderMode=1)
        if warped is not None:
            crop_imgs.append(warped)
    return crop_imgs

def evalu_img(args):
    cv2.namedWindow("test")
    cv2.moveWindow("test",1400,10)
    threshold = np.array([0.5,0.7,0.9])
    base_name = "test_img"
    save_dir = './output'
    crop_size = [112,112]
    #detect_model = MTCNNDet(min_size,threshold)
    detect_model = MtcnnDetector(args)
    #alignface = Align_img(crop_size)
    imgorg = cv2.imread(args.file_in)
    #img = cv2.resize(img,(640,480))
    #img = cv2.cvtColor(imgorg,cv2.COLOR_BGR2RGB)
    img = imgorg
    h,w = img.shape[:2]
    rectangles = detect_model.detectFace(img)
    #draw = img.copy()
    print("num:",len(rectangles))
    if len(rectangles)>0:
        points = np.array(rectangles)
        #print('rec shape',points.shape)
        points = points[:,5:]
        #print("landmarks: ",points)
        points_list = points.tolist()
        # crop_imgs = alignImg(img,crop_size,points_list)
        crop_imgs = alignImg_opencv(img,crop_size,points_list)
        # crop_imgs = alignImg_solve(img,crop_size,points_list)
        #crop_imgs = alignface.extract_image_chips(img,points_list)
        # crop_imgs = alignImg_angle(img,crop_size,points_list)
        for idx_cnt,img_out in enumerate(crop_imgs):
            savepath = os.path.join(save_dir,base_name+'_'+str(idx_cnt)+".jpg")
            #img_out = cv2.resize(img_out,(112,112))
            cv2.imshow("crop",img_out)
            cv2.waitKey(0)
            cv2.imwrite(savepath,img_out)
        for rectangle in rectangles:
            print('w,h',rectangle[2]-rectangle[0],rectangle[3]-rectangle[1])
            score_label = str("{:.2f}".format(rectangle[4]))
            cv2.putText(imgorg,score_label,(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
            cv2.rectangle(imgorg,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(239,185,0),2)
            if len(rectangle) > 5:
                if 1:
                    for i in range(5,15,2):
                        cv2.circle(imgorg,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
                else:
                    rectangle = rectangle[5:]
                    for i in range(5):
                        cv2.circle(imgorg,(int(rectangle[i]),int(rectangle[i+5])),2,(0,255,0))
    else:
        print("No face detected")
    cv2.imshow("test",imgorg)
    cv2.waitKey(0)
    # cv2.imwrite('test.jpg',img)

if __name__ == "__main__":
    #main()
    parm = args()
    
    evalu_img(parm)
    
