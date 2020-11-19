# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2019/05/30 19:09
#project: Face attribute
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face attribute 
####################################################
import mxnet as mx
import numpy as np 
import time 
import cv2
import argparse
import os 
import sys
from tqdm import tqdm
import shutil
from breathmask_model import BreathMask
from load_model import UnsellModel
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg


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
    parser.add_argument('--out_file', type=str, default=None,
                        help='image namesf')
    parser.add_argument('--minsize',type=int,default=50,\
                        help="scale img size")
    parser.add_argument('--img-dir',type=str,dest='img_dir',default="./",\
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


def display(img,id):
    #fram_h,fram_w = img.shape[:2]
    score_label = "{}".format(cfgs.FaceProperty[id])
    cv2.putText(img,score_label,(int(20),int(30)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
    cv2.imshow("video",img)
    cv2.waitKey(0)

def evalue(args):
    '''
    calculate the tpr and fpr for all classes
    real positive: tp+fn
    real negnitive: fp+tn
    R = tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    P = tp/(tp+fp)
    '''
    file_in = args.file_in
    result_out = args.out_file
    img_dir = args.img_dir
    save_dir = args.save_dir
    # Model = BreathMask(args)
    Model = UnsellModel(args)
    if file_in is None:
        print("input file is None",file_in)
        return None
    file_rd = open(file_in,'r')
    file_wr = open(result_out,'w')
    file_cnts = file_rd.readlines()
    total_num = len(file_cnts)
    statistics_dic = dict()
    for name in ['wear_mask']:
        statistics_dic[name+'_tp'] = 0
        statistics_dic[name+'_tn'] = 0
        statistics_dic[name+'_fn'] = 0
        statistics_dic[name+'_fp'] = 0
        #statistics_dic[name] = 0
    for i in tqdm(range(total_num)):
        item_cnt = file_cnts[i]
        item_spl = item_cnt.strip().split(',')
        img_name = item_spl[0]
        real_label = item_spl[1]
        img_path = os.path.join(img_dir,img_name)
        img_data = cv2.imread(img_path)
        save_real_dir = os.path.join(save_dir,real_label)
        if not os.path.exists(save_real_dir):
            os.makedirs(save_real_dir)
        save_path = os.path.join(save_real_dir,img_name.split('/')[-1])
        if img_data is None:
            print('img is none',img_path)
            continue
        # img_data = np.expand_dims(img_data,0)
        img_list = [img_data]
        probility,pred_id = Model.inference(img_list)
        # print(pred_id,real_label)
        pred_id = int(pred_id[0])
        if pred_id == 1:
            if pred_id == int(real_label):
                statistics_dic['wear_mask_tp'] +=1
            else:
                statistics_dic['wear_mask_fp'] +=1
                shutil.copyfile(img_path,save_path)
        else:
            if pred_id == int(real_label):
                statistics_dic['wear_mask_tn'] +=1
            else:
                statistics_dic['wear_mask_fn'] +=1
                shutil.copyfile(img_path,save_path)
    for key_name in ['wear_mask']:
        fn = statistics_dic[key_name+'_fn']
        tp = statistics_dic[key_name+'_tp']
        tn = statistics_dic[key_name+'_tn']
        fp = statistics_dic[key_name+'_fp']
        tp_fn = tp+fn
        fp_tn = fp+tn
        tpr = float(tp) / tp_fn if tp_fn else 0.0
        fpr = float(fp) / fp_tn if fp_tn else 0.0
        precision = float(tp) / (tp+fp) if tp+fp else 0.0
        statistics_dic[key_name+'_tpr'] = tpr
        statistics_dic[key_name+'_fpr'] = fpr
        statistics_dic[key_name+'_P'] = precision
        #file_wr.write('>>> {} result is: tp_fn-{} | fp_tn-{} | tp-{} | fp-{}\n'.format(key_name,\
         #               tp_fn,fp_tn,tp,fp))
        #file_wr.write('\t tpr:{:.4f} | fpr:{:.4f} | Precision:{:.4f}\n'.format(tpr,fpr,precision))
        file_wr.write("{}\t{}\t{}\t{}\n".format(key_name,tpr,fpr,precision))
    file_rd.close()
    file_wr.close()

if __name__ == '__main__':
    args = parms()
    evalue(args)