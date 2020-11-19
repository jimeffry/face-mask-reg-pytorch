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
import numpy as np
from collections import defaultdict 
import time 
import cv2
import argparse
import os 
import sys
import csv
import json
from tqdm import tqdm
from faceattr_model_tf import FaceAttribute
from load_model import UnsellModel
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg


def args():
    parser = argparse.ArgumentParser(description="mtcnn caffe")
    parser.add_argument('--file-in',type=str,dest='file_in',default='None',\
                        help="the file input path")
    parser.add_argument('--out-file',type=str,dest='out_file',default='None',\
                        help="the file output path")
    parser.add_argument('--min-size',type=int,dest='min_size',default=50,\
                        help="scale img size")
    parser.add_argument('--img-path1',type=str,dest='img_path1',default="test1.jpg",\
                        help="img1 saved path")
    parser.add_argument('--img-path2',type=str,dest='img_path2',default="test2.jpg",\
                        help="scale img size")
    parser.add_argument('--base-dir',type=str,dest='base_dir',default="./base_dir",\
                        help="images saved dir")
    parser.add_argument('--gpu', default=None, type=str,help='which gpu to run')
    parser.add_argument('--record-file',type=str,dest='record_file',default=None,\
                        help="output saved file")
    parser.add_argument('--failed-dir',type=str,dest='failed_dir',default="./failed_dir",\
                        help="fpr saved dir")
    parser.add_argument('--faceattr_modelpath', default=None, type=str,help='saved epoch num')
    parser.add_argument('--cmd-type', default="dbtest",dest='cmd_type', type=str,\
                        help="which code to run: videotest,imgtest ")
    parser.add_argument('--threshold', default=0.5, type=float,help='Final confidence threshold')
    return parser.parse_args()

def test_img(args):
    '''
    '''
    img_path1 = args.img_path1
    Model = FaceAttribute(args)
    img_data1 = cv2.imread(img_path1)
    if img_data1 is None:
        print('img is none')
        return None
    #fram_h,fram_w = img_data1.shape[:2]
    tmp,pred_id = Model.inference([img_data1])
    print("pred",tmp)
    score_label = "{}".format(cfg.FaceProperty[pred_id[0]])
    cv2.putText(img_data1,score_label,(int(20),int(30)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
    cv2.imshow("video",img_data1)
    cv2.waitKey(0)

def display(img,id):
    #fram_h,fram_w = img.shape[:2]
    score_label = "{}".format(cfg.FaceProperty[id])
    cv2.putText(img,score_label,(int(20),int(30)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
    cv2.imshow("video",img)
    cv2.waitKey(0)

class SaveCSV(object):
    def __init__(self,savepath,keyword_list):
        '''
        save csv file
        '''
        # 第一次打开文件时，第一行写入表头
            # with open(savepath, "w", newline='', encoding='utf-8') as csvfile:  # newline='' 去除空白行
        self.csvfile = open(savepath, "w", newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.csvfile, fieldnames=keyword_list)  # 写字典的方法
        self.writer.writeheader()  # 写表头的方法
        print("creat csv saver")

    def save(self,itemnew):
        """
        保存csv方法
        :param keyword_list: 保存文件的字段或者说是表头
        :param path: 保存文件路径和名字
        :param item: 要保存的字典对象
        :return:
        """
        try:
            # 接下来追加写入内容
            # with open(savepath, "a", newline='', encoding='utf-8') as csvfile:  # newline='' 一定要写，否则写入数据有空白行
            # writer = csv.DictWriter(csvfile, fieldnames=keyword_list)
            self.writer.writerow(itemnew)  # 按行写入数据
                # print("^_^ write success")

        except Exception as e:
            print("write error==>", e)
            # 记录错误数据
            with open("error.txt", "w") as f:
                f.write(json.dumps(itemnew) + ",\n")
            pass
    def close(self):
        self.csvfile.close()

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
    record_file = args.record_file
    img_dir = args.base_dir
    # Model = FaceAttribute(args)
    Model = UnsellModel(args)
    if file_in is None:
        print("input file is None",file_in)
        return None
    file_rd = open(file_in,'r')
    file_wr = open(result_out,'w')
    key_list = ['filename']
    for tmp in cfg.FaceProperty:
        key_list.append(tmp)
        key_list.append(tmp+'_fg')
    record_w = SaveCSV(record_file,key_list)
    file_cnts = file_rd.readlines()
    total_num = len(file_cnts)
    statistics_dic = defaultdict(lambda : 0)
    for i in tqdm(range(total_num)):
        item_cnt = file_cnts[i]
        item_spl = item_cnt.strip().split(',')
        img_name = item_spl[0]
        real_label = item_spl[1:]
        img_path = os.path.join(img_dir,img_name)
        img_data = cv2.imread(img_path)
        if img_data is None:
            print('img is none',img_path)
            continue
        probility,pred_id = Model.inference([img_data])
        pred_ids = pred_id[0]
        scores = probility[0]
        tmp_item = dict()
        key_id = 0
        tmp_item[key_list[key_id]]=img_name
        for idx in range(cfg.CLS_NUM):
            pred_cls_id = pred_ids[idx]
            real_cls_id = int(real_label[idx])
            pred_name = cfg.FaceProperty[idx]
            #real_name = cfgs.FaceProperty[int(real_label)]
            real_name = cfg.FaceProperty[idx]
            # tmp_item.append(str(scores[idx]))
            # tmp_item.append(real_label[idx])
            key_id+=1
            tmp_item[key_list[key_id]] = "%.3f" % scores[idx]
            key_id+=1
            tmp_item[key_list[key_id]] = real_cls_id
            if real_cls_id:
                statistics_dic[real_name+'_tpfn'] +=1
            else:
                statistics_dic[real_name+'_fptn'] +=1          
            if int(pred_cls_id) == int(real_cls_id)==1:
                statistics_dic[pred_name+'_tp'] +=1
            elif int(pred_cls_id)==int(real_cls_id)==0:
                statistics_dic[pred_name+'_tn'] +=1
            # display(img_data,idx)
        record_w.save(tmp_item)
    for key_name in cfg.FaceProperty:
        tp_fn = statistics_dic[key_name+'_tpfn']
        tp = statistics_dic[key_name+'_tp']
        tn = statistics_dic[key_name+'_tn']
        fp_tn = statistics_dic[key_name+'_fptn']
        fp = fp_tn - tn
        tpr = float(tp) / tp_fn if tp_fn else 0.0
        fpr = float(fp) / fp_tn if fp_tn else 0.0
        precision = float(tp) / (tp+fp) if tp+fp else 0.0
        statistics_dic[key_name+'_tpr'] = tpr
        statistics_dic[key_name+'_fpr'] = fpr
        statistics_dic[key_name+'_P'] = precision
        # file_wr.write('>>> {} result is: tp_fn-{} | fp_tn-{} | tp-{} | fp-{}\n'.format(key_name,\
                    #    tp_fn,fp_tn,tp,fp))
        # file_wr.write('\t tpr:{:.4f} | fpr:{:.4f} | Precision:{:.4f}\n'.format(tpr,fpr,precision))
        file_wr.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(key_name,tpr,fpr,precision,tp_fn,fp_tn,tp,fp))
    file_rd.close()
    file_wr.close()
    record_w.close()

if __name__ == '__main__':
    parms = args()
    cmd_type = parms.cmd_type
    if cmd_type in 'imgtest':
        test_img(parms)
    elif cmd_type in 'evalue':
        evalue(parms)
    else:
        print('Please input right cmd')