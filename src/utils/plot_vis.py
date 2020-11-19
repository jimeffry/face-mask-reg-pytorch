import numpy as np 
import csv
import os
import sys
import argparse
from collections import defaultdict
from matplotlib import pyplot as plt
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
    parser.add_argument('--basename', default='faceattr', type=str,help='save name')
    return parser.parse_args()

def plot_data_mxn(datadict,name,keys):
    num_bins = 10
    row_num = 4
    col_num = 6
    total_num = len(datadict.keys())
    fig, axes = plt.subplots(nrows=row_num, ncols=col_num,figsize=(12,12),constrained_layout=True)
    plt.rcParams.update({"font.size":4})
    # plt.tick_params(labelsize=5)
    # plt.xticks(fontproperties = 'Times New Roman', size = 2)
    # plt.yticks(fontproperties = 'Times New Roman', size = 2)
    for i in range(row_num):
        for j in range(col_num):
            # print(i,j,keys[0])
            bin_cnt,bin_data,patchs = axes[i,j].hist(datadict[str(keys[i*col_num+j])],num_bins,normed=0,color='blue',cumulative=0) #range=(0.0,max_bin)
            # axes[i,j].set_xlabel('score')
            # axes[i,j].set_ylabel('num')
            axes[i,j].set_title('%s' % str(keys[i*col_num+j]),fontdict={'family' : 'Times New Roman', 'size'   : 6})
            label = str(keys[i*col_num+j])
            axes[i,j].annotate(label, (0.2, 0.2), xycoords='axes fraction', va='center')
            # axes[i,j].grid(True)
            if (i*col_num +j+1)== total_num:
                break
    plt.savefig('../logs/%s.png' % name,format='png')
    plt.show()

def plot_data(datadict,name,keys):
    num_bins = 10
    row_num = 1 #4
    col_num = 2 #6
    total_num = len(datadict.keys())
    fig, axes = plt.subplots(nrows=row_num, ncols=col_num,figsize=(12,12),constrained_layout=True)
    plt.rcParams.update({"font.size":4})
    # plt.tick_params(labelsize=5)
    # plt.xticks(fontproperties = 'Times New Roman', size = 2)
    # plt.yticks(fontproperties = 'Times New Roman', size = 2)
    for i in range(row_num):
        for j in range(col_num):
            # print(i,j,keys[0])
            bin_cnt,bin_data,patchs = axes[j].hist(datadict[str(keys[i*col_num+j])],num_bins,normed=0,color='blue',cumulative=0) #range=(0.0,max_bin)
            # axes[i,j].set_xlabel('score')
            # axes[i,j].set_ylabel('num')
            axes[j].set_title('%s' % str(keys[i*col_num+j]),fontdict={'family' : 'Times New Roman', 'size'   : 6})
            label = str(keys[i*col_num+j])
            axes[j].annotate(label, (0.2, 0.2), xycoords='axes fraction', va='center')
            # axes[i,j].grid(True)
            if (i*col_num +j+1)== total_num:
                break
    plt.savefig('../logs/%s.png' % name,format='png')
    plt.show()

def plot_name2score(filein,name):
    '''
    datadict: input data. keys are face attributes, values are scores
    '''
    tp_data_dict,fp_data_dict = getCsvdata(filein)
    name_p = '%s_postive' % name
    name_f = '%s_negtive' % name
    plot_data(tp_data_dict,name_p,cfg.FaceProperty)
    plot_data(fp_data_dict,name_f,cfg.FaceProperty)

def getCsvdata(filein):
    '''
    dataformat: filename, keys...
    return: data_dict
    '''
    # f_in = open(filein,'r')
    # dict_keys = f_in.readline().strip().split(',')
    # f_in.close()
    f_in = open(filein,'r')
    # print('read data dict_keys:',dict_keys)
    # list_out = []
    # for name in dict_keys:
    #     list_out.append([])
    # data_dict = dict(zip(dict_keys,list_out))
    data_p_dict = defaultdict(list)
    data_f_dict = defaultdict(list)
    reader = csv.DictReader(f_in)
    for f_item in reader:
        #print(f_item['filename'])
        for cur_key in cfg.FaceProperty:
            if int(f_item[cur_key+'_fg'])==1:
                data_p_dict[cur_key].append(float(f_item[cur_key]))
            else:
                data_f_dict[cur_key].append(float(f_item[cur_key]))
    f_in.close()
    return data_p_dict,data_f_dict

if __name__=='__main__':
    parms = args()
    file_in = parms.file_in
    savename = parms.basename
    plot_name2score(file_in,savename)