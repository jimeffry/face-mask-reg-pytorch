import sys
import os
import argparse
import shutil
import json

def putrecord(datalist):
    file = open('result.json','w')
    json.dump(datalist,file,ensure_ascii=False)
    file.close()

def load_json(anno_path):
    f = open(anno_path, 'r', encoding='utf8')
    json_data = json.load(f)
    # 当前图像的路径信息
    infolist = []
    for index, img_info in enumerate(json_data):
        info_dict = dict()
        imgpath = img_info['image_name']
        ct = img_info['category']
        score = img_info['score']
        info_dict['image_name'] = imgpath
        info_dict['category'] = ct
        info_dict['score'] = float(score)
        infolist.append(info_dict)
    return infolist

if __name__=='__main__':
    datas = load_json('result2.json')
    putrecord(datas)
