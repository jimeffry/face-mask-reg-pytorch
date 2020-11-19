import numpy as np 
from matplotlib import pyplot as plt 
import os
import sys
import cv2
import tqdm
import shutil

def plothist(datadict):
    # xdata = datadict.keys()
    # ydata = []
    # for tmp in xdata:
    #     ydata.append(datadict[tmp])
    print('total plt:',len(datadict))
    fig, ax = plt.subplots(1, 1, figsize=(9, 3), sharey=True)
    # ax.bar(xdata,ydata)
    xn,bd,paths = ax.hist(datadict,bins=20)
    fw = open('../data/cars_min.txt','w')
    for idx,tmp in enumerate(xn):
        fw.write("{}:{}\n".format(tmp,bd[idx]))
    fw.close()
    plt.savefig('../data/cars_min.png',format='png')
    plt.show()

def get_txtfile(imgdir,outfile):
    fw = open(outfile,'w')
    datas = []
    f1_cnts = os.listdir(imgdir)
    tmp_name = imgdir.split('/')[-1]
    for i in tqdm.tqdm(range(len(f1_cnts))):
        tmp_f = f1_cnts[i].strip()
        tmpdir = os.path.join(imgdir,tmp_f)
        f2_cnts = os.listdir(tmpdir)
        if 'bg' in tmp_f:
            label = 0
        else :
            label = 1
        for imgname in f2_cnts:
            # imgpath = os.path.join(tmpdir,imgname.strip())
            # img = cv2.imread(imgpath)
            # h,w = img.shape[:2]
            # datas.append(min(h,w))
            tmppath = os.path.join(tmp_name+'/'+tmp_f,imgname)
            fw.write("{},{}\n".format(tmppath,label))
    # plothist(datas)


def getdatalist(infile,outfile1,outfile2):
    fin = open(infile,'r')
    fw = open(outfile1,'w')
    fw2 = open(outfile2,'w')
    cnt = 0
    fcnts = fin.readlines()
    total = len(fcnts)
    valc = total-40
    for tmp in fcnts:
        cnt+=1
        imgname = tmp.strip()
        if cnt <60:
            fw2.write(imgname+'\n')
        elif cnt < valc:
            fw.write(imgname+'\n')
        else:
            fw2.write(imgname+'\n')
    fw.close()
    fw2.close()

def saveAndresize(imgdir,savedir):
    '''
    imgdir: dir1/dir2/imgs.png
    savedir:
    '''
    fgdir = os.path.join(savedir,'fg_imgs')
    bgdir = os.path.join(savedir,'bg_imgs')
    if not os.path.exists(fgdir):
        os.makedirs(fgdir)
    if not os.path.exists(bgdir):
        os.makedirs(bgdir)
    f1_cnts = os.listdir(imgdir)
    id_cnt = 6210
    for i in tqdm.tqdm(range(len(f1_cnts))):
        tmp_f = f1_cnts[i].strip()
        tmpdir = os.path.join(imgdir,tmp_f)
        if 'fg' in tmp_f:
            save_dir = fgdir
        else:
            save_dir = bgdir
        f2_cnts = os.listdir(tmpdir)
        for imgname in f2_cnts:
            id_cnt+=1
            imgpath = os.path.join(tmpdir,imgname.strip())
            img = cv2.imread(imgpath)
            if img is None:
                print(imgpath)
                continue
            img = cv2.resize(img,(224,224))
            savepath = os.path.join(save_dir,str(id_cnt)+'.jpg')
            cv2.imwrite(savepath,img)

def get_smokedata(imgdir,outfile):
    fw = open(outfile,'w')
    datas = []
    f1_cnts = os.listdir(imgdir)
    for i in tqdm.tqdm(range(len(f1_cnts))):
        tmp_f = f1_cnts[i].strip()
        tmpdir = os.path.join(imgdir,tmp_f)
        f2_cnts = os.listdir(tmpdir)
        if tmp_f == 'smoking_images':
            label = 1
        elif tmp_f == 'calling_images' :
            label = 2
        else:
            label = 0
        for imgname in f2_cnts:
            # imgpath = os.path.join(tmpdir,imgname.strip())
            # img = cv2.imread(imgpath)
            # h,w = img.shape[:2]
            # datas.append(min(h,w))
            tmppath = os.path.join(tmp_f,imgname)
            fw.write("{},{}\n".format(tmppath,label))
    # plothist(datas)

def getunselldata(imgdir,outfile):
    '''
    imgdir: fg1/images bg1/images
    outfile: fg1/imagename,label_num
    '''
    fw = open(outfile,'w')
    datas = []
    f1_cnts = os.listdir(imgdir)
    for i in tqdm.tqdm(range(len(f1_cnts))):
        tmp_f = f1_cnts[i].strip()
        fw.write(tmp_f+'\n')
    fw.close()

def splitclassfromimg(imgdir,savedir):
    '''
    imgdir: images saved, image_name+bg_score.jpg
    '''
    fgdir = os.path.join(savedir,'fg')
    bgdir = os.path.join(savedir,'bg')
    if not os.path.exists(fgdir):
        os.makedirs(fgdir)
    if not os.path.exists(bgdir):
        os.makedirs(bgdir)
    fcnts = os.listdir(imgdir)
    for tmp in fcnts:
        tmpdir = os.path.join(imgdir,tmp.strip())
        tmpcnts = os.listdir(tmpdir)
        for imgname in tmpcnts:
            tmpsplit = imgname.strip().split('-')
            imgpath = os.path.join(tmpdir,imgname.strip())
            if float(tmpsplit[-1][:-4]) < 0.5:
                savepath = os.path.join(fgdir,imgname.strip())
            else:
                savepath = os.path.join(bgdir,imgname.strip())
            shutil.copyfile(imgpath,savepath)

if __name__=='__main__':
    # get_data('/data/detect/zhatu_car','../data/rubbish_car_all.txt')
    # getdatalist('../data/rubbish_car_all.txt','../data/rubbish_train.txt','../data/rubbish_val.txt')
    # saveAndresize('/data/detect/rubbish_car','/data/detect/zhatu_car')
    # get_smokedata('/data/videos/ehualu/train','../data/ehualu_train.txt')
    # getunselldata('/data/detect/siping_caiji/cls_test','../data/unsell_val.txt')
    # saveAndresize('/data/detect/siping_caiji/cls_test','/data/detect/siping_caiji/')
    # get_txtfile('/data/detect/siping_caiji/test_data','../data/unsell_val.txt')
    # saveAndresize('/data/detect/siping_caiji/tobedone','/data/detect/siping_caiji/cls_done')
    # getunselldata('/data/detect/siping_caiji/cls_done/bg_imgs','../data/unsell_tobedone.txt')
    # getunselldata('/data/videos/ehualu/test','../data/ehualu_test.txt')
    splitclassfromimg('/data/detect/siping_caiji/images','/data/detect/siping_caiji/images_done2')
    # get_txtfile('/data/detect/siping_caiji/train_data','../data/unsell_train.txt')
    # get_txtfile('/data/detect/siping_caiji/test_data','../data/unsell_val.txt')
