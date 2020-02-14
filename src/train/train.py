#-*- coding:utf-8 -*-
import os
import sys
import cv2
import time
import torch
import argparse
import collections
import logging
from matplotlib import pyplot as plt
# import seaborn as sns;sns.set()
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg
sys.path.append(os.path.join(os.path.dirname(__file__),'../prepare_data'))
from factory import dataset_factory, detection_collate
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
from resnet import resnet50
from mobilenet import mobilenet_v2
from shufflenet import shufflenet_v2_x1_0
sys.path.append(os.path.join(os.path.dirname(__file__),'../losses'))
from multiloss import MultiLoss

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
def params():
    parser = argparse.ArgumentParser(
        description='S3FD face Detector Training With Pytorch')
    train_set = parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset',
                        default='ShangHai',
                        choices=['ShangHai', 'mafa','crowedhuman'],
                        help='Train target')
    parser.add_argument('--basenet',
                        default='vgg16_reducedfc.pth',
                        help='Pretrained base model')
    parser.add_argument('--batch_size',
                        default=2, type=int,
                        help='Batch size for training')
    parser.add_argument('--resume',
                        default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--num_workers',
                        default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--cuda',
                        default=True, type=str2bool,
                        help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning-rate',
                        default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum',
                        default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay',
                        default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma',
                        default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--multigpu',
                        default=False, type=str2bool,
                        help='Use mutil Gpu training')
    parser.add_argument('--save_folder',
                        default='weights/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log_dir',
                        default='../logs',
                        help='Directory for saving logs')
    return parser.parse_args()

def train_net(args):
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
                "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    #*******load data
    train_dataset, val_dataset = dataset_factory(args.dataset)
    train_loader = data.DataLoader(train_dataset, args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=True,
                                collate_fn=detection_collate,
                                pin_memory=True)
    val_batchsize = args.batch_size
    val_loader = data.DataLoader(val_dataset, val_batchsize,
                                num_workers=args.num_workers,
                                shuffle=False,
                                collate_fn=detection_collate,
                                pin_memory=True)
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net = resnet50(pretrained=True,num_classes=6).to(device)
    # net = mobilenet_v2(pretrained=True,num_classes=5).to(device)
    # net = shufflenet_v2_x1_0(pretrained=True,num_classes=2).to(device)
    #print(">>",net)
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        state_dict = torch.load(args.resume,map_location=device)
        if args.multigpu:
            state_dict_new = dict()
            for key,value in list(state_dict.items()):
                state_dict_new[key[7:]] = value
            state_dict = state_dict_new
        net.load_state_dict(state_dict)
    if args.multigpu:
        net = torch.nn.DataParallel(net)
        cudnn.benckmark = True
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                        # weight_decay=args.weight_decay)
    optimizer = optim.Adam(net.parameters(),lr=args.lr)#,weight_decay=args.weight_decay)
    criterion = MultiLoss()
    print('Using the specified args:')
    print(args)
    return net,optimizer,criterion,train_loader,val_loader

def createlogger(lpath):
    if not os.path.exists(lpath):
        os.makedirs(lpath)
    logger = logging.getLogger()
    logname= time.strftime('%F-%T',time.localtime()).replace(':','-')+'.log'
    logpath = os.path.join(lpath,logname)
    hdlr = logging.FileHandler(logpath)
    logger.addHandler(hdlr)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    return logger

def main():
    args = params()
    logger = createlogger(args.log_dir)
    net,optimizer,criterion,train_loader,val_loader = train_net(args)
    step_index = 0
    start_epoch = 0
    iteration = 0
    net.train()
    tmp_diff = 0
    # rgb_mean = np.array([123.,117.,104.])[np.newaxis, np.newaxis,:].astype('float32')
    # rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
    rgb_mean = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis,:].astype('float32')
    rgb_std = np.array([0.225, 0.225, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
    loss_hist = collections.deque(maxlen=200)
    for epoch in range(start_epoch, cfg.EPOCHES):
        #losses = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            save_fg = 0
            if args.cuda:
                images = images.cuda() 
                targets = targets.cuda()
            '''
            targets = targets.numpy()
            images = images.numpy()
            for i in range(targets.shape[0]):
                print(np.shape(images[i]))
                tmp_img = np.transpose(images[i],(1,2,0))
                tmp_img = tmp_img *rgb_std
                tmp_img = tmp_img + rgb_mean
                tmp_img = tmp_img * 255
                tmp_img = np.array(tmp_img,dtype=np.uint8)
                tmp_img = cv2.cvtColor(tmp_img,cv2.COLOR_RGB2BGR)
                h,w = tmp_img.shape[:2]
                gt = targets[i]
                print('gt label:',gt)
                cv2.imshow('src',tmp_img)
                plt.show()
                cv2.waitKey(0)
            '''
            # if iteration in cfg.LR_STEPS:
            #     step_index += 1
            #     adjust_learning_rate(args.lr,optimizer, args.gamma, step_index)
            # t0 = time.time()
            out = net(images)
            # backprop
            # optimizer.zero_grad()
            loss = criterion(out, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # t1 = time.time()
            loss_hist.append(float(loss.item()))
            if iteration % 20 == 0:
                #tloss = losses / 100.0
                #print('tl',loss.data,tloss)
                logger.info('epoch:{} || iter:{} || tloss:{:.6f},lossconf:{:.6f} || lr:{:.6f}'.format(epoch,iteration,np.mean(loss_hist),loss.item(),optimizer.param_groups[0]['lr']))
                # val(args,net,val_loader,logger)
            if iteration != 0 and iteration % 200 == 0:
                # sfile = 'csr_' + args.dataset + '_' + repr(iteration) + '.pth'
                sfile = 'bm_'+args.dataset+'_best.pth'
                tmp_val = val(args,net,val_loader,logger)
                if tmp_val > tmp_diff:
                    save_fg = 1
                    tmp_diff = tmp_val
                if save_fg :
                    logger.info('Saving state, iter: %d' % iteration)
                    torch.save(net.state_dict(),os.path.join(args.save_folder, sfile))
            iteration += 1
        if iteration == cfg.MAX_STEPS:
            break
    torch.save(net.state_dict(),os.path.join(args.save_folder,'bm_'+args.dataset+'_final.pth'))

def val(args,net,val_loader,logger):
    net.eval()
    with torch.no_grad():
        t1 = time.time()
        eq_sum = 0.0
        for batch_idx, (images, targets) in enumerate(val_loader):
            if args.cuda:
                images = images.cuda()
                targets = targets.cuda()
            out = net(images).detach()
            out = F.softmax(out,dim=-1)
            pred = torch.argmax(out,dim=1)
            pos_num = pred.eq(targets)
            tmp = pos_num.sum()
            eq_sum += tmp.item()
            # print(eq_sum)
        t2 = time.time()
        total_num = args.batch_size * (batch_idx+1)
        print('Timer: %.4f' % (t2 - t1),'eq:',eq_sum,'total:',total_num)
        logger.info('test acc:%.4f' % (eq_sum/total_num))
    return eq_sum/total_num



def adjust_learning_rate(init_lr,optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = init_lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
       