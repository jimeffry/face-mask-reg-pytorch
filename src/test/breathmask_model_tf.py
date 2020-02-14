###############################################
#created by :  lxy
#Time:  2020/1/7 10:09
#project: head detect
#rversion: 0.1
#tool:   python 3.6
#modified:
#description  histogram
####################################################
import os
import cv2
import sys
import numpy as np
import argparse
from tqdm import tqdm
import time
from tensorflow.python.platform import gfile
import tensorflow as tf

def parms():
    parser = argparse.ArgumentParser(description='s3df demo')
    parser.add_argument('--save_dir', type=str, default='tmp/',
                        help='Directory for detect result')
    parser.add_argument('--modelpath', type=str,
                        default='weights/s3fd.pth', help='trained model')
    parser.add_argument('--threshold', default=0.1, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--ctx', default=True, type=bool,
                        help='gpu run')
    parser.add_argument('--img_dir', type=str, default='tmp/',
                        help='Directory for images')
    parser.add_argument('--file_in', type=str, default='tmp.txt',
                        help='image namesf')
    return parser.parse_args()

class BreathMask(object):
    def __init__(self,args):
        self.loadmodel(args.breath_modelpath)
        
    def loadmodel(self,mpath):
        tf_config = tf.ConfigProto()
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        #tf_config.gpu_options = gpu_options
        tf_config.gpu_options.allow_growth=True  
        tf_config.log_device_placement=False
        self.sess = tf.Session(config=tf_config)
        # self.sess = tf.Session()
        modefile = gfile.FastGFile(mpath, 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(modefile.read())
        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, name='breathmask_graph') 
        # tf.train.write_graph(graph_def, './', 'breathtest.pbtxt', as_text=True)
        # print("************begin to print graph*******************")
        # op = self.sess.graph.get_operations()
        # for m in op:
        #     # if 'input' in m.name or 'output' in m.name or 'confidence' in m.name:
        #     print(m.name)#m.values())
        # print("********************end***************")
        self.input_image = self.sess.graph.get_tensor_by_name('breathmask_graph/images:0') #img_input
        self.conf_out = self.sess.graph.get_tensor_by_name('breathmask_graph/probs:0') #softmax_output

    def propress(self,imgs):
        rgb_mean = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis,:].astype('float32')
        rgb_std = np.array([0.225, 0.225, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        img_out = []
        for img in imgs:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img.astype('float32')
            img /= 255.0
            img -= rgb_mean
            img /= rgb_std
            img = np.transpose(img,(2,0,1))
            img_out.append(img)
        return np.array(img_out)

        
    def inference(self,imglist):
        t1 = time.time()
        bt_img = self.propress(imglist)
        # output = []
        # for i in range(bt_img.shape[0]):
        #     tmp_output = self.sess.run([self.conf_out],feed_dict={self.input_image:np.expand_dims(bt_img[i],0)})
        #     output.append(tmp_output[0][0])
        output = self.sess.run([self.conf_out],feed_dict={self.input_image:bt_img})
        # t2 = time.time()
        # print("debug*********",np.shape(output))
        output = np.array(output[0])
        pred_cls = np.argmax(output,axis=1)
        scores = [output[idx,pred_cls[idx]] for idx in range(output.shape[0])]
        t3 = time.time()
        print('consuming:',t3-t1)
        # showimg = self.label_show(bboxes,imgorg)
        return scores,pred_cls

def convert_pbtxt_to_pb(filename):
    from google.protobuf import text_format
    """Returns a `tf.GraphDef` proto representing the data in the given pbtxt file.
    Args:
    filename: The name of a file containing a GraphDef pbtxt (text-formatted
      `tf.GraphDef` protocol buffer data).
    """
    with tf.gfile.FastGFile(filename, 'r') as f:
        graph_def = tf.GraphDef()
        img_input = tf.placeholder(dtype=tf.float32,shape=[None,3,112,112], name='img_input') 
        file_content = f.read()

        # Merges the human-readable string in `file_content` into `graph_def`.
        text_format.Merge(file_content, graph_def)
        tf.train.write_graph( graph_def , './' , 'breathtest.pb' , as_text = False )
  


if __name__ == '__main__':
    # args = parms()
    convert_pbtxt_to_pb('../breathtest.pbtxt')
