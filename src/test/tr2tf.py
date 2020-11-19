import os
import sys
import cv2
import torch
import onnx
from onnx_tf.backend import prepare
# import tensorflow as tf
import numpy as np
from torch.autograd import Variable
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
from resnet import resnet50,resnet18
from mobilenet import mobilenet_v2
from shufflenet import shufflenet_v2_x1_0

def rename_dict(state_dict):
    state_dict_new = dict()
    for key,value in list(state_dict.items()):
        state_dict_new[key[7:]] = value
    return state_dict_new

def tr2onnx(modelpath):
    # Load the trained model from file
    device = 'cpu'
    # net = shufflenet_v2_x1_0(pretrained=False,num_classes=6)
    net = resnet18(pretrained=False,num_classes=2).to(device)
    # net = mobilenet_v2(pretrained=False,num_classes=6).to(device)
    state_dict = torch.load(modelpath,map_location=device)
    # state_dict = rename_dict(state_dict)
    net.load_state_dict(state_dict)
    net.eval()
    # Export the trained model to ONNX
    dummy_input = Variable(torch.randn(1,224,224,3)) # 8 x 28 picture will be the input to the model
    export_onnx_file = '../models/unsell18.onnx'
    torch.onnx.export(net,
                    dummy_input,
                    export_onnx_file,
                    opset_version=10,
                    do_constant_folding=True, # 是否执行常量折叠优化
                    input_names=["img_input"], # 输入名
                    output_names=["softmax_output"], # 输出名
                    dynamic_axes={"img_input":{0:"batch_size"}, # 批处理变量
                                    "softmax_output":{0:"batch_size"}}
    )

def onnx2tf(modelpath):
    # Load the ONNX file
    model = onnx.load(modelpath)
    # Import the ONNX model to Tensorflow
    tf_rep = prepare(model)
    # Input nodes to the model
    print('inputs:', tf_rep.inputs)
    # Output nodes from the model
    print('outputs:', tf_rep.outputs)
    # All nodes in the model
    print('tensor_dict:')
    # print(tf_rep.tensor_dict)
    # 运行tensorflow模型
    print('Image 1:')
    # img = cv2.imread('/data/detect/breathmask/fg1/1_0.jpg')
    # img = np.transpose(img,(2,0,1))
    # output = tf_rep.run(np.asarray(img, dtype=np.float32)[np.newaxis,:,:, :])
    # print('The digit is classified as ', np.argmax(output))
    tf_rep.export_graph('../models/unsell_resnet18v2.pb')

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


if __name__=='__main__':
    # modelpath = '/data/models/breathmask/bm_mafa_best.pth'
    # tr2onnx(modelpath)
    # modelpath = '../models/breathmask1.onnx'
    # onnx2tf(modelpath)
    tr2onnx('/data/models/unsell_cls/unsell2resnet18_siping_bestv2.pth')
    onnx2tf('../models/unsell18.onnx')