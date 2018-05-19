#Hoang Trung Hieu - 1512159
#Do Thanh Phong - 1512398
#Nhap mon  TGMT 15CNTN

#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1


# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')

CLASSES = ('__background__','car', 'bus', 'van', 'others')


# NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_10000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),'detrac':('DETRAC2015_trainval',)}

# def checkOverLap(a,b):
#     val = abs(abs(a[0]-a[3]) - abs(b[0]-b[3]))
#     if val < 5:
#         if (a[4]>b[4]):
#             b[5] = False
#         else:
#             a[5] = False
#     return val

# --------- TP ------
def sBox(b) :
    return abs(float(b[2])-float(b[0]))*abs(float(b[1])-float(b[3]))
def sOverlap(bbox1, bbox2):
    dx = float(min(bbox1[2], bbox2[2])) - float(max(bbox1[0], bbox2[0]))
    dy = float(min(bbox2[3], bbox1[3])) - float(max(bbox1[1], bbox2[1]))
    if(dx >= 0) and (dy >= 0):
        return dx*dy
    return 0    
def isOverLap(bbox1, bbox2) :
    sOver = sOverlap(bbox1, bbox2)
    tmp = sBox(bbox1) + sBox(bbox2) - sOver
    return (sOver/tmp)*100
# -----------

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX

    list_bbox = []
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        list_bbox.append([bbox[0], bbox[1],bbox[2], bbox[3],score,True])
    
    list_bbox.sort()
    ### TP fix filter ------------------------------
    luckynumber = 69.96
    for i in range(len(list_bbox)):
        if(list_bbox[i][5] == True) :
            q = []
            for j in range(len(list_bbox)):
                if (j > i) and (list_bbox[j][5] == True):
                    if(isOverLap(list_bbox[i], list_bbox[j]) >= luckynumber) :
                        q.append(j)                    
            vt = i
            max_score = list_bbox[i][4]
            for e in range(len(q)):
                if(max_score < list_bbox[q[e]][4]):
                    max_score = list_bbox[q[e]][4]
                    vt = q[e]
            if(vt == i):
                for e in range(len(q)):
                    list_bbox[q[e]][5] = False 
            if(vt != i):
                list_bbox[i][5] = False
                for e in range(len(q)):
                    if(q[e] != vt):
                        list_bbox[q[e]][5] = False
    ### ----------------------------------------------



    # for i in list_bbox:
    #     for j in list_bbox:
    #         if (i!=j):
    #             tmp = checkOverLap(i,j)
    #             # if tmp > 50:
    #             #     break

    for bbox in list_bbox:
        # print(bbox)
        if (bbox[5]):
        # if (True):
            cv2.rectangle(im,(bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
            cv2.putText(im, class_name, (bbox[0], bbox[1]), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    COUNT = 0
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
    
    if not os.path.exists('result'):
        os.makedirs('result')
    cv2.imwrite('result/'+image_name,im)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])

    print('Link >>>' ,tfmodel)


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 21,
                          tag='default', anchor_scales=[4, 8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    # im_names = ['pic1.png', 'pic2.png', 'pic3.png',
    #             'pic4.png', 'pic5.png', 'pic6.png']
    im_names = os.listdir('../data/demo')
    numOfFile = str(len(im_names))
    for index, im_name in enumerate(im_names):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('['+numOfFile+'/'+str(index)+'] Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)

    plt.show()
