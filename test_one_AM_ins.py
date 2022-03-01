#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:27:59 2021

@author: tan
"""

import argparse
import numpy as np
import os
import torch
import logging
import sys
import importlib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/shape_names.txt'))] 
    
def contri_to_color(current_data,contri):
    point_contri = np.sum(contri,axis=1)
    max_contri = np.max(point_contri)
    min_contri = np.min(point_contri)
    positiv_scale = np.log(max_contri+1)
    negtiv_scale = np.log(abs(min_contri)+1)
    color_matrix = np.zeros([point_contri.shape[0],3])
    for i in range(point_contri.shape[0]):
        if point_contri[i] < 0:
            color_matrix[i][2] = np.log(abs(point_contri[i])+1) / negtiv_scale
        elif point_contri[i] > 0:
            color_matrix[i][0] = np.log(point_contri[i]+1) / positiv_scale
    colored_data = np.concatenate((current_data,color_matrix),axis = 1)
    return colored_data

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default='pointnet_cls_msg', help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores with voting [default: 3]')
    return parser.parse_args()

def sampling(points, sample_size):
    num_p = points.shape[0]
    index = range(num_p)
    np.random.seed(1)
    sampled_index = np.random.choice(index,size=sample_size)
    sampled = points[sampled_index]
    return sampled

def test(model, loader, num_class=40, vote_num=1):
    if loader[-3:] == 'npy':
        points = np.load(loader)
    elif loader[-3:] == 'txt':
        points = np.loadtxt(loader,delimiter=',')
    if points.shape[1] > 3:
        points = points[:,0:3]
    if points.shape[0] > 1024:
        points = sampling(points,1024)
    points = np.expand_dims(points,0)
    points = torch.from_numpy(points)
    points = points.transpose(2, 1)
    points = points.float()
    classifier = model.eval()
    pred, bf_sftmx , _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    print('Fc3 layer score:\n', bf_sftmx[0][pred_choice])
    print('Prediction Score:\n', pred[0])
    print('Predict Result: ',pred_choice, SHAPE_NAMES[pred_choice])
    return points, pred_choice


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log_classifier/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    log_string('PARAMETER ...')
    log_string(args)

    '''MODEL LOADING'''
    num_class = 40
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)

    classifier = MODEL.get_model(num_class,normal_channel=args.normal)
    #checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth',map_location=torch.device('cpu'))
    classifier.load_state_dict(checkpoint['model_state_dict'])
    #filename = 'data/modelnet40_normal_resampled/wardrobe/wardrobe_0002.txt'
    #filename = 'visu/avg_ini.npy'
    #filename = 'visu/avg.npy'
    #filename = 'transferability/pn1/airplane7.npy'
    filename = '../AM_airplane_0.npy'
    #filename = 'transferability/temp/stairs9.npy'
    #filename = '../untargeted/'
    with torch.no_grad():
        points, pred = test(classifier.eval(), filename, vote_num=args.num_votes)
        
    
if __name__ == '__main__':
    args = parse_args()
    main(args)

    
    