#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 12:01:08 2021

@author: tan
"""
import numpy as np
import os
import importlib
import torch
import sys
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from data_utils.ModelNetDataLoader import pc_normalize, farthest_point_sample
import random
from gen_pc_ply import write_pointcloud

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def top_k(array, k):
    return array.argsort()[-k:][::-1]

k = 5
selected = 'rdm'
target_cls = 33
num_class = 40
n_points = 1024
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join('data/shape_names.txt'))] 

if torch.cuda.is_available() == True:
    use_GPU = True
else:
    use_GPU = False

if(use_GPU):
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
real_data_path = 'data/modelnet40_normal_resampled/'

#Load classifier
model_name = os.listdir('log_classifier/classification/pointnet_cls_msg'+'/logs')[0].split('.')[0]
MODEL = importlib.import_module(model_name)
classifier = MODEL.get_model(num_class,normal_channel=False)
classifier = classifier.eval()
checkpoint = torch.load('log_classifier/classification/pointnet_cls_msg/checkpoints/best_model.pth',map_location=torch.device(device))
classifier.load_state_dict(checkpoint['model_state_dict'])
print("Classifier: ", classifier, "\n\n")

class_path = real_data_path + str(SHAPE_NAMES[target_cls])+ '/'
class_file = os.listdir(class_path)
print(class_file)

if selected == 'rdm':
    selected_real_data = random.sample(class_file,k)
    for i in range(k):
        print("Processing ", i + 1, "of ", k)
        cur_real_data = np.loadtxt(class_path + selected_real_data[i], delimiter=',').astype(np.float32)
        cur_sampled_real = farthest_point_sample(cur_real_data, n_points)
        cur_sampled_real[:, 0:3] = pc_normalize(cur_sampled_real[:, 0:3])
        cur_sampled_real = cur_sampled_real[:, 0:3]
        write_pointcloud('visu/playground/' + selected + '_' + str(i) + '.ply', cur_sampled_real)
        
elif selected == 'top':
    classifier = classifier.to(device)
    activation_recorder = []
    data_pool = []
    for i in range(len(class_file)):
        print("Processing ", i + 1, "of ", len(class_file))
        cur_real_data = np.loadtxt(class_path + class_file[i], delimiter=',').astype(np.float32)
        cur_sampled_real = farthest_point_sample(cur_real_data, n_points)
        cur_sampled_real[:, 0:3] = pc_normalize(cur_sampled_real[:, 0:3])
        cur_sampled_real = cur_sampled_real[:, 0:3]
        cur_data_tmp = torch.from_numpy(cur_sampled_real).unsqueeze(0).permute(0,2,1).to(device)
        _, bf_sftmx, _ = classifier(cur_data_tmp)
        activation_recorder.append(bf_sftmx[0][target_cls])
        data_pool.append(cur_sampled_real)
    activation_recorder = np.array(activation_recorder)
    top_idx = top_k(activation_recorder, 5)
    n = 0
    for j in top_idx:
        print("Saving top ", n, '...')
        n += 1
        write_pointcloud('visu/playground/' + selected + '_' + str(n) + '.ply', data_pool[j])
    
