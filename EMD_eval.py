#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 17:31:06 2021

@author: tan
"""

from tools.emd import earth_mover_distance
import numpy as np
import os
import importlib
import torch
import sys
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from data_utils.ModelNetDataLoader import pc_normalize, farthest_point_sample
import random


def Chamfer_dis(generated_data, gt_label_repo, real_data_path, model, use_GPU=True, k=5):
    EMD_total = 0
    AM_suc_num = 0
    for ins in range(generated_data.shape[0]):
        model = model.to(device)
        print("Processing number ", ins, " AM instance ...")
        cur_data = generated_data[ins]
        cur_label = gt_label_repo[ins]
        #Check whether gt_label == pred_label
        cur_data_tmp = torch.from_numpy(cur_data).unsqueeze(0).permute(0,2,1).to(device)
        pred_check, _, _ = model(cur_data_tmp)
        if use_GPU == False:
            pred_label = torch.argmax(pred_check,axis=1)[0].detach().numpy()
        else:
            pred_label = torch.argmax(pred_check,axis=1)[0].detach().cpu().numpy()
        if pred_label != cur_label:
            print("GT is ", cur_label, ', but predicted as ', pred_label)
            print("Label check fails, skip current instance...")
            continue
        else:
            AM_suc_num += 1
            class_path = real_data_path + str(SHAPE_NAMES[cur_label])+ '/'
            class_file = os.listdir(class_path)
            selected_real_data = random.sample(class_file,k)
            real_data_mtx = []
            for i in range(k):
                print("Processing ", i + 1, "of ", k)
                cur_real_data = np.loadtxt(class_path + selected_real_data[i], delimiter=',').astype(np.float32)
                cur_sampled_real = farthest_point_sample(cur_real_data, n_points)
                cur_sampled_real[:, 0:3] = pc_normalize(cur_sampled_real[:, 0:3])
                cur_sampled_real = cur_sampled_real[:, 0:3]
                real_data_mtx.append(cur_sampled_real)
            real_data_mtx = torch.from_numpy(np.asarray(real_data_mtx)).to(device)
            cur_data_mtx = torch.from_numpy(np.tile(np.expand_dims(cur_data,0),(5,1,1))).to(device)
            cur_EMD = torch.mean(earth_mover_distance(cur_data_mtx, real_data_mtx, transpose=False))
            print("Chamfer distance of current instance: ", cur_EMD)
            EMD_total += cur_EMD
    EMD_total /= AM_suc_num
    AM_suc_rate = AM_suc_num / generated_data.shape[0]
    return EMD_total, AM_suc_rate





BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

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

#Load generated data
#datapath = "../AMres/ins/"
datapath = "visu/NAED/"
data_files = os.listdir(datapath)
generated_samples = []
gt_label_repo = []
for f in data_files:
    if f[-4:] == '.npy':
        subscript_prev_idx = f.find('_')
        subscript_later_idx = f.rfind('_')
        class_name = f[subscript_prev_idx + 1: subscript_later_idx]
# =============================================================================
#         subscript_later_idx = f.find('_')
#         class_name = f[subscript_later_idx + 5: -5]
# =============================================================================
        class_label = SHAPE_NAMES.index(class_name)
        gt_label_repo.append(class_label)
        cur_data = np.load(datapath + f)
        generated_samples.append(cur_data)
    
generated_samples = np.asarray(generated_samples)
generated_samples = generated_samples[:]      
                  #[num_ins,1024,3]
                  
#Load classifier
model_name = os.listdir('log_classifier/classification/pointnet2_cls_msg'+'/logs')[0].split('.')[0]
MODEL = importlib.import_module(model_name)
classifier = MODEL.get_model(num_class,normal_channel=False)
classifier = classifier.eval()
checkpoint = torch.load('log_classifier/classification/pointnet2_cls_msg/checkpoints/best_model.pth',map_location=torch.device(device))
classifier.load_state_dict(checkpoint['model_state_dict'])
print("Classifier: ", classifier, "\n\n")


real_data_path = 'data/modelnet40_normal_resampled/'

            
total_FID, AM_suc_rate = Chamfer_dis(generated_samples, gt_label_repo, real_data_path, classifier, use_GPU=True, k=5)
print("AM success rate :", AM_suc_rate)
print("Total FID: ", total_FID)
print("Current model: ", datapath)
