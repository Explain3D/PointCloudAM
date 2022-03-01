#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 11:22:46 2021

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
import math
from torch.autograd import Variable
from pytorch3d.loss import chamfer_distance
from tools.emd import earth_mover_distance

import matplotlib.pyplot as plt


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def FID(generated_data, real_data, model, use_GPU):
    
    model = model.to(device)
    
    model = model.eval()

    generated_data = torch.from_numpy(generated_data).unsqueeze(0).permute(0,2,1).to(device)
    
    real_data = torch.from_numpy(real_data).unsqueeze(0).permute(0,2,1).to(device)


    with torch.no_grad():
        pred_g,_,_ = model(generated_data)
        
    if use_GPU == True:
        gen_pred = torch.argmax(pred_g, dim=1).detach().cpu().numpy()[0]
    else:
        gen_pred = torch.argmax(pred_g, dim=1).detach().numpy()[0]
    
    print("Predict generated class: ", SHAPE_NAMES[gen_pred])
    
    if use_GPU == True:
        feature_compare1_g = model.feature_compare1.detach().cpu().numpy()
        feature_compare2_g = model.feature_compare2.detach().cpu().numpy()
    else:
        feature_compare1_g = model.feature_compare1.detach().numpy()
        feature_compare2_g = model.feature_compare2.detach().numpy()
        
    mu1_g = np.mean(feature_compare1_g, axis=1)
    mu2_g = np.mean(feature_compare2_g, axis=1)
    sigma1_g = np.cov(feature_compare1_g, rowvar=False)
    sigma2_g = np.cov(feature_compare2_g, rowvar=False)
    
    with torch.no_grad():
        pred_r,_,_ = model(real_data)
        
    if use_GPU == True:
        real_pred = torch.argmax(pred_r, dim=1).detach().cpu().numpy()[0]
    else:
        real_pred = torch.argmax(pred_r, dim=1).detach().numpy()[0]
        
    print("Predict real class: ", SHAPE_NAMES[real_pred])
    
    if use_GPU == True:
        feature_compare1_r = model.feature_compare1.detach().cpu().numpy()
        feature_compare2_r = model.feature_compare2.detach().cpu().numpy()
    else:
        feature_compare1_r = model.feature_compare1.detach().numpy()
        feature_compare2_r = model.feature_compare2.detach().numpy()
    
    mu1_r = np.mean(feature_compare1_r, axis=1)
    mu2_r = np.mean(feature_compare2_r, axis=1)
    sigma1_r = np.cov(feature_compare1_r, rowvar=False)
    sigma2_r = np.cov(feature_compare2_r, rowvar=False)
    
    FID_dis_fc1 = calculate_frechet_distance(mu1_g, sigma1_g, mu1_r, sigma1_r)
    FID_dis_fc3 = calculate_frechet_distance(mu2_g, sigma2_g, mu2_r, sigma2_r)
    
    FID_dis = (FID_dis_fc1 + FID_dis_fc3)/2

    return FID_dis, gen_pred, real_pred


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
num_class = 40
n_points = 1024
latent_size = 128
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
    
k = 200
real_data_path = 'data/modelnet40_normal_resampled/'

#Load 3Dincept
model_name = os.listdir('log_classifier/classification/pointnet_cls_msg'+'/logs')[0].split('.')[0]
MODEL = importlib.import_module(model_name)
incpt_3d = MODEL.get_model(num_class,normal_channel=False)
incpt_3d = incpt_3d.eval()
checkpoint = torch.load('log_classifier/classification/pointnet_cls_msg/checkpoints/best_model.pth',map_location=torch.device(device))
incpt_3d.load_state_dict(checkpoint['model_state_dict'])
print("3D inception model: ", incpt_3d, "\n\n")

FID_list = []
Chamfer_list = []
EMD_list = []
for std in range(11):
    cur_std = std/10
    print("Current variance: ",cur_std)
    #input_vector = torch.empty(1024,3).normal_(mean=0,std=cur_std)
    input_vector = torch.empty(1024,3).uniform_(-cur_std,cur_std)
    input_vector = input_vector.detach().cpu().numpy()
    print(np.max(input_vector[:,0]),np.max(input_vector[:,1]),np.max(input_vector[:,2]))
    cur_gen_FID = 0
    valid = 0

    real_data_mtx = []
    for i in range(k):
        print("Processing ", i + 1, "of ", k)
        cur_label = np.random.randint(0,40)
        class_path = real_data_path + str(SHAPE_NAMES[cur_label])+ '/'
        class_file = os.listdir(class_path)
        selected_real_data = random.sample(class_file,1)
        cur_real_data = np.loadtxt(class_path + selected_real_data[0], delimiter=',').astype(np.float32)
        cur_sampled_real = farthest_point_sample(cur_real_data, n_points)
        cur_sampled_real[:, 0:3] = pc_normalize(cur_sampled_real[:, 0:3])
        cur_sampled_real = cur_sampled_real[:, 0:3]
        real_data_mtx.append(cur_sampled_real)
        #Cal FID
        cur_FID, gen_pred, real_pred = FID(input_vector, cur_sampled_real, incpt_3d, use_GPU)
        cur_gen_FID += cur_FID
        valid += 1
            
    real_data_mtx = torch.from_numpy(np.asarray(real_data_mtx))
    cur_data_mtx = torch.from_numpy(np.tile(np.expand_dims(input_vector,0),(k,1,1)))
    cur_chamfer_dis, _ = chamfer_distance(cur_data_mtx, real_data_mtx)
    cur_emd_dis = torch.mean(earth_mover_distance(cur_data_mtx.to(device), real_data_mtx.to(device), transpose=False))
    
    print("Chamfer distance of current instance: ", cur_chamfer_dis)
    if valid == 0:    #AM generaton failed
        print("No valid AM instances available..., Fail!")
    cur_gen_FID /= valid
    print("FID of current instance: ", cur_gen_FID)
    print("EMD of current instance: ", cur_emd_dis)
    FID_list.append(cur_gen_FID)
    Chamfer_list.append(cur_chamfer_dis.detach().cpu())
    EMD_list.append(cur_emd_dis.detach().cpu())

plt.plot(FID_list,label='FID')
plt.plot(Chamfer_list,label='Chamfer')
plt.legend()
plt.savefig('visu/FID_Uniform.png')
plt.close()
plt.plot(EMD_list,label='EMD')
plt.legend()
plt.savefig('visu/EMD_Uniform.png')

Chamfer_npy = np.expand_dims(np.asarray(Chamfer_list),0)
FID_npy = np.expand_dims(np.asarray(FID_list),0)
EMD_npy = np.expand_dims(np.asarray(EMD_list),0)
final_npy = np.concatenate((FID_npy,Chamfer_npy,EMD_npy),axis=0)
np.save("visu/Uniform.npy",final_npy)