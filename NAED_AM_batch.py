#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import utils
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import struct
from gen_pc_ply import write_pointcloud
from torch.autograd import Variable
import os
import importlib
import torch.optim as optim
import sys
from data_utils.ModelNetDataLoader import pc_normalize, farthest_point_sample
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# =============================================================================
# def add_sparse_Gauss_noise(vector,noise_mean,noise_var,noise_rate=0.05, n_weight = 0):
#     v_shape = vector.size()
#     #G_noise = torch.empty(v_shape,device = device).normal_(mean=noise_mean,std=noise_var)
#     G_noise = torch.empty(v_shape).normal_(mean=noise_mean,std=noise_var)
#     noise_mask = torch.empty(v_shape).uniform_() > 0.95
#     G_noise = torch.mul(G_noise,noise_mask))
#     res = vector + torch.Tensor(1)#(n_weight * G_noise.to(device))
#     return res
# =============================================================================

shape_names = []
SHAPE_NAME_FILE = 'data/shape_names.txt'
with open(SHAPE_NAME_FILE, "r") as f:
    for tmp in f.readlines():
        tmp = tmp.strip('\n')
        shape_names.append(tmp)


batch_size = 1
output_folder = "output/" # folder path to save the results
save_results = True # save the results to output_folder
use_GPU = True # use GPU, False to use CPU
latent_size = 128 # bottleneck size of the Autoencoder model
num_class = 40
tar_label = 0
optim_steps = 20000
initialization = 'avg'
maximize_logstfmx = False
lr = 5e-6
incor_discri = True    #incorporating discriminator
repeat_time = 10

from pytorch3d.loss import chamfer_distance # chamfer distance for calculating point cloud distance

# =============================================================================
# DATA_PATH = 'data/modelnet40_normal_resampled/'
# TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split='train', normal_channel=False)
# TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split='test', normal_channel=False)
# trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=batch_size, shuffle=True, num_workers=1)
# testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=batch_size, shuffle=False, num_workers=1)
# point_size = trainDataLoader.dataset[0][0].shape[0]
# print(point_size)
# =============================================================================



model_name = os.listdir('log_classifier/classification/pointnet_cls_msg'+'/logs')[0].split('.')[0]
MODEL = importlib.import_module(model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#generator = model.PointCloudAE_4l_de(point_size,latent_size)
classifier = MODEL.get_model(num_class,normal_channel=False)
classifier = classifier.eval()

checkpoint = torch.load('log_classifier/classification/pointnet_cls_msg/checkpoints/best_model.pth',map_location=torch.device(device))
classifier.load_state_dict(checkpoint['model_state_dict'])
print("Classifier: ", classifier, "\n\n")

generator = torch.load('log_generator/NAED.pth',map_location=torch.device(device))

    
print("Generator: ", generator, "\n\n")

if incor_discri == True:
    discriminator = torch.load('log_discriminator/NAED_dis.pth',map_location=torch.device(device))

generator = generator.to(device)
classifier = classifier.to(device)

for param_g in generator.parameters():
    param_g.requires_grad = False
    
for param_c in classifier.parameters():
    param_c.requires_grad = False

# =============================================================================
# ini_ins = torch.from_numpy(TRAIN_DATASET[1][0]).unsqueeze(0).permute(0,2,1)
# ini_latent = generator.encoder(ini_ins)
# input_vector = Variable(ini_latent, requires_grad=True)
# =============================================================================

#Train

# =============================================================================
# act_rec = []
# for steps in range(optim_steps):
#     #input_vector = generator.add_noise_to_vector(input_vector, noise_mean=0, noise_var=1, keep_rate=0.99, n_weight = 1e-3, device=device)
#     pc1 = generator.decoder(input_vector)
#     pc1 = pc1.permute(0,2,1)
#     v2 = generator.encoder(pc1)
#     generated = generator.decoder(v2)
#     generated = generated.permute(0,2,1)
#     activation,bf_sftx,_ = classifier(generated)
#     if maximize_logstfmx == True:
#         tar_actv = activation[0,tar_label]
#         loss = - torch.exp(tar_actv)
#     else:
#         tar_actv = bf_sftx[0,tar_label]
#         loss = - torch.exp(tar_actv)
#     loss.backward()
#     optimizer.step()
#     print("Step: ", steps)
#     print("Activation: ", tar_actv)
#     
#     act_rec.append(tar_actv.clone().detach())
#     
# AM_ins = generated.permute(0,2,1)#generator.decoder(input_vector)
# actv_aft,bf_stfx_aft,_ = classifier(AM_ins.permute(0,2,1))
# print(actv_aft,bf_stfx_aft)
# 
# if use_GPU == False:
#     AM_ins = AM_ins.squeeze().detach().numpy()
# else:
#     AM_ins = AM_ins.squeeze().detach().cpu().numpy()
# write_pointcloud('visu/AM_ins.ply',AM_ins)
# 
# act_rec = np.array(act_rec)
# plt.plot(act_rec)
# plt.savefig('visu/activation.png')
# =============================================================================


for tar_label in range(40):
    for rep in range(repeat_time):
        print('visu/NAED/' + 'AM_' + str(shape_names[tar_label]) + '_' + str(rep) + '.ply')
        if os.path.exists('visu/NAED/' + 'AM_' + str(shape_names[tar_label]) + '_' + str(rep) + '.ply') == True:
            print("AM already generated, skip!\n")
            continue
        
        print("In Processing: ", 'visu/NAED/' + 'AM_' + str(shape_names[tar_label]) + '_' + str(rep) + '.ply')
        
        if initialization == 'ins':
            real_data_path = 'data/modelnet40_normal_resampled/'
            class_path = real_data_path + str(shape_names[tar_label])+ '/'
            class_file = os.listdir(class_path)
            selected_real_data = random.sample(class_file,1)
            print('Loading random instance as initialization...')
            cur_real_data = np.loadtxt(class_path + selected_real_data[0], delimiter=',').astype(np.float32)
            cur_sampled_real = farthest_point_sample(cur_real_data, 1024)
            cur_sampled_real[:, 0:3] = pc_normalize(cur_sampled_real[:, 0:3])
            cur_sampled_real = cur_sampled_real[:, 0:3]
            cur_sampled_real = torch.from_numpy(cur_sampled_real).unsqueeze(0).permute(0,2,1)
            input_vector = Variable(generator.encoder2(generator.encoder1(cur_sampled_real)),requires_grad=True)
        
        elif initialization == 'avg':
            cur_shape_name = shape_names[tar_label]
            avg_ini = np.load('initializations/'+ str(cur_shape_name) + '_avg.npy')
            print("Loading " + str(cur_shape_name) + '_avg.npy as initialization...')
            avg_ini = torch.from_numpy(avg_ini).unsqueeze(0).permute(0,2,1).float().to(device)
            input_vector = Variable(generator.encoder2(generator.encoder1(avg_ini)),requires_grad=True)
            
        elif initialization == 'rdm':
            input_vector = Variable(torch.randn(1, latent_size,device = device), requires_grad=True)
            
        optimizer = optim.Adam([input_vector], lr=lr)
        act_rec = []
        best_gen = torch.zeros_like(input_vector)
        best_actv = -float("Inf")
        
        for steps in range(optim_steps):
            if steps % 50 == 0 and steps > 50:
                act_rec_prev = torch.Tensor(act_rec[-50:-25])
                act_rec_cur = torch.Tensor(act_rec[-25:])
                if torch.mean(act_rec_cur) < torch.mean(act_rec_prev):
                    print("AM stopped, add noise!\n")
                    input_vector = generator.add_noise_to_vector(input_vector, noise_mean=0, noise_var=1e-5, keep_rate=0.0, n_weight = 1, device=device)
            
            pc1 = generator.decoder(input_vector)
            pc1 = pc1.permute(0,2,1)
            v2 = generator.encoder1(pc1)
            v3 = generator.encoder2(v2)
            generated = generator.decoder(v3)
            generated = generated.permute(0,2,1)
            
        # =============================================================================
        #     generated = generator.decoder(input_vector)
        #     generated = generated.permute(0,2,1)
        # =============================================================================
            
            activation,bf_sftx,_ = classifier(generated)
            if maximize_logstfmx == True:
                tar_actv = activation[0,tar_label]
                loss = - torch.exp(tar_actv)
            else:
                tar_actv = bf_sftx[0,tar_label]
                loss = - tar_actv
                
            if incor_discri == True:
                reconstruct_loss = -discriminator(generated)[0]
                loss += reconstruct_loss
                
            loss.backward()
            optimizer.step()
            print("Step: ", steps)
            print("Activation: ", tar_actv)
            
            if tar_actv > best_actv:
                best_actv = tar_actv
                best_gen = generated
                print("Better AM found!")
            
            act_rec.append(tar_actv.clone().detach())
            
        AM_ins = best_gen.permute(0,2,1)#generator.decoder(input_vector)
        actv_aft,bf_stfx_aft,_ = classifier(AM_ins.permute(0,2,1))
        print(actv_aft,bf_stfx_aft)
                
        if use_GPU == False:
            AM_ins = AM_ins.squeeze().detach().numpy()
        else:
            AM_ins = AM_ins.squeeze().detach().cpu().numpy()
        np.save('visu/NAED/' + 'AM_' + str(shape_names[tar_label]) + '_' + str(rep) + '.npy',AM_ins)
        write_pointcloud('visu/NAED/' + 'AM_' + str(shape_names[tar_label]) + '_' + str(rep) + '.ply',AM_ins)
        
        act_rec = np.array(act_rec)
        plt.plot(act_rec)
        plt.savefig('visu/NAED/activation_'+ shape_names[tar_label] +'.png')


# =============================================================================
# def generation(h): # test with a batch of inputs
#     with torch.no_grad():
#         h = h.to(device)
#         output = generator.decoder(h)
#         #loss, _ = chamfer_distance(data, output)
#     return  output.cpu()
# 
# def classification(model, data, num_class=40, vote_num=1):
#     data = data.permute(0,2,1)
#     pred, bf_sftmx , _ = classifier(data)
#     pred_choice = pred.data.max(1)[1]
#     print('Fc3 layer score:\n', bf_sftmx[0][pred_choice])
#     print('Prediction Score:\n', pred[0][pred_choice])
#     print('Predict Result: ',pred_choice, SHAPE_NAMES[pred_choice])
#     return pred_choice
# 
# if(save_results):
#     utils.clear_folder(output_folder)
# 
# startTime = time.time()
# # =============================================================================
# # test_input = torch.from_numpy(testDataLoader.dataset[0][0])
# # write_pointcloud('output/input.ply',test_input)
# # np.save('output/input.npy',test_input)
# # =============================================================================
# test_h = torch.from_numpy(np.random.rand(128)).float()
# output = generation(test_h)
# #print("Loss: ", loss)
# pred = classification(classifier,output)
# output = np.squeeze(output.detach().cpu().numpy())
# np.save('output/random.npy',output)
# write_pointcloud('output/random.ply',output)
# # =============================================================================
# # pc = o3d.geometry.PointCloud()
# # pc.points = o3d.utility.Vector3dVector(output)
# # o3d.io.write_point_cloud('output/test.ply', pc)
# # =============================================================================
# 
# test_time = time.time() - startTime
# print("Processing time :", test_time)
# =============================================================================


        

