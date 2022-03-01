#!/usr/bin/env python
# coding: utf-8


#!/usr/bin/env python
# coding: utf-8


import numpy as np
import time
import utils
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import model
import struct
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from gen_pc_ply import write_pointcloud
from torch.autograd import Variable
import os
import importlib
import torch.optim as optim
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/shape_names.txt'))]

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
tar_label = 7
optim_steps = 5000
avg_initialization = True
maximize_logstfmx = True

from pytorch3d.loss import chamfer_distance # chamfer distance for calculating point cloud distance

DATA_PATH = 'data/modelnet40_normal_resampled/'
TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split='train', normal_channel=False)
TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split='test', normal_channel=False)
trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=batch_size, shuffle=True, num_workers=1)
testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=batch_size, shuffle=False, num_workers=1)
point_size = trainDataLoader.dataset[0][0].shape[0]
print(point_size)



model_name = os.listdir('log_classifier/classification/pointnet_cls_msg'+'/logs')[0].split('.')[0]
MODEL = importlib.import_module(model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#generator = model.PointCloudAE(point_size,latent_size)
classifier = MODEL.get_model(num_class,normal_channel=False)
classifier = classifier.eval()

checkpoint = torch.load('log_classifier/classification/pointnet_cls_msg/checkpoints/best_model.pth',map_location=torch.device(device))
classifier.load_state_dict(checkpoint['model_state_dict'])

print("Classifier: ", classifier, "\n\n")

generator = torch.load('log_generator/best_model_avg.pth',map_location=torch.device(device))

    
print("Generator: ", generator, "\n\n")


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
if avg_initialization == True:
    cur_shape_name = shape_names[tar_label]
    avg_ini = np.load('initializations/'+ str(cur_shape_name) + '_avg.npy')
    print("Loading " + str(cur_shape_name) + '_avg.npy as initialization...')
    avg_ini = torch.from_numpy(avg_ini).unsqueeze(0).permute(0,2,1).float().to(device)
    input_vector = Variable(generator.encoder(avg_ini),requires_grad=True)
else:
    input_vector = Variable(torch.randn(1, latent_size,device = device), requires_grad=True)
optimizer = optim.Adam([input_vector], lr=1e-4)
#Train
act_rec = []
#Train
for steps in range(optim_steps):
    input_vector = generator.add_noise_to_vector(input_vector, noise_mean=0, noise_var=1, keep_rate=0.99, n_weight = 1e-3, device=device)
    generated = generator.decoder(input_vector)
    generated = generated.permute(0,2,1)
    activation,bf_sftx,_ = classifier(generated)
    if maximize_logstfmx == True:
        tar_actv = activation[0,tar_label]
        loss = - torch.exp(tar_actv)
    else:
        tar_actv = bf_sftx[0,tar_label]
        loss = - torch.exp(tar_actv)
    loss.backward()
    optimizer.step()
    print("Step: ", steps)
    print("Activation: ", tar_actv)
    
    act_rec.append(tar_actv.clone().detach())
    
AM_ins = generated.permute(0,2,1)#generator.decoder(input_vector)
actv_aft,bf_stfx_aft,_ = classifier(AM_ins.permute(0,2,1))
print(actv_aft,bf_stfx_aft)

if use_GPU == False:
    AM_ins = AM_ins.squeeze().detach().numpy()
else:
    AM_ins = AM_ins.squeeze().detach().cpu().numpy()
write_pointcloud('visu/AM_ins.ply',AM_ins)

act_rec = np.array(act_rec)
plt.plot(act_rec)


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


        



        

