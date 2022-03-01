#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import utils
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import VAEmodel.AED as model
import torch.optim as optim
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import tqdm
import torch.nn.functional as F

# In[2]:


batch_size = 32
output_folder = "output_AED/" # folder path to save the results
save_results = True # save the results to output_folder
use_GPU = True # use GPU, False to use CPU
latent_size = 128 # bottleneck size of the Autoencoder model
d_weight = 1
f_weight = 1
train_VAE_steps = 5
p_discriminator = 1e-4
num_epochs = 150

print("Batch Size: ", batch_size)   
print("Using GPU: ", use_GPU)
print("Latent vector size: ", latent_size)
print("Weight of discriminator loss: ", d_weight)
print("Weight of latent distance loss: ", f_weight)

# In[3]:


#pc_array = np.load("data/train.npy")
#print(pc_array.shape)
def normalize(dataset):
    print("Normalizing ", str(dataset), " ......")
    for i in range(dataset.shape[0]):
        dataset[i] -= np.min(dataset[i])
        dataset[i] /= np.max(dataset[i])
        print('Processing ', i)
    return dataset
# load dataset from numpy array and divide 90%-10% randomly for train and test sets
#train_loader, test_loader = GetDataLoaders(npArray=pc_array, batch_size=batch_size)
DATA_PATH = 'data/modelnet40_normal_resampled/'
TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split='train', normal_channel=False)
TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split='test', normal_channel=False)
trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=batch_size, shuffle=True, num_workers=1)
testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=batch_size, shuffle=False, num_workers=1)

#train_loader = normalize(np.load("data/train.npy"))
#test_loader = normalize(np.load("data/test.npy"))
#Normalizing:
point_size = trainDataLoader.dataset[0][0].shape[0]
print(point_size)
#train_loader = torch.from_numpy(train_loader)
#test_loader = torch.from_numpy(test_loader)


net = model.PointCloudAE(point_size,latent_size)
discriminator = model.Discriminator()

print("Number of Parameters in VAE", sum(p.numel() for p in net.parameters()))
print("Number of Parameters in discriminator", sum(p.numel() for p in discriminator.parameters()))

print(net)

if(use_GPU):
    device = torch.device("cuda:0")
    if torch.cuda.device_count() > 1: # if there are multiple GPUs use all
        net = torch.nn.DataParallel(net)
else:
    device = torch.device("cpu")

net = net.to(device)
discriminator = discriminator.to(device)



from pytorch3d.loss import chamfer_distance # chamfer distance for calculating point cloud distance

net = net.eval()
discriminator = discriminator.eval()


VAE_optimizer = optim.Adam(net.parameters(), lr=0.0005)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr = 0.0005)


def train_epoch(net,discriminator):
    epoch_g_loss = 0
    epoch_d_loss = 0
    for batch_id, data in enumerate(trainDataLoader, 0):
        print("\nTrain number ", batch_id ," batches...")

        #Check discriminator loss
        #if discriminator loss < 0, means: d_fake -> 0, real -> 1 ==> discriminator is dominating
        #skip current step of trining discriminator
        #if discriminator loss > 0, means:  d_fake -> 1, real -> 0 ==> discriminator is deficient
        #Freeze generator for serval steps..
        
        data_for_check = data[0]    #[32,1024,3]
        data_for_check = data_for_check.to(device)
        fake_check = net(data_for_check.permute(0,2,1)) # transpose data for NumberxChannelxSize format
        d_real = discriminator(data_for_check.permute(0,2,1))
        d_fake = discriminator(fake_check.permute(0,2,1))
        d_loss_for_check = d_fake.mean() - d_real.mean()   #ideal prediction: fake -> 0    real -> 1
        
        if d_loss_for_check > 0:
            print("Current d_loss :", d_loss_for_check, "\n")
            print("\nd_loss > 0, discriminator is deficient, skip training generator and training discriminator...")
            #discriminator = discriminator.train()
            #Train discriminator
            data_for_d = data[0]    #[32,1024,3]
            data_for_d = data_for_d.to(device)
            fake = net(data_for_d.permute(0,2,1)) # transpose data for NumberxChannelxSize format
            d_real = discriminator(data_for_d.permute(0,2,1))
            d_fake = discriminator(fake.permute(0,2,1))
            discriminator_optimizer.zero_grad()
            d_loss = d_fake.mean() - d_real.mean()   #ideal prediction: fake -> 0    real -> 1
            print("d_loss: ", d_loss)    
            d_loss.backward()
            discriminator_optimizer.step()
            
            epoch_d_loss += d_loss.item()
            #Print Acc of discrinimator, it should be around 0.5
            acc_gen = sum(d_fake<0.5)/d_fake.shape[0]
            acc_real = sum(d_real>0.5)/d_real.shape[0]
            print("Batch Acc. of Discriminator: ", (acc_gen + acc_real).detach().cpu().numpy()[0]/2)
        
        if d_loss_for_check < 0:
            print("Current d_loss :", d_loss_for_check, "\n")
            #Train VAE, 3 losses are used:
                #1. Reconstruction loss: Chamfer or EMD
                #2. Discriminator loss: try to fool discriminator
                #3. latent distance loss: Chamfer or L2(have a try) on latent distance f1
            #if batch_id % train_VAE_steps == 0:
            print("\nd_loss < 0, discriminator dominates, skip trining discriminator and training VAE..")
            discriminator = discriminator.eval()
            VAE_optimizer.zero_grad()
            data_for_VAE = data[0]
            data_for_VAE = data_for_VAE.to(device)
            output = net(data_for_VAE.permute(0,2,1)) # transpose data for NumberxChannelxSize format
            
            #Loss 1
            chamfer_dis, _ = chamfer_distance(data_for_VAE, output)  #[batch_size,1024,3]
            
            #Loss 2
            d_fake = discriminator(output.permute(0,2,1))
            
            #Loss 3
            f_fake = net.encoder1(output.permute(0,2,1)).permute(0,2,1)#.unsqueeze(-1).repeat(1,1,3)               #[batch_size,512]  #new:[32,128,1024]
            f_real = net.encoder1(data_for_VAE.permute(0,2,1)).permute(0,2,1)#.unsqueeze(-1).repeat(1,1,3)         #[batch_size,512]  #new:[32,128,1024]
            feat_dis, _ = chamfer_distance(f_fake, f_real)
            
            if d_loss_for_check < -0.75:
                with torch.no_grad():
                    for param in discriminator.parameters():
                        param.add_(torch.randn(param.size()).to(device) * p_discriminator)
                    print("Discriminator is over performing, adding noise..")
            
            g_loss = chamfer_dis + f_weight * feat_dis - d_weight * d_fake.mean()
            g_loss.backward()
            VAE_optimizer.step()
            print("Chamfer loss: ", chamfer_dis)
            print("Feature loss: ", feat_dis)
            print("Discriminator loss: ", g_loss)
            epoch_g_loss += g_loss.item()
        
    return epoch_g_loss/batch_id, epoch_d_loss/batch_id



def test_batch(data,net,discriminator): # test with a batch of inputs
    with torch.no_grad():
        data = data.to(device)
        output = net(data.permute(0,2,1))
        chamfer_dis, _ = chamfer_distance(data, output)
        discriminator = discriminator.eval()
        d_fake = discriminator(output.permute(0,2,1))
        g_test_los = chamfer_dis - d_fake.mean()
        print("\nTest Chamfer Distance: ", chamfer_dis)
        print("Test VAE loss: ", g_test_los)
        
    return g_test_los.item(), output.cpu()



def test_epoch(net,discriminator): # test with all test set
    with torch.no_grad():
        epoch_loss = 0
        for j, data in enumerate(testDataLoader, 0):
            if j > 10:
                break
            print("\nTesting ", j, "batches...")    
            data = data[0]
            loss, output = test_batch(data,net,discriminator)
            epoch_loss += loss

    return epoch_loss/j



if(save_results):
    utils.clear_folder(output_folder)



train_VAE_loss_list = []  
train_d_loss_list = []
test_loss_list = []  

best_loss = float('inf')

for i in range(num_epochs) :

    startTime = time.time()
    
    train_VAE_loss, train_d_loss = train_epoch(net,discriminator) #train one epoch, get the average loss
    train_VAE_loss_list.append(train_VAE_loss)
    train_d_loss_list.append(train_d_loss)
    
    test_loss = test_epoch(net,discriminator) # test with test set
    test_loss_list.append(test_loss)

    print("\nSaving parameters...\n")
    torch.save(net, 'log_generator/AED.pth')
    torch.save(discriminator, 'log_discriminator/AED_dis.pth')
    
    epoch_time = time.time() - startTime
    print("epoch " + str(i) + " train VAE loss : " + str(train_VAE_loss) + " train discriminator loss : " + str(train_d_loss) + " test loss : " + str(test_loss) + " epoch time : " + str(epoch_time) + "\n")
    writeString = "epoch " + str(i) + " train VAE loss : " + str(train_VAE_loss) + " train discriminator loss : " + str(train_d_loss) + " test loss : " + str(test_loss) + " epoch time : " + str(epoch_time) + "\n"
    
    # plot train/test loss graph
    plt.plot(train_VAE_loss_list, label="Train VAE loss")
    plt.plot(train_d_loss_list, label="Train Discriminator loss")
    plt.plot(test_loss_list, label="Test loss")
    plt.legend()


    # write the text output to file
    with open(output_folder + "prints.txt","a") as file: 
        file.write(writeString)

    # update the loss graph
    plt.savefig(output_folder + "loss.png")
    plt.close()

    # save input/output as image file
    data_repo = []
    for j in range(10):
        cur_idx = np.random.randint(0,len(TEST_DATASET))
        cur_data = TEST_DATASET._get_item(cur_idx)[0]
        print("Picking up number ", cur_idx, "instance")
        data_repo.append(cur_data)
    test_samples = torch.from_numpy(np.asarray(data_repo))
    loss , test_output = test_batch(test_samples,net,discriminator)
    test_output = test_output.squeeze()
    print(test_samples.shape)
    print(test_output.shape)
    utils.plotPCbatch(test_samples, test_output, show=False, save=True, name = (output_folder  + "epoch_" + str(i)))


        

