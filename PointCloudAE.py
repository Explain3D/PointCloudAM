#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import utils
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import model
import torch.optim as optim
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import tqdm


# In[2]:


batch_size = 32
output_folder = "output_avg/" # folder path to save the results
save_results = True # save the results to output_folder
use_GPU = True # use GPU, False to use CPU
latent_size = 128 # bottleneck size of the Autoencoder model

print("Batch Size: ", batch_size)
print("Using GPU: ", use_GPU)
print("Latent vector size: ", latent_size)

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


print(net)

if(use_GPU):
    device = torch.device("cuda:0")
    if torch.cuda.device_count() > 1: # if there are multiple GPUs use all
        net = torch.nn.DataParallel(net)
else:
    device = torch.device("cpu")

net = net.to(device)



from pytorch3d.loss import chamfer_distance # chamfer distance for calculating point cloud distance

optimizer = optim.Adam(net.parameters(), lr=0.0005)


def train_epoch():
    epoch_loss = 0
    for batch_id, data in enumerate(trainDataLoader, 0):
        print("Train number ", batch_id ," batches...")
        optimizer.zero_grad()
        data = data[0]
        data = data.to(device)
        output = net(data.permute(0,2,1)) # transpose data for NumberxChannelxSize format
        loss, _ = chamfer_distance(data, output) 
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss/batch_id



def test_batch(data): # test with a batch of inputs
    with torch.no_grad():
        data = data.to(device)
        output = net(data.permute(0,2,1))
        loss, _ = chamfer_distance(data, output)
        
    return loss.item(), output.cpu()



def test_epoch(): # test with all test set
    with torch.no_grad():
        epoch_loss = 0
        for j, data in enumerate(testDataLoader, 0):
            if j > 10:
                break
            print("Testing ", j, "batches...")    
            data = data[0]
            loss, output = test_batch(data)
            epoch_loss += loss

    return epoch_loss/j



if(save_results):
    utils.clear_folder(output_folder)



train_loss_list = []  
test_loss_list = []  

best_loss = float('inf')

for i in range(100) :

    startTime = time.time()
    
    net = net.train()
    train_loss = train_epoch() #train one epoch, get the average loss
    train_loss_list.append(train_loss)
    
    net = net.eval()
    test_loss = test_epoch() # test with test set
    test_loss_list.append(test_loss)
    
    if test_loss < best_loss:
        best_loss = test_loss
        print("\nBetter model found, saving parameters...\n")
        torch.save(net, 'log_generator/best_model_l4.pth')
    
    epoch_time = time.time() - startTime
    print("epoch " + str(i) + " train loss : " + str(train_loss) + " test loss : " + str(test_loss) + " epoch time : " + str(epoch_time) + "\n")
    writeString = "epoch " + str(i) + " train loss : " + str(train_loss) + " test loss : " + str(test_loss) + " epoch time : " + str(epoch_time) + "\n"
    
    # plot train/test loss graph
    plt.plot(train_loss_list, label="Train")
    plt.plot(test_loss_list, label="Test")
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
    net = net.eval()
    loss , test_output = test_batch(test_samples)
    test_output = test_output.squeeze()
    print(test_samples.shape)
    print(test_output.shape)
    utils.plotPCbatch(test_samples, test_output, show=False, save=True, name = (output_folder  + "epoch_" + str(i)))


        

