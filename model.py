import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet import PointNetEncoder, feature_transform_reguliarzer

'''
PointNet AutoEncoder
Learning Representations and Generative Models For 3D Point Clouds
https://arxiv.org/abs/1707.02392
'''

class PointCloudAE(nn.Module):
    def __init__(self, point_size, latent_size):
        super(PointCloudAE, self).__init__()
        
        self.latent_size = latent_size
        self.point_size = point_size
        
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.latent_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.latent_size)
        
        self.dec1 = nn.Linear(self.latent_size,256)
        self.dec2 = nn.Linear(256,256)
        self.dec3 = nn.Linear(256,self.point_size*3)

    def encoder(self, x): 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_size)
        return x
    
    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.point_size, 3)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def add_noise_to_vector(self, vector, noise_mean, noise_var, keep_rate, n_weight, device):
        with torch.no_grad():
            v_shape = vector.size()
            G_noise = torch.empty(v_shape).normal_(mean=noise_mean,std=noise_var)
            noise_mask = torch.empty(v_shape).uniform_() > keep_rate
            G_noise = torch.mul(G_noise,noise_mask)
            vector.add_(G_noise.to(device) * n_weight)
        return vector

    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=3)    #[batch_size,1024]
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 40)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.features_to_prob = nn.Sequential(
                    nn.Linear(40, 1),
                    nn.Sigmoid()
                )
    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return self.features_to_prob(x)
    
class PointCloudAE_4l_de(nn.Module):
    def __init__(self, point_size, latent_size):
        super(PointCloudAE_4l_de, self).__init__()
        
        self.latent_size = latent_size
        self.point_size = point_size
        
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.latent_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.latent_size)
        
        self.dec1 = nn.Linear(self.latent_size,256)
        self.dec2 = nn.Linear(256,512)
        self.dec3 = nn.Linear(512,1024)
        self.dec4 = nn.Linear(1024,self.point_size*3)

    def encoder(self, x): 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_size)
        return x
    
    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = self.dec4(x)
        return x.view(-1, self.point_size, 3)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def add_noise_to_vector(self, vector, noise_mean, noise_var, keep_rate, n_weight, device):
        with torch.no_grad():
            v_shape = vector.size()
            G_noise = torch.empty(v_shape).normal_(mean=noise_mean,std=noise_var)
            noise_mask = torch.empty(v_shape).uniform_() > keep_rate
            G_noise = torch.mul(G_noise,noise_mask)
            vector.add_(G_noise.to(device) * n_weight)
        return vector