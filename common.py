import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import torch.utils as utils
from torchvision import datasets, transforms
from sklearn.utils import resample
import librosa
import librosa.display

class VAE(nn.Module):
    def __init__(self, x_dim, z_dim):
      super(VAE, self).__init__()
      self.x_dim = x_dim
      self.z_dim = z_dim
      self.fc1 = nn.Linear(x_dim, 20)
      self.bn1 = nn.BatchNorm1d(20)
      self.fc2_mean = nn.Linear(20, z_dim)
      self.fc2_var = nn.Linear(20, z_dim)

      self.fc3 = nn.Linear(z_dim, 20)
      self.drop1 = nn.Dropout(p=0.2)
      self.fc4 = nn.Linear(20, x_dim)

    def encoder(self, x):
      x = x.view(-1, self.x_dim)
      x = F.relu(self.fc1(x))
#      x = self.bn1(x)
      mean = self.fc2_mean(x)
      log_var = self.fc2_var(x)
      return mean, log_var

    def sample_z(self, mean, log_var, device):
      epsilon = torch.randn(mean.shape, device=device)
      return mean + epsilon * torch.exp(0.5*log_var)

    def decoder(self, z):
      y = F.relu(self.fc3(z))
      y = self.drop1(y)
      y = torch.sigmoid(self.fc4(y))
      return y

    def forward(self, x, device):
      x = x.view(-1, self.x_dim)
      mean, log_var = self.encoder(x)
      delta = 1e-8
      KL = 0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))
      z = self.sample_z(mean, log_var, device)
      y = self.decoder(z)
      # 本来はmeanだがKLとのスケールを合わせるためにsumで対応
      reconstruction = torch.sum(x * torch.log(y + delta) + (1 - x) * torch.log(1 - y + delta))
      lower_bound = [KL, reconstruction]
      return -sum(lower_bound), z, y