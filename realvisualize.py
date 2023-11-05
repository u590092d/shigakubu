import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import IPython.display
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
import os
import scipy.signal
import re
import sys
import keyboard
from PIL import Image
import threading
import time
import queue
import pyaudio
import sys
share_data_params=np.loadtxt('share_data_params.csv', delimiter=',')
sr= share_data_params[0].astype(int)
n_fft= share_data_params[1].astype(int)
hop_length= share_data_params[2].astype(int)
n_mels= share_data_params[3].astype(int)
dataset_mean= share_data_params[4].astype(float)
dataset_std= share_data_params[5].astype(float)
x_size= share_data_params[6].astype(int)
y_size= share_data_params[7].astype(int)

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
    
def normal(x):

  x_scaled = (x-x.min())/(x.max()-x.min())

  return x_scaled

def realtime_recording():
    CHUNK = 1024  # 音声データのチャンクサイズ
    FORMAT = pyaudio.paInt16  # 音声データのフォーマット
    CHANNELS = 1  # モノラル
    RATE = 16000  # サンプリングレート

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)
    return p, stream
def stop_recording(audio, stream):
    stream.stop_stream()
    stream.close()
    audio.terminate()

def output_data(stream):
    audio_data = stream.read(3072)
    audio_data = np.frombuffer(audio_data, dtype='int16')/32768
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,center=False)
    log_mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)
    l2 = log_mel_spec - dataset_mean
    np_meldata = l2/dataset_std
    np_meldata = normal(np_meldata)
    tensor_meldata =torch.from_numpy(np_meldata.astype(np.float32)).clone()
    x = tensor_meldata.to(device)
    tmp,z,y=model(x,device)
    z = z.cpu().detach().numpy()
    z = z[0] 
    return z

def read_plot_data(stream,ax):
    audio_data = stream.read(3072)
    audio_data = np.frombuffer(audio_data, dtype='int16')/32768
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,center=False)
    log_mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)
    l2 = log_mel_spec - dataset_mean
    np_meldata = l2/dataset_std
    np_meldata = normal(np_meldata)
    tensor_meldata =torch.from_numpy(np_meldata.astype(np.float32)).clone()
    x = tensor_meldata.to(device)
    tmp,z,y=model(x,device)
    z = z.cpu().detach().numpy()
    z = z[0] 
    return z

device = torch.device("cuda")



model_save_path = "model_2dim.pth"
model = VAE(x_dim=x_size*y_size, z_dim=2).to(device)
model.load_state_dict(torch.load(model_save_path))

model.eval()
points=np.loadtxt('z_points.csv', delimiter=',')
label=np.loadtxt('z_labels.csv', delimiter=',')

label = label.astype(int)

labeltoword = ["a","i","u","e","o","a:","i:","u:","e:","o:"]
colors = ["red", "green", "blue", "orange", "purple", "brown", "fuchsia", "grey", "olive", "lightblue"]

fig =plt.figure(figsize=(5,5))
ax=fig.add_subplot(1,1,1)


for p, l in zip(points, label):
    ax.scatter(p[0], p[1], marker="${}$".format(labeltoword[l]), c=colors[l])
ax.axis("off")
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
plt.savefig("z.png")
plt.clf()
plt.close()




audio_list = []
(audio,stream) = realtime_recording()
time.sleep(1)
fig_anm =plt.figure(figsize=(5,5))
ax=fig_anm.add_subplot(1,1,1)
im = Image.open("z.png")
ax.imshow(im)
while True:
  if keyboard.is_pressed('escape'):
    print("break")
    sys.exit()
    break
  z=output_data(stream)
  
  ax.scatter(z[0],z[1], c="pink", alpha=1, linewidths=2,
      edgecolors="red")

  ax.set_xlim(-2,2)
  ax.set_ylim(-2,2)

  xlim = ax.get_xlim()
  ylim = ax.get_ylim()

  ax.imshow(im, extent=[*xlim, *ylim], aspect='auto', alpha=0.6)
  plt.draw()
  plt.pause(0.1)
  plt.cla()

"""
while  True:
  if keyboard.is_pressed('escape'):
    print("break")
    break
  
  z=output_data(stream)
  ax.scatter(z[0],z[1], c="pink", alpha=1, linewidths=2,
      edgecolors="red")
  ax.set_xlim(-2.5,2.5)
  ax.set_ylim(-2.5,2.5)
  plt.draw()
  plt.pause(0.01)
  plt.cla()
"""