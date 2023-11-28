import numpy as np
import importlib
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
import common_func as cmm
from sklearn.svm import SVC
importlib.reload(cmm)
share_data_params=np.loadtxt('share_data_params.csv', delimiter=',')
sr= share_data_params[0].astype(int)
n_fft= share_data_params[1].astype(int)
hop_length= share_data_params[2].astype(int)
n_mels= share_data_params[3].astype(int)
dataset_mean= share_data_params[4].astype(float)
dataset_std= share_data_params[5].astype(float)
x_size= share_data_params[6].astype(int)
y_size= share_data_params[7].astype(int)
frame_len = share_data_params[8].astype(int)

print(share_data_params)


device = torch.device("cuda")



model_save_path = "model_2dim.pth"
model = cmm.VAE(x_dim=x_size*y_size, z_dim=2).to(device)
model.load_state_dict(torch.load(model_save_path))

model.eval()
points=np.loadtxt('z_points.csv', delimiter=',')
label=np.loadtxt('z_labels.csv', delimiter=',').astype(int)

label = label.astype(int)

labeltoword = ["a","i","u","e","o","a:","i:","u:","e:","o:"]
colors = ["red", "green", "blue", "orange", "purple", "brown", "fuchsia", "grey", "olive", "lightblue"]

random_seed = 123
algorithm = SVC(kernel='rbf', probability=True, random_state=random_seed)
algorithm.fit(points,label)



audio_list = []
(audio,stream) = cmm.realtime_recording(sr)
time.sleep(1)

visualizer = cmm.visualizer2D()

audio_data ,sr= librosa.load("a.wav")
print(audio_data.shape)
latent = cmm.slice_encode(sr,n_fft,hop_length,n_mels,frame_len,dataset_mean,dataset_std,model,device,input_data=audio_data)
print(latent[0].time_start)
print(latent[0].time_end)


for i in latent:
  print(i.predict(algorithm))


tmp_y = [x.predicted_label for x in latent]
tmp_x = [x.time_end for x in latent]

plt.figure(figsize=(10,10))
plt.plot(tmp_x,tmp_y)
plt.show()

print(tmp_x[0])
print(tmp_y[0])
latent_before_sorted = cmm.Points(latent)

latent_sorted = latent_before_sorted.point_sort()

for point in latent_sorted.points:
  print("label:"+str(point.predicted_label)+" start:"+str(round((1/sr)*point.time_start,2))+" end:"+str(round((1/sr)*point.time_end,2)))