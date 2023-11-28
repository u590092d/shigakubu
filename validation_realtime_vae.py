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

while True:
  if keyboard.is_pressed('escape'):
    print("break")
    sys.exit()
    break
  z=cmm.output_data(sr,n_fft,hop_length,n_mels,frame_len,dataset_mean,dataset_std,model,device,stream=stream)
  if z.z!=None:
    value = z.get_z()
    visualizer.visual(value[0][0],value[0][1])
  print("z_predict:"+str(z.predict(predict_algorithm=algorithm)))

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