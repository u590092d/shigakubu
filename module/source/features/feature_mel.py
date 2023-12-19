import numpy as np
import librosa
import librosa.display
import os
from torch.utils.data.dataset import random_split

__all__ = [ 
   "feature_mel"
]

def feature_mel(source_data_path,sr=24000,n_fft=512,hop_len=256,n_mels=64,save_flag=False,dist_folder_path=None):
  # メルスペクトログラムの抽出
  mel_data = []
  augmented_data = np.loadtxt(source_data_path,delimiter=',')

  print(augmented_data.shape)
  for audio in augmented_data:
      mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels,center=False)
      log_mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)
      #log_mel_spec = librosa.amplitude_to_db(mel_spec, ref=2.0e-06)
      mel_data.append(log_mel_spec)
  np_meldata = np.array(mel_data)
  print(np_meldata.shape)
  if save_flag and (dist_folder_path != None):
     save_file_name = "feature_mels_sr-{}_nfft-{}_hop-length-{}_nmels-{}.csv".format(sr,n_fft,hop_len,n_mels)
     np.savetxt(os.path.join(dist_folder_path,save_file_name),np_meldata,delimiter=",")
  return np_meldata