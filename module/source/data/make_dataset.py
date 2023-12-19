import numpy as np
import librosa
import librosa.display
import os
import re
from pathlib import Path
from sklearn.utils import resample
import random 
if __name__ == "__main__":
   import sys
   sys.path.append("~/source")

from ..utils import Segment
from ..utils import SegmentationLabel


__all__=[
   "phoneme_segmentation"
]

class ExtentionException(Exception):
    pass

class EmptyLabelException(Exception):
    pass
class RecordData:
  def __init__(self,record_data):
    self.record_data =record_data

  def pop_extend(self,get_num):
    tmp=[]
    while len(len(tmp)>get_num):
      tmp.append(self.pop())
    np.array(tmp)
    np.flipud(tmp)
    tmp = tmp[len(tmp)-(get_num+1):]
    return np.concatenate(tmp, 0)

  def push(self,push_data):
    return self.record_data.append(push_data)

def read_lab(filename):
    """
    read label file (.lab) generated by Julius segmentation kit and
    return SegmentationLabel object
    """
    try:
        if not re.search(r'\.lab$', filename):
            raise ExtentionException("read_lab supports only .lab")
    except ExtentionException as e:
        print(e)
        return None

    with open(filename, 'r') as f:
        labeldata = [line.split() for line in f if line != '']
        segments = [Segment(tStart=float(line[0]), tEnd=float(line[1]),
                            label=line[2])
                    for line in labeldata]
        return SegmentationLabel(segments)

    
def read_wave_in_jvs(wave_path,label_path,sr,time_span=800,threshold=0.1,target=['a','i','u','e','o']):
  wave_data, _ = librosa.load(wave_path, sr=sr)
  label = read_lab(label_path)
  input_data = []
  input_label_data = []
  for seg in label.segments:
    if seg.label in target:
      start = int(seg.tStart*sr)
      end = int(seg.tEnd*sr)
      if (end - start) <= time_span:
        continue
      tmp_wave = wave_data[start:end]
      wavelen = len(tmp_wave)
      for i in range(0,wavelen//time_span):
        tmp = tmp_wave[time_span*i:time_span*(i+1)]#0.05秒
        tmp = np.array(tmp)
        if np.max(tmp)>threshold:
          input_data.append(tmp)
          input_label_data.append(seg.label)


  return np.array(input_data,dtype=object),np.array(input_label_data,dtype=object)

def read_jvs(source_data_path,folder_num,sr,time_span=800,threshold=0.1,target=['a','i','u','e','o']):
  folder_path = os.path.join(source_data_path,f'jvs{folder_num:03d}','parallel100')
  wave_folder_path = os.path.abspath(os.path.join(folder_path,'wav24kHz16bit'))
  label_folder_path = os.path.abspath(os.path.join(folder_path,'lab','mon'))

  input_data=[]

  input_label_data=[]

  for i in range(1,101):
    filename = f"VOICEACTRESS100_{i:03d}"
    wave_path = os.path.join(wave_folder_path,filename+".wav")
    label_path = os.path.join(label_folder_path,filename+".lab")
    tmp_data,tmp_label=read_wave_in_jvs(wave_path,label_path,sr,time_span,threshold,target=target)
    if(tmp_data.shape[0]!=0):
      input_data.append(tmp_data)
      input_label_data.append(tmp_label)
  input_data = np.concatenate(input_data)
  input_label_data = np.concatenate(input_label_data)
  return np.array(input_data,dtype=object),np.array(input_label_data,dtype=object)


def phoneme_segmentation(source_folder_path,dist_folder_path,sr=24000,frame_len=2048,cut_len=4096,phoneme=['a','i','u','e','o'],save_flag = False):
  #read data from jvs
  save_file_name = ""
  for i,p in enumerate(phoneme):
     save_file_name += p
     if i != len(phoneme):
       save_file_name += "-"
  save_file_name += "{}_{}_{}".format(sr,frame_len,cut_len)
  save_file_label_name = save_file_name + "_label.csv"
  save_file_data_name = save_file_name + "_data.csv"
  read_data = []
  read_label = []
  no_label=[6,28,30,37,58,74,89]
  count=0
  for i in range(1,100):
    if i in no_label:
      continue
    tmp_data,tmp_label = read_jvs(source_folder_path,i,sr,cut_len,target=phoneme)
    center = cut_len//2
    tmp_data = [arr[center-frame_len//2:center+frame_len//2] for arr in tmp_data]
    read_data.append(tmp_data)
    read_label.append(tmp_label)
    count=count+1
    if count >= 100:
      break
  read_data = np.concatenate(read_data)
  read_label = np.concatenate(read_label)
  print(read_data.shape)

  read_data = np.array(read_data,dtype="float64")
  for i,l in enumerate(read_label):
    read_label[i] = phoneme.index(l)%len(phoneme)
  read_label = np.array(read_label,dtype="int64")

  # 各クラスの数をカウント
  class_counts = np.bincount(read_label)
  for cnt in class_counts:
    print(cnt)
  # 最少のクラス数を取得
  min_class_count = np.min(class_counts)

  # 各クラスごとにサンプリング数を均等化
  balanced_data = []
  balanced_label = []

  for class_label in np.unique(read_label):
      class_indices = np.where(read_label == class_label)[0]
      balanced_indices = resample(class_indices, n_samples=min_class_count, random_state=42)

      balanced_data.extend(read_data[balanced_indices])
      balanced_label.extend(read_label[balanced_indices])

  balanced_data = np.array(balanced_data)
  balanced_label = np.array(balanced_label)

  print(balanced_data.shape)
  sample_num = 50000
  if balanced_data.shape[0]>sample_num:
    random_index = random.sample(range(balanced_data.shape[0]),sample_num)
    random_index = np.array(random_index)
    balanced_data = balanced_data[random_index]
    balanced_label = balanced_label[random_index]

  if save_flag:

    np.savetxt(os.path.join(dist_folder_path,save_file_label_name),balanced_label , delimiter=',')
    np.savetxt(os.path.join(dist_folder_path,save_file_data_name),balanced_data , delimiter=',')
       
  return balanced_data,balanced_label


def main():
  get_data,get_label = phoneme_segmentation()
  print(get_data)

if __name__ == "__main__":
   main()
