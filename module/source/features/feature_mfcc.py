import librosa
import sklearn
import numpy as np
from collections import defaultdict
import scipy.signal
from ..utils import *

__all__= [
  "feature_mfcc"
]

def mfcc(wave,sr,n_fft):
  mfcc = librosa.feature.mfcc(wave,sr=sr,n_fft=n_fft)
  mfcc = np.average(mfcc,axis=1)
  mfcc = mfcc.flatten()
  mfcc = mfcc.tolist()
  mfcc.pop(0)
  mfcc = mfcc[:12]
  return np.array(mfcc)

def feature_mfcc(source_data,sr):
  mfccs = []
  for wave in source_data:
    mfcc = mfcc(wave,sr,n_fft=len(wave))
    mfccs.append(mfcc)
  return mfccs

