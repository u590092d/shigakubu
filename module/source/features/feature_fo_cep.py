import numpy as np
import os
import scipy.signal
import matplotlib.pyplot as plt
from ..utils import *
from scipy.fftpack import fft
__all__ = [
  "feature_fo_cep"
]
    
#ケプストラム分析
# 窓関数の設定

#プリエンファシス(高域強調)
def preEmphasis(wave, p=0.97):
    # 係数 (1.0, -p) のFIRフィルタを作成
    return scipy.signal.lfilter([1.0, -p], 1, wave)

def calc_fo_by_LPC(source_data,sr,lpc_order,fo_num):
   
  fo = []
  log_data = None
  log_lpc_data = None
  frame = 10
  sample = len(source_data[0])
  
  for i,wave in enumerate(source_data):
    print(i)
    wave = np.array(wave)
    wave = wave/max(abs(wave))
    p=0.97
    wave = preEmphasis(wave,p)
    hamming = np.hamming(len(wave))
    wave = wave * hamming
    r = autocorr(wave,lpc_order+1)
    a,e = LevinsonDurbin(r,lpc_order)

    spec = np.abs(fft(wave, sample))
    logspec = 20 * np.log10(spec)

    w, h = scipy.signal.freqz(np.sqrt(e), a, sample, "whole")
    lpcspec = np.abs(h)
    loglpcspec = 20 * np.log10(lpcspec)
  
    maxId = scipy.signal.argrelmax(loglpcspec[:sample//2],order=3)
    maxId = maxId[0]
    fo.append(maxId[:fo_num])
    if i == frame:
      log_data = logspec
      log_lpc_data = loglpcspec
  fig = plt.figure()
  fscale = np.fft.fftfreq(sample,d=1.0/sr)[:sample//2]
  ax = fig.add_subplot(1, 1, 1)
  ax.plot(fscale, log_data[:sample//2])
  ax.plot(fscale, log_lpc_data[:sample//2], "r", linewidth=2)
  ax.axvline(fscale[fo[frame][0]], ls = "--", color = "navy")
  ax.axvline(fscale[fo[frame][1]], ls = "--", color = "navy")
  plt.show()
  fo = np.array(fo)
  fo = fo[:,np.newaxis,:]
  return fo

def calc_fo(log_spectrum,fo_num):
  #フォルマント
  fo = []
  for i,wave_data in enumerate(log_spectrum):
    tmp = []
    for j,data in enumerate(wave_data):
      max_index = scipy.signal.argrelmax(data, order=3)
      max_index = max_index[0][0:fo_num]
      tmp.append(max_index)
    fo.append(tmp)

  fig = plt.figure(figsize=(10,6))
  ax = fig.add_subplot(1, 1, 1)
  frame = 10
  ax.plot(log_spectrum[frame][0][0:log_spectrum.shape[2]])
  ax.axvline(fo[frame][0][0], ls = "--", color = "navy")
  ax.axvline(fo[frame][0][1], ls = "--", color = "navy")
  plt.xlabel('Time')
  plt.ylabel('Amplitude')
  plt.title(f'Waveform of frame {frame}')
  plt.show()
  fo = np.array(fo)
  return fo
  
def feature_fo_cep(source_data,sr=24000,frame_len=512,hop_len=256,dim=32,fo_num = 2,lpc_order = 14,save_flag = False, dist_folder_path = None,LPC_flag = True):
  #ケプストラム分析
  # 窓関数の設定
  window = np.hamming(512)

  # フレームごとにケプストラム解析を行う
  cepstrum_data =[]
  origin_data = []
  count = 0
  for wave in source_data:
    cepstrum = []
    tmp = []
    cepstrum_all = []
    log_tmp=[]
   
    count = count+1
    for i in range(0, len(wave) - hop_len, hop_len):
        if (i>len(wave)) or (i+frame_len>(len(wave))):
          break
        
        # 窓関数をかける
        frame = wave[i:i+frame_len] * window
        tmp.append(frame)
        # フーリエ変換
        spectrum = np.fft.fft(frame)
        # 対数振幅スペクトル
        log_spectrum = np.log(np.abs(spectrum))
        log_tmp.append(log_spectrum)
        # 逆フーリエ変換
        ceps = np.fft.ifft(log_spectrum)
        tmp_ceps = np.fft.ifft(log_spectrum)
        
        cepstrum_all.append(tmp_ceps)
        ceps[dim:len(ceps)-dim]=0
        # ケプストラムのリストに追加
        cepstrum.append(ceps)
    origin_data.append(tmp)
    cepstrum_data.append(cepstrum)

  print("test") 
  # ケプストラムの配列に変換
  cepstrum_data = np.array(cepstrum_data)
  print("cepstrum")
  # 声道スペクトルの抽出
  vocal_tract_spectrum = np.exp(np.real(cepstrum_data))
  print("vocal_tract")
  # 声道スペクトルからケプストラムへ変換
  tmp = np.log(vocal_tract_spectrum)
  # ケプストラムから対数振幅スペクトルへ変換
  log_spectrum = np.fft.fft(tmp)
  half_index = log_spectrum.shape[2]//2
  log_spectrum = log_spectrum.real[:,:,0:half_index]
  if LPC_flag: 
    fo = calc_fo_by_LPC(source_data,sr,lpc_order,fo_num)
  else:
    fo = calc_fo(log_spectrum,fo_num)
  return log_spectrum,fo