import numpy as np
import librosa
import librosa.display
import os

__all__ = [
  "augment_data_by_pitch"
]

def augment_data_by_pitch(sr,source_data,source_label,save_flag=False,dist_folder_path=None):
  pitch_shifted_label = []
  pitch_shifted_data = []
  pitch_ran = np.arange(-3,3,0.5)

  for n,audio in enumerate(source_data):
    for i in pitch_ran:
      audio=audio.astype(np.float32)
      shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=i)  # ピッチを上げる
      pitch_shifted_data.append(shifted)
      pitch_shifted_label.append(source_label[n])
  
  if save_flag and (dist_folder_path!=None):
    np.savetxt(os.path.join(dist_folder_path,"augmented_data_by_pitch.csv"),pitch_shifted_data,delimiter=',')
    np.savetxt(os.path.join(dist_folder_path,"augmented_label_by_pitch.csv"),pitch_shifted_label,delimiter=',')
  
  return pitch_shifted_data,pitch_shifted_label