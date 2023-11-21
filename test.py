# -*- coding: utf-8 -*-
import numpy as np
import wave
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pydub import AudioSegment

# 音声ファイルの読み込み
sound = AudioSegment.from_file("a.wav", "wav")
# 音声データをリストで抽出
list_sound = sound.get_array_of_samples()
# リストをnumpy配列に変換
data = np.array(list_sound)

# サンプリングレートの取得
wr = wave.open("a.wav", "r")
fs = wr.getframerate()
wr.close()

# プロットの準備
fig, ax = plt.subplots()
x = np.arange(0, len(data) / fs, 1 / fs) # 時間軸
line, = ax.plot(x, data) # 波形
cursor, = ax.plot([0, 0], [-32768, 32767], color="red") # カーソル
ax.set_xlabel("time [s]")
ax.set_ylabel("amplitude")
ax.set_xlim(0, len(data) / fs)
ax.set_ylim(-32768, 32767)
ax.grid()

# PyAudioのストリームを開く
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(sound.sample_width),
                channels=sound.channels,
                rate=sound.frame_rate,
                output=True)

# 音声を再生する関数
def play(stream, data):
    # バイナリデータに変換
    data = data.astype(np.int16).tobytes()
    # ストリームに出力
    stream.write(data)

# アニメーションを更新する関数
def update(i):
    # カーソルの位置を更新
    cursor.set_data([i / fs, i / fs], [-32768, 32767])
    # 音声を再生
    play(stream, data[i:i+1024])
    return cursor,

# アニメーションの作成
ani = animation.FuncAnimation(fig, update, frames=len(data), interval=1000 / fs, blit=True)

# グラフの表示
plt.show()

# ストリームを閉じる
stream.stop_stream()
stream.close()
p.terminate()
