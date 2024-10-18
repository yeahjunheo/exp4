# 適当な楽曲のメロディを歌って録音し，その波形，スペクトログラム，
# 音高を示せ．音高は自己相関もしくはSHS で推定し，どちらを推定に用いたのかを示すこと．

import numpy as np
import math
import librosa
import matplotlib.pyplot as plt

SR = 16000
size_frame = 2**11
size_shift = 16000 // 100
hamming_window = np.hamming(size_frame)


# 配列 a の index 番目の要素がピーク（両隣よりも大きい）であれば True を返す
def is_peak(a, index):
    # （自分で実装すること，passは消す）
    if index == 0 or index == len(a) - 1:
        return False
    return a[index - 1] < a[index] and a[index] > a[index + 1]


# x, _ = librosa.load("shs-test-midi.wav", sr=SR)
x, _ = librosa.load("shs-test.wav", sr=SR)

spectrogram = []
omega = []

for i in range(0, len(x) - size_frame, size_shift):
    idx = int(i)
    x_frame = x[idx : idx + size_frame]
    x_frame = x_frame * hamming_window

    # spectrogram
    fft_spec = np.fft.rfft(x_frame)
    spectrogram.append(np.abs(fft_spec))
    
    # auto correlate
    autocorr = np.correlate(x_frame, x_frame, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]
    peakindices = [i for i in range(len(autocorr) - 1) if is_peak(autocorr, i)]
    peakindices = [i for i in peakindices if i != 0]
    if len(peakindices) > 0:
        max_peak_index = max(peakindices, key=lambda index: autocorr[index])
        tau = max_peak_index / SR
        omega.append(1 / tau)
    else:
        omega.append(0)
    
    
# Show waveform
plt.figure(figsize=(10, 4))
plt.xlabel("Frame")
plt.ylabel("Amplitude")
plt.title("Waveform")
plt.plot(x)
plt.show()

# スペクトログラムを描画
plt.figure(figsize=(10, 4))
plt.xlabel("Frame")
plt.ylabel("Frequency [Hz]")
plt.title("Spectrogram")
# スペクトログラムの周波数方向を2000Hzまでに制限して見やすくする
SPEC_VIEW_MAX = 2000
spectrogram_ = []
for i in range(len(spectrogram)):
    spectrogram_.append(spectrogram[i][: int(len(spectrogram[0]) * SPEC_VIEW_MAX / SR)])
plt.imshow(
    np.array(spectrogram_).T,
    aspect="auto",
    origin="lower",
    cmap="jet",
    interpolation="none",
    extent=[0, len(spectrogram), 0, SPEC_VIEW_MAX],
)
plt.colorbar()

plt.show()

# Show omega
fig = plt.figure(figsize=(10, 4))
plt.xlabel("Frame")
plt.ylabel("Frequency [Hz]")
plt.title("Fundamental Frequency")
plt.plot(omega)
plt.show()
