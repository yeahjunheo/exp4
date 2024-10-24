# NMF を実装し，歌声と伴奏の両方を含む適当な楽曲に対して実行せよ．
# その際に，基底の数（k）を調整し，結果がどのように変化するかを観察せよ．
# 分離した音声を求める（位相復元）のではなく，行列UとH をグラフィカルに表示できればよい．

import numpy as np
import librosa
import matplotlib.pyplot as plt

SR = 16000
size_frame = 2**11
size_shift = 16000 // 100
hamming_window = np.hamming(size_frame)

x, _ = librosa.load("nmf_piano_sample.wav", sr=SR)

spectrogram = []

for i in range(0, len(x) - size_frame, size_shift):
    idx = int(i)
    x_frame = x[idx : idx + size_frame]
    x_frame = x_frame * hamming_window

    # spectrogram
    fft_spec = np.fft.rfft(x_frame)
    spectrogram.append(np.abs(fft_spec))
    

spectrogram = np.array(spectrogram).T

# NMF
N, M = spectrogram.shape 
k = 2
max_iter = 100

U  = np.random.rand(N, k)
H = np.random.rand(k, M)

for i in range(max_iter):
    U = U * np.dot(spectrogram, H.T) / np.dot(U, np.dot(H, H.T))
    H = H * np.dot(U.T, spectrogram) / np.dot(np.dot(U.T, U), H)
        
plt.figure()
plt.imshow(U, aspect='auto', origin='lower')
plt.colorbar()
plt.title("U")
plt.show()

plt.figure()
plt.imshow(H, aspect='auto', origin='lower')
plt.colorbar()
plt.title("H")
plt.show()
