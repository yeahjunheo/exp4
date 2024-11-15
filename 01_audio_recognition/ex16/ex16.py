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


def shs(spectrum, freqs, candidate_freqs, num_harmonics=5):
    likelihood = np.zeros(len(candidate_freqs))

    for i, f0 in enumerate(candidate_freqs):
        harmonic_sum = 0
        for h in range(1, num_harmonics + 1):
            harmonic_freq = h * f0
            idx = np.argmin(np.abs(freqs - harmonic_freq))
            harmonic_sum += spectrum[idx]
        likelihood[i] = harmonic_sum

    return candidate_freqs[np.argmax(likelihood)]


# x, _ = librosa.load("shs-test-midi.wav", sr=SR)
x, _ = librosa.load("../task3/songs/twinkle-twinkle-little-star-short.mp3", sr=SR)

spectrogram = []
shs_freq = []

freqs = np.linspace(0, SR / 2, size_frame // 2 + 1)
candidate_freqs = np.linspace(50, 400, 500)

for i in range(0, len(x) - size_frame, size_shift):
    idx = int(i)
    x_frame = x[idx : idx + size_frame]
    x_frame = x_frame * hamming_window

    # spectrogram
    fft_spec = np.fft.rfft(x_frame)
    spectrogram.append(np.abs(fft_spec))

    # shs
    estimated_f0 = shs(np.abs(fft_spec), freqs, candidate_freqs)
    shs_freq.append(estimated_f0)

print(shs_freq)


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

# show shs
plt.figure(figsize=(10, 4))
plt.xlabel("Frame")
plt.ylabel("Estimated Pitch (Hz)")
plt.title("SHS Pitch Estimation Over Time")
plt.plot(shs_freq)
plt.show()
