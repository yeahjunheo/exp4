# 適当な楽曲から10 秒程度の区間を切り出し，スペクトログラムとクロマグラム
# （横軸を時間とし，縦軸方向に各時間のクロマベクトルを並べたもの）を図示せよ．
# クロマグラムに対して，各フレームでの和音を推定しその結果を示せ．
# 例えばC Major = 0, C# Major = 1, . . . , B Minor = 23 として，
# 横軸を時間，縦軸を和音とした折れ線グラフが利用できる．楽曲の出典を明示すること．

import numpy as np
import math
import matplotlib.pyplot as plt
import librosa

SR = 16000
size_frame = 2 ** 11
size_shift = 16000 // 100
hamming_window = np.hamming(size_frame)


def hz2nn(frequency):
    return int(round(12.0 * (math.log(frequency / 440.0) / math.log(2.0)))) + 69


def chroma_vector(spectrum, frequencies):
    # 0 = C, 1 = C#, 2 = D, ..., 11 = B

    # 12次元のクロマベクトルを作成（ゼロベクトルで初期化）
    cv = np.zeros(12)

    # スペクトルの周波数ビン毎に
    # クロマベクトルの対応する要素に振幅スペクトルを足しこむ
    for s, f in zip(spectrum, frequencies):
        nn = hz2nn(f)
        cv[nn % 12] += abs(s)

    return cv


x, _ = librosa.load("easy_chords.wav", sr=SR)

spectrogram = []
chromagram = []
chords = []

for i in range(0, len(x) - size_frame, size_shift):
    idx = int(i)
    x_frame = x[idx : idx + size_frame]

    # spectrogram
    fft_spec = np.fft.rfft(x_frame * hamming_window)
    spectrogram.append(np.abs(fft_spec))

    # chromagram
    frequencies = np.linspace((SR / 2) / len(fft_spec), SR / 2, len(fft_spec))
    chroma = chroma_vector(fft_spec, frequencies)
    chromagram.append(chroma)

    # chord estimation
    chroma = np.append(chroma, chroma)
    a_root, a_3, a_5 = 1.0, 0.5, 0.8
    chord_estimate = np.zeros(24)
    for i in range(12):
        major = a_root * chroma[i] + a_3 * chroma[i + 4] + a_5 * chroma[i + 7]
        minor = a_root * chroma[i] + a_3 * chroma[i + 3] + a_5 * chroma[i + 7]
        chord_estimate[i] = major
        chord_estimate[i + 12] = minor
    chord = np.argmax(chord_estimate)
    chords.append(chord)


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
# クロマグラムを描画
plt.figure(figsize=(10, 4))
plt.xlabel("Frame")
plt.ylabel("Pitch Class")
plt.title("Chromagram")
plt.imshow(
    np.array(chromagram).T,
    aspect="auto",
    origin="lower",
    cmap="jet",
    interpolation="none",
)
plt.colorbar()
plt.show()
# コード認識結果を描画
plt.figure(figsize=(10, 4))
plt.xlabel("Frame")
plt.ylabel("Chord")
plt.title("Chord Recognition Result")
plt.ylim(-0.5, 23.5)
plt.plot(chords)
plt.show()
