#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 音声ファイルを読み込み，フーリエ変換を行う．
#

# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa


# 配列 a の index 番目の要素がピーク（両隣よりも大きい）であれば True を返す
def is_peak(a, index):
    # （自分で実装すること，passは消す）
    if index == 0 or index == len(a) - 1:
        return False
    return a[index - 1] < a[index] and a[index] > a[index + 1]


# サンプリングレート
SR = 16000

# フレームサイズ
size_frame = 2**11  # 2のべき乗

# シフトサイズ
size_shift = 16000 / 100  # 0.01 秒 (10 msec)

hamming_window = np.hamming(size_frame)

# スペクトログラムを保存するlist
omega = []

# 音声ファイルの読み込み
# x, _ = librosa.load("../ex1/aiueo_long.wav", sr=SR)
# audio for a
# x, _ = librosa.load("../ex1/aiueo_long.wav", sr=SR, offset=0.7, duration=0.7)
# audio for i
# x, _ = librosa.load("../ex1/aiueo_long.wav", sr=SR, offset=1.8, duration=0.7)
# audio for u
# x, _ = librosa.load("../ex1/aiueo_long.wav", sr=SR, offset=2.8, duration=0.7)
# audio for e
# x, _ = librosa.load("../ex1/aiueo_long.wav", sr=SR, offset=3.7, duration=0.7)
# audio for o
# x, _ = librosa.load("../ex1/aiueo_long.wav", sr=SR, offset=4.6, duration=0.7)
# 音声ファイルの読み込み
# x, _ = librosa.load("../ex1/aiueo_short2.wav", sr=SR)
# audio for a
x, _ = librosa.load("../ex1/aiueo_short.wav", sr=SR, offset=0.8, duration=0.3)
# audio for i
# x, _ = librosa.load("../ex1/aiueo_short.wav", sr=SR, offset=1.8, duration=0.3)
# audio for u
# x, _ = librosa.load("../ex1/aiueo_short.wav", sr=SR, offset=2.7, duration=0.3)
# audio for e
# x, _ = librosa.load("../ex1/aiueo_short.wav", sr=SR, offset=3.8, duration=0.3)
# audio for o
# x, _ = librosa.load("../ex1/aiueo_short.wav", sr=SR, offset=4.8, duration=0.3)



# size_shift分ずらしながらsize_frame分のデータを取得
# np.arange関数はfor文で辿りたい数値のリストを返す
# 通常のrange関数と違うのは3つ目の引数で間隔を指定できるところ
# (初期位置, 終了位置, 1ステップで進める間隔)
for i in np.arange(0, len(x) - size_frame, size_shift):
    idx = int(i) 
    x_frame = x[idx : idx + size_frame] * hamming_window
    
    autocorr = np.correlate(x_frame, x_frame, "full")
    
    autocorr = autocorr[len(autocorr) // 2 :]

    peakindices = [i for i in range(len(autocorr)) if is_peak(autocorr, i)]

    peakindices = [i for i in peakindices if i != 0]

    # インデックスに対応する周波数を得る
    if len(peakindices) > 0:
        max_peak_index = max(peakindices, key=lambda index: autocorr[index])
        tau = max_peak_index / SR
        omega.append(1 / tau)

omega_avg = np.mean(omega)

fig = plt.figure()

plt.xlabel("time [s]")
time_axis = np.arange(0, len(omega) * size_shift / SR, size_shift / SR)
plt.ylabel("omega [Hz]")
plt.title(f'mean omega: {omega_avg:.2f} Hz')
plt.plot(time_axis, omega)
plt.ylim(0, 200)

# Add grid lines for more precise measurement
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add minor ticks for more precise measurement
plt.minorticks_on()

plt.show()

# save the plotted graph
fig.savefig("correlate_short_i.png")