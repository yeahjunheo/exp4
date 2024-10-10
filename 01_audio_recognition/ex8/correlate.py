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
size_frame = 4096  # 2のべき乗

# シフトサイズ
size_shift = 16000 / 100  # 0.01 秒 (10 msec)

# スペクトログラムを保存するlist
frequency = []

# 音声ファイルの読み込み
x, _ = librosa.load("../ex1/aiueo_long.wav", sr=SR)
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


# size_shift分ずらしながらsize_frame分のデータを取得
# np.arange関数はfor文で辿りたい数値のリストを返す
# 通常のrange関数と違うのは3つ目の引数で間隔を指定できるところ
# (初期位置, 終了位置, 1ステップで進める間隔)
for i in np.arange(0, len(x) - size_frame, size_shift):

    # 該当フレームのデータを取得
    idx = int(i)  # arangeのインデクスはfloatなのでintに変換
    x_frame = x[idx : idx + size_frame]

    # 自己相関が格納された，長さが len(x)*2-1 の対称な配列を得る
    autocorr = np.correlate(x_frame, x_frame, "full")

    # 不要な前半を捨てる
    autocorr = autocorr[len(autocorr) // 2 :]

    # ピークのインデックスを抽出する
    peakindices = [i for i in range(len(autocorr)) if is_peak(autocorr, i)]

    # インデックス0 がピークに含まれていれば捨てる
    peakindices = [i for i in peakindices if i != 0]

    # インデックスに対応する周波数を得る
    # （自分で実装すること）
    if peakindices != []:
        max_peak_index = max(peakindices, key=lambda index: autocorr[index])
        # インデックスに対応する周波数を得る
        # （自分で実装すること）
        tau = max_peak_index / SR
        frequency.append(1 / tau)


# 画像として保存するための設定
fig = plt.figure()


# plot volume graph
time = np.arange(0, len(x)) * size_shift / SR
plt.plot(np.linspace(0,(len(x)-size_frame)/16000, len(frequency)), frequency)
plt.xlabel("Time (seconds)")
plt.ylabel("frequency (Hz)")

plt.show()

# 【補足】
# 縦軸の最大値はサンプリング周波数の半分 = 16000 / 2 = 8000 Hz となる

# 画像ファイルに保存
fig.savefig("plot-correlate-long.png")
