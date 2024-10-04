#
# 計算機科学実験及演習 4「音響信号処理」
# uses the recorded aiueo file to produce a graph of the volume of the audio file
#

# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load("../01_recording_aiueo/aiueo_long.wav", sr=SR)
# x, _ = librosa.load("../01_recording_aiueo/aiueo_short.wav", sr=SR)

#
# 短時間フーリエ変換
#

# フレームサイズ
size_frame = 512  # 2のべき乗

# シフトサイズ
size_shift = 16000 / 100  # 0.01 秒 (10 msec)

# volume threshhold
threshold = -60  # for aiueo_long.wav
# threshold = -40  # for aiueo_short.wav
flag = False
# container for volume
volume_db = []

# size_shift分ずらしながらsize_frame分のデータを取得
# np.arange関数はfor文で辿りたい数値のリストを返す
# 通常のrange関数と違うのは3つ目の引数で間隔を指定できるところ
# (初期位置, 終了位置, 1ステップで進める間隔)
for i in np.arange(0, len(x) - size_frame, size_shift):

    # 該当フレームのデータを取得
    idx = int(i)  # arangeのインデクスはfloatなのでintに変換
    x_frame = x[idx: idx + size_frame]
    volume = 20 * np.log10(np.mean(x_frame**2))

    # ============= Exercise 6 ===============#
    # aise flag if volume is above threshold
    # as a check for speach in the audio file
    # print to console the time when flag is raised
    # and when the flag is lowered.
    time_in_seconds = i / SR
    if volume > threshold and not flag:
        print(f"Speach started at: {time_in_seconds:.2f} seconds")
        flag = True
    elif volume < threshold and flag:
        print(f"Speach ended at: {time_in_seconds:.2f} seconds")
        flag = False
    # ========================================#

    # フレームの音量を計算
    volume_db.append(volume)

# 画像として保存するための設定
fig = plt.figure()

# plot volume graph
time = np.arange(0, len(volume_db)) * size_shift / SR
plt.plot(time, volume_db)
plt.xlabel("Time (seconds)")
plt.ylabel("Volume (dB)")
plt.title("Volume of aiueo_long.wav")
# plt.title("Volume of aiueo_short.wav")

# 画像の表示
plt.show()

# 【補足】
# 縦軸の最大値はサンプリング周波数の半分 = 16000 / 2 = 8000 Hz となる

# 画像ファイルに保存
fig.savefig("plot-volume-long.png")
# fig.savefig("plot-volume-short.png")
