import matplotlib.pyplot as plt
import numpy as np
import librosa

# サンプリングレート
SR = 16000
size_frame = 512
size_shift = 16000 / 100

# x, _ = librosa.load("../ex1/aiueo_long.wav", sr=SR)
x, _ = librosa.load("../ex1/aiueo_short.wav", sr=SR)

volume_db = []

threshold = -40
flag = False

for i in np.arange(0, len(x) - size_frame, size_shift):
    idx = int(i)
    x_frame = x[idx: idx + size_frame]
    volume = 20 * np.log10(np.mean(x_frame**2))

    # ============= Exercise 6 ===============#
    # code to log to terminal when speech starts
    # and ends in the audio files
    time_in_seconds = i / SR
    if volume > threshold and not flag:
        print(f"Speach started at: {time_in_seconds:.2f} seconds")
        flag = True
    elif volume < threshold and flag:
        print(f"Speach ended at: {time_in_seconds:.2f} seconds")
        flag = False
    # ========================================#

    volume_db.append(volume)

fig = plt.figure()

# plot volume graph
time = np.arange(0, len(volume_db)) * size_shift / SR
plt.plot(time, volume_db)
plt.xlabel("Time (seconds)")
plt.ylabel("Volume (dB)")
# plt.title("Volume of aiueo_long.wav")
plt.title("Volume of aiueo_short.wav")

plt.show()

# fig.savefig("plot-volume-long.png")
fig.savefig("plot-volume-short.png")
