import numpy as np
import math
import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile

SR = 16000
size_frame = 512
size_shift = 16000 / 100

x, _ = librosa.load("../ex15/shs-test.wav", sr=SR)

f_s = SR

test_params = [
    (1.0, 1000),
    (1.0, 5000),
    (1.0, 10000),
    (1.0, 50000),
    (1.0, 100000),
    (5.0, 1000),
    (5.0, 5000),
    (5.0, 10000),
    (5.0, 50000),
    (5.0, 100000),
    (10.0, 1000),
    (10.0, 5000),
    (10.0, 10000),
    (10.0, 50000),
    (10.0, 100000),
]

duration = len(x)


def generate_sinusoid(sampling_rate, frequency, duration):
    sampling_interval = 1.0 / sampling_rate
    t = np.arange(sampling_rate * duration) * sampling_interval
    waveform = np.sin(2.0 * math.pi * frequency * t)
    return waveform


fig, ax = plt.subplots(3, 5, figsize=(30, 9), constrained_layout=True)

for i, (D, R) in enumerate(test_params):
    sin_wave = 1 + D * generate_sinusoid(SR, R / f_s, duration / SR)
    x_changed = x * sin_wave
    x_changed = (x_changed * 32768.0).astype("int16")

    filename = f"tremolo_D={D}_R={R}.wav"
    scipy.io.wavfile.write(filename, int(SR), x_changed)

    y, _ = librosa.load(filename, sr=SR)

    fft_spec_y = np.fft.rfft(y)
    fft_log_abs_spec_y = np.log(np.abs(fft_spec_y))

    row = i // 5
    col = i % 5

    ax[row, col].plot(fft_log_abs_spec_y)
    ax[row, col].set_title(f"D={D}, R={R}")
    ax[row, col].set_xlabel("Frequency (Hz)")
    ax[row, col].set_ylabel("Amplitude")

fig.savefig("tremolo_all_params.png")


fig2, ax2 = plt.subplots(3, 5, figsize=(30, 9), constrained_layout=True)

volume_db_orig = []

for j in np.arange(0, len(x) - size_frame, size_shift):
    idx = int(j)
    x_frame = x[idx : idx + size_frame]
    volume = 20 * np.log10(np.mean(x_frame**2))
    volume_db_orig.append(volume)

for i, (D, R) in enumerate(test_params):
    filename = f"tremolo_D={D}_R={R}.wav"
    y, _ = librosa.load(filename, sr=SR)

    volume_db = []

    for j in np.arange(0, len(y) - size_frame, size_shift):
        idx = int(j)
        y_frame = y[idx : idx + size_frame]
        volume = 20 * np.log10(np.mean(y_frame**2))
        volume_db.append(volume)
    
    print(volume_db)

    row = i // 5
    col = i % 5

    time = np.arange(0, len(volume_db)) * size_shift / SR

    ax2[row, col].plot(time, volume_db)
    ax2[row, col].plot(time, volume_db_orig, color="red")
    ax2[row, col].set_title(f"D={D}, R={R}")
    ax2[row, col].set_xlabel("Frame")
    ax2[row, col].set_ylabel("Volume (dB)")

fig2.savefig("volume_all_params_volume.png")
