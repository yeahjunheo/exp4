# 適当な楽曲のメロディを歌って録音し，その波形，スペクトログラム，
# 音高を示せ．音高は自己相関もしくはSHS で推定し，どちらを推定に用いたのかを示すこと．

import numpy as np
import math
import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile

SR = 16000

x, _ = librosa.load("../ex11/aiueo.wav", sr=SR)

# training files
training_audio = [
    ("a", 1.0, 1.0),
    ("i", 3.0, 1.0),
    ("u", 6.0, 1.0),
    ("e", 8.5, 1.0),
    ("o", 11.0, 1.0),
]

frequency = 6400.0
duration = len(x)


def generate_sinusoid(sampling_rate, frequency, duration):
    sampling_interval = 1.0 / sampling_rate
    t = np.arange(sampling_rate * duration) * sampling_interval
    waveform = np.sin(2.0 * math.pi * frequency * t)
    return waveform


# create voice change
sin_wave = generate_sinusoid(SR, frequency, duration / SR)

sin_wave = sin_wave * 0.9

x_changed = x * sin_wave

x_changed = (x_changed * 32768.0).astype("int16")

filename = f"voice_change_{frequency}.wav"
scipy.io.wavfile.write(filename, int(SR), x_changed)

y, _ = librosa.load(filename, sr=SR)

# spectrum calculations
fig, ax = plt.subplots(len(training_audio), 2, figsize=(10, 2 * len(training_audio)), layout="constrained")

for i, test in enumerate(training_audio):
  start = int(test[1] * SR)
  end = int((test[1] + test[2]) * SR)

  original = x[start:end]
  changed = y[start:end]

  fft_spec_original = np.fft.rfft(original)
  fft_log_abs_spec_original = np.log(np.abs(fft_spec_original))

  fft_spec_changed = np.fft.rfft(changed)
  fft_log_abs_spec_changed = np.log(np.abs(fft_spec_changed))

  ax[i, 0].plot(fft_log_abs_spec_original)
  ax[i, 0].set_title(f"Original {test[0]}")
  ax[i, 0].set_xlabel("frequency [Hz]")
  ax[i, 0].set_ylabel("amplitude")

  ax[i, 1].plot(fft_log_abs_spec_changed)
  ax[i, 1].set_title(f"Changed {test[0]}")
  ax[i, 1].set_xlabel("frequency [Hz]")
  ax[i, 1].set_ylabel("amplitude")

fig.suptitle(f"Voice Change Analysis for Frequency {frequency} Hz")
fig.savefig(f"voice_change_all_{frequency}.png")
