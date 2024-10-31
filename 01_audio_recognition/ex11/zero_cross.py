import numpy as np
import matplotlib.pyplot as plt
import librosa

# Constants
SR = 16000  # Sampling rate
size_frame = 2**11  # Frame size (2^12 samples)
size_shift = int(SR / 100)  # 10 ms frame shift
hamming_window = np.hamming(size_frame)


# Function to calculate the zero-crossing rate
def zero_cross_short(waveform):
    d = np.array(waveform)
    # Sum the number of times the signal changes sign
    return sum([1 if x < 0.0 else 0 for x in d[1:] * d[:-1]])


def is_peak(a, index):
    if index == 0 or index == len(a) - 1:
        return False
    return a[index - 1] < a[index] and a[index] > a[index + 1]


audio_file = "../ex1/aiueo_short.wav"
x, _ = librosa.load(audio_file, sr=SR)

# fundamental frequency and zero-crossing rate operation
f0_values = []
zcr_values = []

for i in np.arange(0, len(x) - size_frame, size_shift):
    idx = int(i)
    x_frame = x[idx : idx + size_frame] * hamming_window

    zcr = zero_cross_short(x_frame)
    zcr_values.append(zcr)

    autocorr = np.correlate(x_frame, x_frame, "full")
    autocorr = autocorr[len(autocorr) // 2 :]
    peakindices = [i for i in range(len(autocorr)) if is_peak(autocorr, i)]
    peakindices = [i for i in peakindices if i != 0]

    if len(peakindices) > 0:
        max_peak_index = max(peakindices, key=lambda index: autocorr[index])
        tau = max_peak_index / SR
        f0 = 1 / tau
    else:
        f0 = 0

    median_zcr = np.median(zcr_values) if len(zcr_values) > 0 else 0
    if zcr > median_zcr:
        f0_values.append(0)
    else:
        f0_values.append(f0)

f0_values = np.array(f0_values)

spectrogram = []

for i in np.arange(0, len(x) - size_frame, size_shift):
    idx = int(i)
    x_frame = x[idx : idx + size_frame] * hamming_window
    fft_spec = np.fft.rfft(x_frame)
    fft_log_abs_spec = np.log(np.abs(fft_spec))
    size_target = int(len(fft_log_abs_spec) * (300 / (SR / 2)))
    fft_log_abs_spec = fft_log_abs_spec[:size_target]
    spectrogram.append(fft_log_abs_spec)


fig, ax = plt.subplots(figsize=(14, 6))

time_axis = np.arange(len(f0_values)) * size_shift / SR
frequency_axis = np.linspace(0, SR / 2, len(spectrogram[0]))

img = ax.imshow(
    np.flipud(np.array(spectrogram).T),
    aspect="auto",
    extent=[0, len(x) / SR, 0, 300],
    cmap="viridis",
    interpolation="nearest",
)
ax.set_ylim(0, 300)

ax.plot(
    time_axis, f0_values, color="r", label="Fundamental Frequency (F0)", linewidth=2
)

ax.set_xlabel("Time [s]")
ax.set_ylabel("Frequency [Hz]")
ax.grid(True)

plt.show()

# fig.savefig("zero_cross.png")
