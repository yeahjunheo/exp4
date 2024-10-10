#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# ケプストラムを計算する関数
#

# see also `src/plot_spectrum.py`, `src/cepstrum.py`

import numpy as np
import matplotlib.pyplot as plt
import librosa


# スペクトルを受け取り，ケプストラムを返す関数
def get_cepstrum(amplitude_spectrum):

	log_spectrum = np.log(amplitude_spectrum)
	cepstrum = np.fft.fft(log_spectrum)

	return cepstrum


# サンプリングレート
SR = 16000

# choose from ["catena", "separato"]

# load sound file (.wav)
x, _ = librosa.load('../ex1/aiueo_long.wav', sr=SR)


# Step1,2: 振幅スペクトルの対数をとる
# fft_spec = np.fft.rfft(x)
amplitude_spectrum = np.fft.rfft(x)
cepstrum = get_cepstrum(np.abs(amplitude_spectrum))

fft_log_abs_spec = np.log(np.abs(amplitude_spectrum))

print("length of cepstrum:", len(cepstrum))

# Step3: Get cepstrum of the 13 lowest frequency
cepstrum[14:-14] = 0

# Step4: apply inverse-FFT
spectrum_envelope = np.fft.ifft(cepstrum)

# save figure
fig = plt.figure()
plt.xlabel('frequency [Hz]')
plt.ylabel('amplitude')
plt.plot(fft_log_abs_spec)
plt.plot(spectrum_envelope)
plt.show()
