#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# ケプストラムを計算する関数
#

import numpy as np
import matplotlib.pyplot as plt
import librosa

# スペクトルを受け取り，ケプストラムを返す関数
def cepstrum(amplitude_spectrum):
	log_spectrum = np.log(amplitude_spectrum)
	cepstrum = np.fft.fft(log_spectrum)
	return cepstrum

SR = 16000

# audio for long_a
audio_files = [
	("long_a", "../ex1/aiueo_long.wav", 0.7, 0.7),
	("long_i", "../ex1/aiueo_long.wav", 1.8, 0.7),
	("long_u", "../ex1/aiueo_long.wav", 2.8, 0.7),
	("long_e", "../ex1/aiueo_long.wav", 3.7, 0.7),
	("long_o", "../ex1/aiueo_long.wav", 4.6, 0.7),
	("short_a", "../ex1/aiueo_short.wav", 0.8, 0.3),
	("short_i", "../ex1/aiueo_short.wav", 1.8, 0.3),
	("short_u", "../ex1/aiueo_short.wav", 2.7, 0.3),
	("short_e", "../ex1/aiueo_short.wav", 3.8, 0.3),
	("short_o", "../ex1/aiueo_short.wav", 4.8, 0.3),
]

for name, file, offset, duration in audio_files:
	x, _ = librosa.load(file, sr=SR, offset=offset, duration=duration)
	amplitude_spectrum = np.fft.rfft(x)
	fft_log_abs_spec = np.log(np.abs(amplitude_spectrum))
	cepstrum_data = cepstrum(np.abs(amplitude_spectrum))
	cepstrum_data[13:-13] = 0
	spectrum_envelope = np.fft.ifft(cepstrum_data)
	
	fig = plt.figure()
	plt.xlabel('frequency [Hz]')
	plt.ylabel('amplitude')
	plt.plot(fft_log_abs_spec)
	plt.plot(spectrum_envelope)
	plt.show()
	
	fig.savefig(f"cepstrum_{name}.png")