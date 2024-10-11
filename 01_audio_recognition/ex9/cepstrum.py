#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# ケプストラムを計算する関数
#

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import librosa

# スペクトルを受け取り，ケプストラムを返す関数
def cepstrum(amplitude_spectrum):
	log_spectrum = np.log(amplitude_spectrum)
	cepstrum = np.fft.fft(log_spectrum)
	return cepstrum

SR = 16000

x, _ = librosa.load("../ex1/aiueo_long.wav", sr=SR)
# x, _ = librosa.load("../ex1/aiueo_short.wacv", sr=SR)

amplitude_spectrum = np.fft.rfft(x)

fft_log_abs_spec = np.log(np.abs(amplitude_spectrum))

cepstrum = cepstrum(np.abs(amplitude_spectrum))

cepstrum[13:-13] = 0

spectrum_envelope = np.fft.ifft(cepstrum)

fig = plt.figure()
plt.xlabel('frequency [Hz]')
plt.ylabel('amplitude')
plt.plot(fft_log_abs_spec)
plt.plot(spectrum_envelope)
plt.show()

fig.savefig("cepstrum_long.png")
# fig.savefig("cepstrum_short.png")