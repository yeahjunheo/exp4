# 2 種類の「あいうえお」の一方を用いてモデルを学習し，他方を用いて母音を認識せよ．
# 認識された音素の推移をスペクトログラムに並べて（もしくは重ねて）示せ．
# 例えば「あ」=0, 「い」=1, 「う」=2,「え」=3, 「お」=4 とした時系列のグラフにするなど．

import numpy as np
import matplotlib.pyplot as plt
import librosa

SR = 16000
size_frame = 2**11
size_shift = 16000 / 100
hamming_window = np.hamming(size_frame)

# training files
training_audio = [
	("short_a", "aiueo_01_train.wav", 1.0, 1.0),
	("short_i", "aiueo_01_train.wav", 2.7, 1.0),
	("short_u", "aiueo_01_train.wav", 4.8, 1.0),
	("short_e", "aiueo_01_train.wav", 7.0, 1.0),
	("short_o", "aiueo_01_train.wav", 10.0, 1.0),
]

def cepstrum(x_frame, D=13):
    amplitude_spectrum = np.abs(np.fft.rfft(x_frame))
    log_spectrum = np.log(amplitude_spectrum)
    cepstrum = np.fft.rfft(log_spectrum).real
    return cepstrum[:D]

mean_ceps = []
var_ceps = []
for audio in training_audio:
    audio_data, _ = librosa.load(audio[1], sr=SR, offset=audio[2], duration=audio[3])
    
    cepstrum_coefficients = []
    
    for i in np.arange(0, len(audio_data), size_shift):
        idx = int(i)
        x_frame = audio_data[idx: idx + size_frame]
        if len(x_frame) == size_frame:
            xframe = x_frame * hamming_window
            cepstrum_coefficients.append(cepstrum(x_frame))
        
    mean_ceps.append(np.mean(cepstrum_coefficients, axis=0))
    var_ceps.append(np.var(cepstrum_coefficients, axis=0))

# testing files
test_audio, _ = librosa.load("aiueo_02_test.wav", sr=SR)

def gaussian_likelihood(cepstrum, mean, variance):
    return -np.sum(np.log(np.sqrt(variance)) + ((cepstrum - mean) ** 2) /(2*variance) )

test_ceps = []
for i in np.arange(0, len(test_audio) - size_frame, size_shift):
    idx = int(i)
    x_frame = test_audio[idx: idx + size_frame]
    if len(x_frame) == size_frame:
        xframe = x_frame * hamming_window
        test_ceps.append(cepstrum(x_frame))
        
print(test_ceps)

result_list = []
for frame in test_ceps:
    likelihoods = []
    for i in range(len(training_audio)):
        likelihoods.append(gaussian_likelihood(frame, mean_ceps[i], var_ceps[i]))
        
    max_log_likelihood = -1000000000000
    result = -1
    for i in range(len(likelihoods)):
        if likelihoods[i] > max_log_likelihood:
            max_log_likelihood = likelihoods[i]
            result = i
    result_list.append(result)

plt.figure()
plt.plot(result_list)
plt.show()