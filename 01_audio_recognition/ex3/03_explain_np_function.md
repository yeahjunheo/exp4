# Exercise 03

## 高速フーリエ変換を実行する関数である np.fft.rfft について，その動作を説明せよ．特に，入力と出力の意味について詳しく説明すること．

Taking a look at the numpy [documentation](https://numpy.org/doc/2.0/reference/generated/numpy.fft.rfft.html), we can see that this is an extension of the Fast Fourier Transform (FTT). The main function of this function is to compute the one-dimensional n-point discrete fourier transform of a real-valued array.

```python
numpy.fft.rfft(a, n=None, axis=-1, norm=None, out=None)
```

### Input:

The input to `numpy.fft.rfft()` is a one-dimensional numpy array containing real-valued data. This array represents the time-domain signal from which you want to compute the Fourier transform. In our case, we will make use of `liburosa` and the audio we recorded to directly input it into the numpy function.

### Output:

The output of `numpy.fft.rfft()` is a complex-valued numpy array. Specifically, for an input array of length `n`, the output will have `n//2 + 1` complex elements. These elements represent the amplitude and phase information of the signal in the frequency domain. The first element corresponds to the zero-frequency component (DC component), and subsequent elements represent positive frequencies up to the Nyquist frequency (half of the sampling rate).

### Functionality:

The primary function of `numpy.fft.rfft()` is to compute the one-dimensional FFT efficiently for real-valued input data. The FFT transforms the input signal from the time domain to the frequency domain, providing insights into the frequency components present in the signal. This transformation is essential in various applications such as signal processing, spectral analysis, filtering, and in our case, audio recognition. The output complex array can be further processed for tasks such as spectral analysis (identifying dominant frequencies), denoising (filtering out noise in specific frequency bands), which is what we are aiming to do with our current experiment. 