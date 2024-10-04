# Exercise 03

## 高速フーリエ変換を実行する関数であるnp.fft.rfft について，その動作を説明せよ．特に，入力と出力の意味について詳しく説明すること．

Taking a look at the numpy [documentation](https://numpy.org/doc/2.0/reference/generated/numpy.fft.rfft.html), we can see that this is an extension of the Fast Fourier Transform (FTT). The main function of this function is to compute the one-dimensional n-point discrete fourier transform of a real-valued array.

```python
numpy.fft.rfft(a, n=None, axis=-1, norm=None, out=None)
```

In terms of input and output, this function taks in an array from the audio wav file we recorded. Then it returns a complex ndarray, with the appropriate transfomrations made after computing the one-dimensional DFT using FFT. 