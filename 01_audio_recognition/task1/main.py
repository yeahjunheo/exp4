import librosa
import numpy as np
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

size_frame = 4096
SR = 16000
size_shift = 16000 / 100

hamming_window = np.hamming(size_frame)

x, _ = librosa.load("../ex11/aiueo2.wav", sr=SR)
# x, _ = librosa.load("../ex14/easy_chords.wav", sr=SR)

duration = len(x) / SR

chord_names = {
    "0": "C MAJOR",
    "1": "C# MAJOR",
    "2": "D MAJOR",
    "3": "D# MAJOR",
    "4": "E MAJOR",
    "5": "F MAJOR",
    "6": "F# MAJOR",
    "7": "G MAJOR",
    "8": "G# MAJOR",
    "9": "A MAJOR",
    "10": "A# MAJOR",
    "11": "B MAJOR",
}


def zero_cross_short(waveform):
    d = np.array(waveform)
    return sum([1 if x < 0.0 else 0 for x in d[1:] * d[:-1]])


def is_peak(a, index):
    if index == 0 or index == len(a) - 1:
        return False
    return a[index - 1] < a[index] and a[index] > a[index + 1]


def cepstrum(amplitude_spectrum):
    log_spectrum = np.log(amplitude_spectrum)
    cepstrum = np.fft.fft(log_spectrum)
    return cepstrum


def hz2nn(frequency):
    return int(round(12.0 * (math.log(frequency / 440.0) / math.log(2.0)))) + 69


def chroma_vector(spectrum, frequencies):
    # 0 = C, 1 = C#, 2 = D, ..., 11 = B

    # 12次元のクロマベクトルを作成（ゼロベクトルで初期化）
    cv = np.zeros(12)

    # スペクトルの周波数ビン毎に
    # クロマベクトルの対応する要素に振幅スペクトルを足しこむ
    for s, f in zip(spectrum, frequencies):
        nn = hz2nn(f)
        cv[nn % 12] += abs(s)

    return cv


# ====================Main operations====================#
# time_shift dependent
spectrogram = []
f0_values = []
zcr_values = []
volume_db = []
chromagram = []
chords = []

for i in np.arange(0, len(x) - size_frame, size_shift):
    idx = int(i)
    x_frame = x[idx : idx + size_frame]

    # volume
    volume = 20 * np.log10(np.mean(x_frame**2))
    volume_db.append(volume)

    x_frame = x[idx : idx + size_frame] * hamming_window

    # spectrogram
    fft_spec = np.fft.rfft(x_frame)
    fft_log_abs_spec = np.log(np.abs(fft_spec))
    spectrogram.append(fft_log_abs_spec)

    # chromagram
    frequencies = np.linspace((SR / 2) / len(fft_spec), SR / 2, len(fft_spec))
    chroma = chroma_vector(fft_spec, frequencies)
    chromagram.append(chroma)

    # chord estimation
    chroma = np.append(chroma, chroma)
    a_root, a_3, a_5 = 1.0, 0.5, 0.8
    chord_estimate = np.zeros(24)
    for i in range(12):
        major = a_root * chroma[i] + a_3 * chroma[i + 4] + a_5 * chroma[i + 7]
        minor = a_root * chroma[i] + a_3 * chroma[i + 3] + a_5 * chroma[i + 7]
        chord_estimate[i] = major
        chord_estimate[i + 12] = minor
    chord = np.argmax(chord_estimate)
    chords.append(chord)

    # vowel identification
    zcr = zero_cross_short(x_frame)
    zcr_values.append(zcr)

    # fundamental frequency
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

# ============================================================= #
# ============================ GUI ============================ #
# ============================================================= #

root = tk.Tk()
root.title("EXP4-AUDIO-TASK-1")

tabControl = ttk.Notebook(root)

tab1 = ttk.Frame(tabControl)
tabControl.add(tab1, text="Spectrogram")

tab2 = ttk.Frame(tabControl)
tabControl.add(tab2, text="Spectrum")

tabControl.pack(expand=1, fill="both")

# ============================================================= #
# =========================== tab 1 =========================== #
# ============================================================= #

frame1 = ttk.Frame(tab1)
frame1.grid(row=0, column=0, padx=10, pady=10)
frame2 = ttk.Frame(tab1)
frame2.grid(row=0, column=1, rowspan=2, padx=10, pady=10)

# checkbox frame
checkbox_frame = ttk.Frame(frame2)
checkbox_frame.grid(row=0, column=0, sticky="w")

show_f0_var = tk.BooleanVar()
show_f0_checkbox = ttk.Checkbutton(
    checkbox_frame, text="Fund. Freq.", variable=show_f0_var
)
show_f0_checkbox.grid(row=0, column=0, sticky="w", padx=10, pady=5)

show_zcr_var = tk.BooleanVar()
show_zcr_checkbox = ttk.Checkbutton(
    checkbox_frame, text="Zero cross", variable=show_zcr_var
)
show_zcr_checkbox.grid(row=1, column=0, sticky="w", padx=10, pady=5)

show_volume_var = tk.BooleanVar()
show_volume_checkbox = ttk.Checkbutton(
    checkbox_frame, text="Volume", variable=show_volume_var
)
show_volume_checkbox.grid(row=2, column=0, sticky="w", padx=10, pady=5)

# modify frame
modify_frame = ttk.Frame(frame2)
modify_frame.grid(row=1, column=0)

x_axis_min = tk.DoubleVar()
x_axis_min.set(0)
x_axis_max = tk.DoubleVar()
x_axis_max.set(duration)

y_axis_min = tk.DoubleVar()
y_axis_min.set(0)
y_axis_max = tk.DoubleVar()
y_axis_max.set(8000)

x_axis_label = ttk.Label(modify_frame, text="X-axis range")
x_axis_label.grid(row=0, column=0, columnspan=2, pady=5)
x_axis_min_entry = ttk.Entry(modify_frame, textvariable=x_axis_min, width=10)
x_axis_min_entry.grid(row=1, column=0, padx=5, pady=5)
x_axis_max_entry = ttk.Entry(modify_frame, textvariable=x_axis_max, width=10)
x_axis_max_entry.grid(row=1, column=1, padx=5, pady=5)

y_axis_label = ttk.Label(modify_frame, text="Y-axis range")
y_axis_label.grid(row=2, column=0, columnspan=2, pady=5)
y_axis_min_entry = ttk.Entry(modify_frame, textvariable=y_axis_min, width=10)
y_axis_min_entry.grid(row=3, column=0, padx=5, pady=5)
y_axis_max_entry = ttk.Entry(modify_frame, textvariable=y_axis_max, width=10)
y_axis_max_entry.grid(row=3, column=1, padx=5, pady=5)

confirm_button = ttk.Button(modify_frame, text="Confirm")
confirm_button.grid(row=4, column=0, columnspan=2, pady=5)

fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=frame1)
toolbar = NavigationToolbar2Tk(canvas, frame1)
toolbar.update()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
ax.set_xlabel("sec")
ax.set_ylabel("frequency [Hz]")
ax.imshow(
    np.flipud(np.array(spectrogram).T),
    extent=[0, duration, 0, 8000],
    aspect="auto",
    interpolation="nearest",
)

axV = ax.twinx()
axV.set_ylabel("Volume [dB]")

canvas.get_tk_widget().pack(side="left")


def update_spectrogram():
    ax.clear()
    ax.set_xlabel("sec")
    ax.set_ylabel("frequency [Hz]")
    ax.imshow(
        np.flipud(np.array(spectrogram).T),
        extent=[0, duration, 0, 8000],
        aspect="auto",
        interpolation="nearest",
    )
    axV.clear()
    axV.set_ylabel("Volume [dB]")
    axV.yaxis.set_label_position("right")

    if show_f0_var.get():
        ax.plot(
            np.arange(len(f0_values)) * size_shift / SR,
            f0_values,
            color="r",
            label="F0",
        )

    if show_zcr_var.get():
        ax.plot(
            np.arange(len(zcr_values)) * size_shift / SR,
            zcr_values,
            color="g",
            label="ZCR",
        )

    if show_volume_var.get():
        axV.plot(
            np.array(range(len(volume_db))) * size_shift / SR,
            volume_db,
            color="b",
            label="Volume",
        )

    try:
        x_min = float(x_axis_min.get())
        x_max = float(x_axis_max.get())
        y_min = float(y_axis_min.get())
        y_max = float(y_axis_max.get())

        if x_min >= x_max or y_min >= y_max:
            raise ValueError

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    except ValueError:
        pass

    if show_f0_var.get() or show_zcr_var.get():
        ax.legend()

    canvas.draw()


show_f0_var.trace_add("write", lambda *args: update_spectrogram())
show_zcr_var.trace_add("write", lambda *args: update_spectrogram())
show_volume_var.trace_add("write", lambda *args: update_spectrogram())
confirm_button.config(command=update_spectrogram)

# ============================================================= #
# ============================ tab 2 ========================== #
# ============================================================= #


def _draw_spectrum(v):
    # スライドバーの値からスペクトルのインデクスおよびそのスペクトルを取得
    index = int((len(spectrogram) - 1) * (float(v) / duration))
    spectrum = spectrogram[index]

    curr_chord = chord_names[str(chords[index])]

    # 直前のスペクトル描画を削除し，新たなスペクトルを描画
    plt.cla()
    x_data = np.fft.rfftfreq(size_frame, d=1 / SR)
    ax2.plot(x_data, spectrum)
    ax2.set_ylim(-10, 5)
    ax2.set_xlim(0, SR / 2)
    ax2.set_ylabel("amblitude")
    ax2.set_xlabel("frequency [Hz]")
    ax2.text(
        0.95,
        0.95,
        curr_chord,
        transform=ax2.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    canvas2.draw()


# スペクトルを表示する領域を確保
# ax2, canvs2 を使って上記のコールバック関数でグラフを描画する
fig2, ax2 = plt.subplots()
canvas2 = FigureCanvasTkAgg(fig2, master=tab2)
canvas2.get_tk_widget().pack(
    side="top"
)  # "top"は上部方向にウィジェットを積むことを意味する

# スライドバーを作成
scale = tk.Scale(
    command=_draw_spectrum,  # ここにコールバック関数を指定
    master=tab2,  # 表示するフレーム
    from_=0,  # 最小値
    to=duration,  # 最大値
    resolution=size_shift / SR,  # 刻み幅
    label="スペクトルを表示する時間[sec]",
    orient=tk.HORIZONTAL,  # 横方向にスライド
    length=600,  # 横サイズ
    width=50,  # 縦サイズ
    font=("", 20),  # フォントサイズは20pxに設定
)
scale.pack(side="top")

root.mainloop()
