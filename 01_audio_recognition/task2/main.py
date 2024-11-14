import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import math
import tkinter as tk
import scipy.io.wavfile

from tkinter import ttk
from tkinter import filedialog
from playsound import playsound
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.animation import FuncAnimation

# Tkinter GUI setup
root = tk.Tk()
root.title("Audio Recognition")
root.geometry("1500x1000")

# Global parameters
x = None
x_path = None
y = None
z = None
SR = 16000
spectrogram = []
voice_change_spectrum = []
f0_values = []
zcr_values = []
volume_db = []
volume_db_tremolo = []
chords = []
voice_change_chords = []
duration = 0

# Constants
size_frame = 2048
f_s = SR
SIZE_SHIFT = 16000 / 100
hamming_window = np.hamming(size_frame)
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


# Menu bar
menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff=0)


def load_audio():
    global x, duration, x_path
    x_path = ""
    file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav *.mp3")])
    if not file_path:
        return

    try:
        x, _ = librosa.load(file_path, sr=SR)
        x_path = os.path.relpath(file_path, start=os.curdir)
        duration = len(x) / SR
        compute_features()
    except Exception as e:
        tk.messagebox.showerror("Error", f"Error loading audio file: {e}")


def exit_app():
    # Delete existin audio
    if os.path.exists("audio_voice_change.wav"):
        os.remove("audio_voice_change.wav")
    if os.path.exists("audio_tremolo.wav"):
        os.remove("audio_tremolo.wav")

    root.quit()


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
    cv = np.zeros(12)

    for s, f in zip(spectrum, frequencies):
        nn = hz2nn(f)
        cv[nn % 12] += abs(s)

    return cv


def generate_sinusoid(sampling_rate, frequency, duration):
    sampling_interval = 1.0 / sampling_rate
    t = np.arange(sampling_rate * duration) * sampling_interval
    waveform = np.sin(2.0 * math.pi * frequency * t)
    return waveform


def play_audio(filename):
    def _play_audio():
        if filename:
            abs_path = os.path.abspath(filename)
            os.system(f"afplay {abs_path}")
        else:
            tk.messagebox.showerror("Error", "No audio file selected")

    return _play_audio


def voice_change():
    global y

    if x is None:
        return

    frequency = voice_change_frequency.get()
    print(frequency)
    sin_wave = generate_sinusoid(SR, frequency, duration)
    sin_wave = sin_wave * 0.9
    x_changed = x * sin_wave
    x_changed = (x_changed * 32768.0).astype("int16")
    filename = "audio_voice_change.wav"
    scipy.io.wavfile.write(filename, int(SR), x_changed)
    y, _ = librosa.load(filename, sr=SR)
    voice_change_computation()
    update_voice_change_button()


def voice_change_computation():
    global voice_change_spectrum, voice_change_chords
    voice_change_spectrum = []
    voice_change_chords = []

    for i in np.arange(0, len(y) - size_frame, SIZE_SHIFT):
        idx = int(i)
        y_frame = y[idx : idx + size_frame] * hamming_window

        # Compute the spectrogram and chromagram for this frame
        fft_spec = np.fft.rfft(y_frame)
        fft_log_abs_spec = np.log(np.abs(fft_spec))
        voice_change_spectrum.append(fft_log_abs_spec)

        frequencies = np.linspace((SR / 2) / len(fft_spec), SR / 2, len(fft_spec))
        chroma = chroma_vector(fft_spec, frequencies)

        # Chord estimation
        chroma = np.append(chroma, chroma)
        a_root, a_3, a_5 = 1.0, 0.5, 0.8
        chord_estimate = np.zeros(24)
        for i in range(12):
            major = a_root * chroma[i] + a_3 * chroma[i + 4] + a_5 * chroma[i + 7]
            minor = a_root * chroma[i] + a_3 * chroma[i + 3] + a_5 * chroma[i + 7]
            chord_estimate[i] = major
            chord_estimate[i + 12] = minor
        chord = np.argmax(chord_estimate) % 12
        voice_change_chords.append(chord)

    # Enable the slider after data is ready
    scale2.config(state="normal")
    _draw_spectrum_2(0)  # Initialize the plot with the first spectrum


def tremolo():
    global z

    if x is None:
        return

    D = tremolo_D.get()
    R = tremolo_R.get()
    sin_wave = 1 + D * generate_sinusoid(SR, R / f_s, duration)
    x_changed = x * sin_wave
    x_changed = (x_changed * 32768.0).astype("int16")

    filename = "audio_tremolo.wav"
    scipy.io.wavfile.write(filename, int(SR), x_changed)

    z, _ = librosa.load(filename, sr=SR)
    tremolo_computation()
    update_tremolo_button()


def tremolo_computation():
    global volume_db_tremolo
    volume_db_tremolo = []

    for j in np.arange(0, len(z) - size_frame, SIZE_SHIFT):
        idx = int(j)
        z_frame = z[idx : idx + size_frame]
        volume = 20 * np.log10(np.mean(z_frame**2))
        volume_db_tremolo.append(volume)

    update_volume_db()


def compute_features():
    x_axis_max.set(duration)
    scale.config(to=duration)
    scale2.config(to=duration)

    global spectrogram, f0_values, zcr_values, volume_db, chords, voice_change_chords
    spectrogram = []
    f0_values = []
    zcr_values = []
    volume_db = []
    chords = []
    voice_change_chords = []

    for i in np.arange(0, len(x) - size_frame, SIZE_SHIFT):
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

        # chord estimation
        chroma = np.append(chroma, chroma)
        a_root, a_3, a_5 = 1.0, 0.5, 0.8
        chord_estimate = np.zeros(24)
        for i in range(12):
            major = a_root * chroma[i] + a_3 * chroma[i + 4] + a_5 * chroma[i + 7]
            minor = a_root * chroma[i] + a_3 * chroma[i + 3] + a_5 * chroma[i + 7]
            chord_estimate[i] = major
            chord_estimate[i + 12] = minor
        chord = np.argmax(chord_estimate) % 12
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
    update_spectrogram()
    _draw_spectrum(0)


filemenu.add_command(label="Load audio...", command=load_audio)  # load audio file
filemenu.add_separator()
filemenu.add_command(label="Exit", command=exit_app)
menubar.add_cascade(label="File", menu=filemenu)
root.config(menu=menubar)

# Notebook
notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

# Tabs
tab1 = tk.Frame(notebook)
notebook.add(tab1, text="Spectrogram")
tab2 = tk.Frame(notebook)
notebook.add(tab2, text="Spectrum")
tab3 = tk.Frame(notebook)
notebook.add(tab3, text="Voice Change")
tab4 = tk.Frame(notebook)
notebook.add(tab4, text="Tremolo")

# Tab 1 contents
tab1_left_frame = tk.Frame(tab1)
tab1_left_frame.pack(side="left")
tab1_right_frame = tk.Frame(tab1)
tab1_right_frame.pack(side="right")

checkbox_frame = tk.Frame(tab1_right_frame, pady=25)
checkbox_frame.pack(side="top")

show_f0_var = tk.BooleanVar()
show_f0_checkbox = tk.Checkbutton(
    checkbox_frame, text="Fund. Freq.", variable=show_f0_var
)
show_f0_checkbox.pack(side="top", anchor="w")

show_zcr_var = tk.BooleanVar()
show_zcr_checkbox = tk.Checkbutton(
    checkbox_frame, text="Zero cross", variable=show_zcr_var
)
show_zcr_checkbox.pack(side="top", anchor="w")

show_volume_var = tk.BooleanVar()
show_volume_checkbox = tk.Checkbutton(
    checkbox_frame, text="Volume", variable=show_volume_var
)
show_volume_checkbox.pack(side="top", anchor="w")

play_x_audio = tk.Button(checkbox_frame, text="Play audio")
play_x_audio.config(command=lambda: playsound(x_path))
play_x_audio.pack(side="top", anchor="w")

modify_frame = tk.Frame(tab1_right_frame, pady=25)
modify_frame.pack(side="bottom", anchor="w")

x_axis_min, x_axis_max = tk.DoubleVar(value=0), tk.DoubleVar(value=duration)
y_axis_min, y_axis_max = tk.DoubleVar(value=0), tk.DoubleVar(value=8000)

for i, (label, var_min, var_max) in enumerate(
    [("X-axis range", x_axis_min, x_axis_max), ("Y-axis range", y_axis_min, y_axis_max)]
):
    tk.Label(modify_frame, text=label).grid(row=2 * i, column=0, columnspan=2, pady=5)
    tk.Entry(modify_frame, textvariable=var_min, width=10).grid(
        row=2 * i + 1, column=0, padx=5, pady=5
    )
    tk.Entry(modify_frame, textvariable=var_max, width=10).grid(
        row=2 * i + 1, column=1, padx=5, pady=5
    )

confirm_button = tk.Button(modify_frame, text="Confirm")
confirm_button.grid(row=4, column=0, columnspan=2, pady=5)

tremolo_frame = tk.Frame(tab1_right_frame, pady=25)
tremolo_frame.pack(side="bottom", fill="both")

tremolo_D = tk.DoubleVar(value=1.0)
tremolo_R = tk.DoubleVar(value=10000.0)
tremolo_D_label = tk.Label(tremolo_frame, text="Tremolo Depth")
tremolo_D_label.pack(side="top", anchor="center")
tremolo_D_entry = tk.Entry(tremolo_frame, textvariable=tremolo_D)
tremolo_D_entry.pack(side="top", anchor="center")
tremolo_R_label = tk.Label(tremolo_frame, text="Tremolo Rate")
tremolo_R_label.pack(side="top", anchor="center")
tremolo_R_entry = tk.Entry(tremolo_frame, textvariable=tremolo_R)
tremolo_R_entry.pack(side="top", anchor="center")
tremolo_confirm_button = tk.Button(tremolo_frame, text="Confirm", command=tremolo)
tremolo_confirm_button.pack(side="top", anchor="center")

tremolo_audio = tk.Button(
    tremolo_frame,
    text="Play tremolo",
    command=play_audio("audio_tremolo.wav"),
    state="disabled",
)
tremolo_audio.pack(side="top", anchor="center")


def update_tremolo_button():
    if z is not None:
        tremolo_audio.config(state="normal")
    else:
        tremolo_audio.config(state="disabled")


voice_change_frame = tk.Frame(tab1_right_frame, pady=25)
voice_change_frame.pack(side="bottom", fill="both")

voice_change_frequency = tk.DoubleVar(value=6400)
voice_change_frequency_label = tk.Label(voice_change_frame, text="Frequency")
voice_change_frequency_label.pack(side="top", anchor="center")
voice_change_frequency_entry = tk.Entry(
    voice_change_frame, textvariable=voice_change_frequency
)
voice_change_frequency_entry.pack(side="top", anchor="center")
voice_change_confirm_button = tk.Button(
    voice_change_frame, text="Confirm", command=voice_change
)
voice_change_confirm_button.pack(side="top", anchor="center")

voice_change_audio = tk.Button(
    voice_change_frame,
    text="Play voice change",
    command=play_audio("audio_voice_change.wav"),
    state="disabled",  # Initially disabled
)
voice_change_audio.pack(side="top", anchor="center")


def update_voice_change_button():
    if y is not None:
        voice_change_audio.config(state="normal")
    else:
        voice_change_audio.config(state="disabled")


fig, ax = plt.figure(figsize=(8, 6), dpi=75), plt.gca()
canvas = FigureCanvasTkAgg(fig, master=tab1_left_frame)
canvas.get_tk_widget().pack(side="top")
ax.set_xlabel("sec", fontsize=10)
ax.set_ylabel("frequency [Hz]", fontsize=10)

if x is not None:
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
    if x is not None:
        ax.imshow(
            np.flipud(np.array(spectrogram).T),
            extent=[0, duration, 0, 8000],
            aspect="auto",
            interpolation="nearest",
        )
    axV.clear()
    axV.set_ylabel("Volume [dB]")
    axV.yaxis.set_label_position("right")

    if show_f0_var.get() and x is not None:
        ax.plot(
            np.arange(len(f0_values)) * SIZE_SHIFT / SR,
            f0_values,
            color="r",
            label="F0",
        )

    if show_zcr_var.get() and x is not None:
        ax.plot(
            np.arange(len(zcr_values)) * SIZE_SHIFT / SR,
            zcr_values,
            color="g",
            label="ZCR",
        )

    if show_volume_var.get() and x is not None:
        axV.plot(
            np.array(range(len(volume_db))) * SIZE_SHIFT / SR,
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

    if (show_f0_var.get() or show_zcr_var.get()) and x is not None:
        ax.legend()

    canvas.draw()


show_f0_var.trace_add("write", lambda *args: update_spectrogram())
show_zcr_var.trace_add("write", lambda *args: update_spectrogram())
show_volume_var.trace_add("write", lambda *args: update_spectrogram())
confirm_button.config(command=update_spectrogram)


# Tab 2 contents
def _draw_spectrum(v):
    if x is None:
        return

    index = int((len(spectrogram) - 1) * (float(v) / duration))
    spectrum = spectrogram[index]

    curr_chord = chord_names[str(chords[index])]

    # 直前のスペクトル描画を削除し，新たなスペクトルを描画
    ax2.cla()
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


fig2, ax2 = plt.figure(figsize=(8, 6), dpi=60), plt.gca()
canvas2 = FigureCanvasTkAgg(fig2, master=tab2)
canvas2.get_tk_widget().pack(side="top")
scale = tk.Scale(
    command=_draw_spectrum,  # ここにコールバック関数を指定
    master=tab2,  # 表示するフレーム
    from_=0,  # 最小値
    to=duration,  # 最大値 (initially set to 0)
    resolution=SIZE_SHIFT / SR,  # 刻み幅
    label="スペクトルを表示する時間",
    orient=tk.HORIZONTAL,  # 横方向にスライド
    length=600,  # 横サイズ
    width=50,  # 縦サイズ
    font=("", 20),  # フォントサイズは20pxに設定
)
scale.pack(side="top")


def _draw_spectrum_2(v):
    if y is None or len(voice_change_spectrum) == 0:
        return

    # Calculate the index for the spectrum based on the slider value
    index = int((len(voice_change_spectrum) - 1) * (float(v) / duration))
    spectrum = voice_change_spectrum[index]

    # Get the current chord name
    curr_chord = chord_names.get(str(voice_change_chords[index]))

    # Clear and redraw the plot with updated data
    ax3.clear()  # Clear ax3 before drawing new data
    y_data = np.fft.rfftfreq(size_frame, d=1 / SR)
    ax3.plot(y_data, spectrum)
    ax3.set_ylim(-10, 5)
    ax3.set_xlim(0, SR / 2)
    ax3.set_ylabel("Amplitude")
    ax3.set_xlabel("Frequency [Hz]")
    ax3.text(
        0.95,
        0.95,
        curr_chord,
        transform=ax3.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    canvas3.draw()


fig3, ax3 = plt.figure(figsize=(8, 6), dpi=60), plt.gca()
canvas3 = FigureCanvasTkAgg(fig3, master=tab3)
canvas3.get_tk_widget().pack(side="top")

scale2 = tk.Scale(
    command=_draw_spectrum_2,
    master=tab3,
    from_=0,
    to=duration,
    resolution=SIZE_SHIFT / SR,
    label="Display Spectrum at Time [sec]",
    orient=tk.HORIZONTAL,
    length=600,
    width=50,
    font=("", 20),
    state="disabled",  # Start with disabled state
)
scale2.pack(side="top")


def update_volume_db():
    ax4.clear()
    ax4.set_xlabel("sec")
    ax4.set_ylabel("volume [dB]")
    if z is not None:
        ax4.plot(
            np.array(range(len(volume_db_tremolo))) * SIZE_SHIFT / SR,
            volume_db_tremolo,
            color="b",
            label="Tremolo Volume",
        )
    if x is not None:
        ax4.plot(
            np.array(range(len(volume_db))) * SIZE_SHIFT / SR,
            volume_db,
            color="r",
            label="Original Volume",
        )
    ax4.legend()
    canvas4.draw()


fig4, ax4 = plt.figure(figsize=(8, 6), dpi=75), plt.gca()
canvas4 = FigureCanvasTkAgg(fig4, master=tab4)
canvas4.get_tk_widget().pack(
    side="left"
)  # "top"は上部方向にウィジェットを積むことを意味する
ax4.set_xlabel("sec")
ax4.set_ylabel("volume [dB]")


def play_animation():
    fig4.clear()
    ax4.set_xlabel("sec")
    ax4.set_ylabel("volume [dB]")
    t = np.arange(0, len(volume_db)) * SIZE_SHIFT / SR
    t_tremolo = np.arange(0, len(volume_db_tremolo)) * SIZE_SHIFT / SR

    (line1,) = ax4.plot(t, volume_db, label="Original Volume", color="r")
    (line2,) = ax4.plot(t_tremolo, volume_db_tremolo, label="Tremolo Volume", color="b")
    ax4.set(
        xlim=[0, duration],
        ylim=[
            min(min(volume_db), min(volume_db_tremolo)),
            max(max(volume_db), max(volume_db_tremolo)),
        ],
        xlabel="Time [s]",
        ylabel="Volume [dB]",
    )
    ax.legend()

    def update(frame):
        line1.set_data(t[:frame], volume_db[:frame])
        line2.set_data(t_tremolo[:frame], volume_db_tremolo[:frame])
        return line1, line2

    ani = FuncAnimation(fig4, update, frames=len(t), interval=(duration * 1000) / len(t), blit=True)
    canvas4.draw()


# play animation
play_animation_button = tk.Button(tab4, text="Play Audio")
play_animation_button.config(command=play_animation)
play_animation_button.pack(side="top")

# execute the GUI
root.mainloop()
