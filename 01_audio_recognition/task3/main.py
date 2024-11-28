#
# 計算機科学実験及演習 4「音響信号処理」
#
# 簡易カラオケシステム
#
# mp3ファイルを別スレッドで再生しつつ
# マイクからの音声入力に対してスペクトログラムとパワーを計算して表示する
# 上記をリアルタイムで逐次処理を行う
#

# ライブラリの読み込み
import pyaudio
import numpy as np
import threading
import time
import math
import librosa

# matplotlib関連
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# GUI関連
import tkinter
from tkinter import ttk as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# mp3ファイルを読み込んで再生
from pydub import AudioSegment
from pydub.utils import make_chunks
import os

# サンプリングレート
SAMPLING_RATE = 16000

# フレームサイズ
FRAME_SIZE = 2048

# サイズシフト
SHIFT_SIZE = int(SAMPLING_RATE / 20)  # 今回は0.05秒

# スペクトルをカラー表示する際に色の範囲を正規化するために
# スペクトルの最小値と最大値を指定
# スペクトルの値がこの範囲を超えると，同じ色になってしまう
SPECTRUM_MIN = -5
SPECTRUM_MAX = 1

# 音量を表示する際の値の範囲
VOLUME_MIN = -120
VOLUME_MAX = -10

FFREQ_MIN = 0
FFREQ_MAX = 300

# log10を計算する際に，引数が0にならないようにするためにこの値を足す
EPSILON = 1e-10

# ハミング窓
hamming_window = np.hamming(FRAME_SIZE)

# グラフに表示する縦軸方向のデータ数
MAX_NUM_SPECTROGRAM = int(FRAME_SIZE / 2)

# グラフに表示する横軸方向のデータ数
NUM_DATA_SHOWN = 100

# remote parameters
music_pitch = 0.0
music_tempo = 1.0

# GUIの開始フラグ（まだGUIを開始していないので、ここではFalseに）
is_gui_running = False
is_select_song = False

# song file name
# filename = ""

#
# (1) GUI / グラフ描画の処理
#

# ここでは matplotlib animation を用いて描画する
# 毎回 figure や ax を初期化すると処理時間がかかるため
# データを更新したら，それに従って必要な部分のみ再描画することでリアルタイム処理を実現する


# matplotlib animation によって呼び出される関数
# ここでは最新のスペクトログラムと音量のデータを格納する
# 再描画はmatplotlib animationが行う
def animate(frame_index):
    ax1_sub.set_array(spectrogram_data)
    ax2_sub.set_data(time_x_data, ffreq_data)

    ax3_sub.set_array(spectrogram_data_music)
    ax4_sub.set_data(time_x_data, melody_data)

    return ax1_sub, ax2_sub, ax3_sub, ax4_sub


# GUIで表示するための処理（Tkinter）
root = tkinter.Tk()
root.wm_title("EXP4-AUDIO-SAMPLE")


# 終了ボタンが押されたときに呼び出される関数
# ここではGUIを終了する
def _quit():
    root.quit()
    root.destroy()


display_frame = tkinter.Frame(root)
display_frame.pack(side="left", fill="both", expand=True)

remote_frame = tkinter.Frame(root)
remote_frame.pack(side="right", fill="x", expand=True)

# スペクトログラムを描画
fig, ax = plt.subplots(2)
fig.set_size_inches(8, 4)
canvas = FigureCanvasTkAgg(fig, master=display_frame)

# 横軸の値のデータ
time_x_data = np.linspace(
    0, NUM_DATA_SHOWN * (SHIFT_SIZE / SAMPLING_RATE), NUM_DATA_SHOWN
)
# 縦軸の値のデータ
freq_y_data = np.linspace(8000 / MAX_NUM_SPECTROGRAM, 8000, MAX_NUM_SPECTROGRAM)

# とりあえず初期値（ゼロ）のスペクトログラムと音量のデータを作成
# この numpy array にデータが更新されていく
spectrogram_data = np.zeros((len(freq_y_data), len(time_x_data)))
ffreq_data = np.zeros(len(time_x_data))

spectrogram_data_music = np.zeros((len(freq_y_data), len(time_x_data)))
melody_data = np.zeros(len(time_x_data))

# スペクトログラムを描画する際に横軸と縦軸のデータを行列にしておく必要がある
# これは下記の matplotlib の pcolormesh の仕様のため
X = np.zeros(spectrogram_data.shape)
Y = np.zeros(spectrogram_data.shape)
for idx_f, f_v in enumerate(freq_y_data):
    for idx_t, t_v in enumerate(time_x_data):
        X[idx_f, idx_t] = t_v
        Y[idx_f, idx_t] = f_v

# pcolormeshを用いてスペクトログラムを描画
# 戻り値はデータの更新 & 再描画のために必要
ax1_sub = ax[0].pcolormesh(
    X,
    Y,
    spectrogram_data,
    shading="nearest",  # 描画スタイル
    cmap="jet",  # カラーマップ
    norm=Normalize(
        SPECTRUM_MIN, SPECTRUM_MAX
    ),  # 値の最小値と最大値を指定して，それに色を合わせる
)

# 音量を表示するために反転した軸を作成
ax2 = ax[0].twinx()

# 音量をプロットする
# 戻り値はデータの更新 & 再描画のために必要
(ax2_sub,) = ax2.plot(time_x_data, ffreq_data, c="purple")

# ラベルの設定
ax[0].set_xlabel("sec")  # x軸のラベルを設定
ax[0].set_ylabel("frequency [Hz]")  # y軸のラベルを設定
ax2.set_ylabel("fundamental frequency [Hz]")  # 反対側のy軸のラベルを設定

# 音量を表示する際の値の範囲を設定
ax2.set_ylim([FFREQ_MIN, FFREQ_MAX])

ax3_sub = ax[1].pcolormesh(
    X,
    Y,
    spectrogram_data_music,
    shading="nearest",
    cmap="jet",
    norm=Normalize(SPECTRUM_MIN, SPECTRUM_MAX),
)

ax4 = ax[1].twinx()
(ax4_sub,) = ax4.plot(time_x_data, melody_data, c="y", linewidth=5)

ax[1].set_xlabel("sec")
ax[1].set_ylabel("frequency [Hz]")
ax[1].set_ylim([0, 1000])
ax4.set_ylabel("Melody")
ax4.set_ylim([0, 500])

# maplotlib animationを設定
ani = animation.FuncAnimation(
    fig,
    animate,  # 再描画のために呼び出される関数
    interval=100,  # 100ミリ秒間隔で再描画を行う（PC環境によって処理が追いつかない場合はこの値を大きくするとよい）
    blit=False,  # blitting処理を行うため描画処理が速くなる（環境によってうまく動作しない場合はこれをFalseにしてみること）
)

# matplotlib を GUI(Tkinter) に追加する
toolbar = NavigationToolbar2Tk(canvas, display_frame)
canvas.get_tk_widget().pack()

# TODO: Convert this bit to progress bar instead of just string vaiables
# Display progress bar for the song
text = tkinter.StringVar()
text.set("0.0")
style = ttk.Style()
style.theme_use("default")
style.configure("TProgressbar", thickness=15, troughcolor="white", background="blue")

song_progress = ttk.Progressbar(
    master=display_frame,
    orient="horizontal",
    length=fig.get_size_inches()[0] * fig.dpi,
    mode="determinate",
    style="TProgressbar",
)

song_progress.pack()

# display lyrics for the song
# in two parts: top and bottom
lyrics_frame = tkinter.Frame(master=display_frame)
lyrics_frame.pack(side="bottom", fill="both", expand=True)

lyrics_top = tkinter.Label(lyrics_frame, text="", font=("", 30))
lyrics_top.pack(side="top")

lyrics_bottom = tkinter.Label(lyrics_frame, text="", font=("", 20), fg="gray")
lyrics_bottom.pack(side="bottom")

# set tempo of the song
remote_tempo_frame = tkinter.Frame(
    master=remote_frame, highlightbackground="gray", highlightthickness=5
)
remote_tempo_frame.pack(side="left", fill="both", padx=10, pady=10)

tempo_label = tkinter.Label(remote_tempo_frame, text="Tempo", font=("", 30), width=5)
tempo_label.pack(side="top")

tempo_up_btn = tkinter.Button(remote_tempo_frame, text="+1", font=("", 30), width=3)
tempo_up_btn.pack(side="top")

tempo_value = tkinter.StringVar()
tempo_value.set(str(music_tempo))
tempo_label = tkinter.Label(remote_tempo_frame, textvariable=tempo_value, font=("", 30))
tempo_label.pack(side="top")

tempo_down_btn = tkinter.Button(remote_tempo_frame, text="-1", font=("", 30), width=3)
tempo_down_btn.pack(side="top")


def increase_tempo():
    global music_tempo
    music_tempo = round(min(music_tempo + 0.1, 2), 1)
    # print(music_tempo)
    tempo_value.set(str(music_tempo))


def decrease_tempo():
    global music_tempo
    music_tempo = round(max(music_tempo - 0.1, 0.5), 1)
    tempo_value.set(str(music_tempo))


tempo_up_btn.config(command=increase_tempo)
tempo_down_btn.config(command=decrease_tempo)


# restart playing the song
def restart_song():
    global is_gui_running, now_playing_sec, t_play_music

    # Stop the current playback
    is_gui_running = False
    t_play_music.join()

    # Reset the playback position
    now_playing_sec = 0.0

    # Restart the playback thread
    is_gui_running = True
    t_play_music = threading.Thread(target=play_music)
    t_play_music.setDaemon(True)
    t_play_music.start()


# restart the song
restart_btn = tkinter.Button(
    remote_frame, text="Restart", font=("", 30), command=restart_song
)
restart_btn.pack()

# 終了ボタンを作成
button = tkinter.Button(master=remote_frame, text="終了", command=_quit, font=("", 30))
button.pack()

#
# play music and record voice
#

#
# (2) マイク入力のための処理
#


# 配列 a の index 番目の要素がピーク（両隣よりも大きい）であれば True を返す
def is_peak(a, index):
    # （自分で実装すること，passは消す）
    if index == 0 or index == len(a) - 1:
        return False
    return a[index - 1] < a[index] and a[index] > a[index + 1]


x_stacked_data = np.array([])


# フレーム毎に呼び出される関数
def input_callback(in_data, frame_count, time_info, status_flags):
    # この関数は別スレッドで実行するため
    # メインスレッドで定義した以下の２つの numpy array を利用できるように global 宣言する
    # これらにはフレーム毎のスペクトルと音量のデータが格納される
    global x_stacked_data, spectrogram_data, ffreq_data

    # 現在のフレームの音声データをnumpy arrayに変換
    x_current_frame = np.frombuffer(in_data, dtype=np.float32)

    # 現在のフレームとこれまでに入力されたフレームを連結
    x_stacked_data = np.concatenate([x_stacked_data, x_current_frame])

    # フレームサイズ分のデータがあれば処理を行う
    if len(x_stacked_data) >= FRAME_SIZE:
        # フレームサイズからはみ出した過去のデータは捨てる
        x_stacked_data = x_stacked_data[len(x_stacked_data) - FRAME_SIZE :]

        # スペクトルを計算
        fft_spec = np.fft.rfft(x_stacked_data * hamming_window)
        fft_log_abs_spec = np.log10(np.abs(fft_spec) + EPSILON)[:-1]

        # ２次元配列上で列方向（時間軸方向）に１つずらし（戻し）
        # 最後の列（＝最後の時刻のスペクトルがあった位置）に最新のスペクトルデータを挿入
        spectrogram_data = np.roll(spectrogram_data, -1, axis=1)
        spectrogram_data[:, -1] = fft_log_abs_spec

        # calcualte volume
        volume = 20 * np.log10(np.mean(x_current_frame**2) + EPSILON)

        # fundamental frequency
        autocorr = np.correlate(x_current_frame, x_current_frame, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        peakindices = [i for i in range(len(autocorr) - 1) if is_peak(autocorr, i)]
        peakindices = [i for i in peakindices if i != 0]
        if len(peakindices) > 0:
            max_peak_index = max(peakindices, key=lambda index: autocorr[index])
            tau = max_peak_index / SAMPLING_RATE
            ffreq = 1 / tau
        else:
            ffreq = 0

        # Shift the ffreq_data array to the left and update the last element
        if volume > -70:
            ffreq_data = np.roll(ffreq_data, -1)
            ffreq_data[-1] = ffreq
        else:
            ffreq_data = np.roll(ffreq_data, -1)
            ffreq_data[-1] = 0

    # 戻り値は pyaudio の仕様に従うこと
    return None, pyaudio.paContinue


# マイクからの音声入力にはpyaudioを使用
# ここではpyaudioの再生ストリームを作成
# 【注意】シフトサイズごとに指定された関数が呼び出される
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=SAMPLING_RATE,
    input=True,  # ここをTrueにするとマイクからの入力になる
    frames_per_buffer=SHIFT_SIZE,  # シフトサイズ
    stream_callback=input_callback,  # ここでした関数がマイク入力の度に呼び出される（frame_per_bufferで指定した単位で）
)


#
# (3) mp3ファイル音楽を再生する処理
#


# mp3ファイル名
# ここは各自の音源ファイルに合わせて変更すること
filename = "./songs/twinkle-twinkle-little-star-short.mp3"
# filename = "./songs/shs-test.wav"
# filename = "twinkle-twinkle-little-star.mp3"

#
# 【注意】なるべく1チャネルの音声を利用すること
# ステレオ（2チャネル）の場合は SoX などでモノラルに変換できる
# sox stereo.wav -c 1 mono.wav
#

# pydubを使用して音楽ファイルを読み込む
audio_data = AudioSegment.from_mp3(filename)

# 音声ファイルの再生にはpyaudioを使用
# ここではpyaudioの再生ストリームを作成
p_play = pyaudio.PyAudio()
stream_play = p_play.open(
    format=p.get_format_from_width(
        audio_data.sample_width
    ),  # ストリームを読み書きするときのデータ型
    channels=audio_data.channels,  # チャネル数
    rate=audio_data.frame_rate,  # サンプリングレート
    output=True,  # 出力モードに設定
)


# sub-harmonic summation to extract melody
def shs(spectrum, freqs, candidate_freqs, num_harmonics=5):
    likelihood = np.zeros(len(candidate_freqs))

    for i, f0 in enumerate(candidate_freqs):
        harmonic_sum = 0
        for h in range(1, num_harmonics + 1):
            harmonic_freq = h * f0
            idx = np.argmin(np.abs(freqs - harmonic_freq))
            harmonic_sum += spectrum[idx]
        likelihood[i] = harmonic_sum

    return candidate_freqs[np.argmax(likelihood)]


def generate_sinusoid(sampling_rate, frequency, duration):
    sampling_interval = 1.0 / sampling_rate
    t = np.arange(sampling_rate * duration) * sampling_interval
    waveform = np.sin(2.0 * math.pi * frequency * t)
    return waveform


# 楽曲のデータを格納
x_stacked_data_music = np.array([])
freqs = np.linspace(0, SAMPLING_RATE / 2, FRAME_SIZE // 2 + 1)
candidate_freqs = np.linspace(50, 400, 500)

melody_computed = []


def extract_melody():
    x, _ = librosa.load(filename, sr=SAMPLING_RATE)

    spectrogram = []

    global melody_computed

    for i in range(0, len(x) - FRAME_SIZE, SHIFT_SIZE):
        idx = int(i)
        x_frame = x[idx : idx + FRAME_SIZE]
        x_frame = x_frame * hamming_window

        # spectrogram
        fft_spec = np.fft.rfft(x_frame)
        spectrogram.append(np.abs(fft_spec))

        # shs
        estimated_f0 = shs(np.abs(fft_spec), freqs, candidate_freqs)
        melody_computed.append(estimated_f0)

    # # Smooth the melody_computed to get rid of random hikes and downs
    # melody_computed = np.array(melody_computed)
    # smoothed_melody = np.copy(melody_computed)
    # for i in range(1, len(melody_computed) - 1):
    #     smoothed_melody[i] = (
    #         melody_computed[i - 1] + melody_computed[i] + melody_computed[i + 1]
    #     ) / 3

    # melody_computed = smoothed_melody.tolist()


# pydubで読み込んだ音楽ファイルを再生する部分のみ関数化する
# 別スレッドで実行するため
def play_music():
    # この関数は別スレッドで実行するため
    # メインスレッドで定義した以下の２つの変数を利用できるように global 宣言する
    global is_gui_running, audio_data, now_playing_sec, x_stacked_data_music, spectrogram_data_music, melody_data, melody_computed, music_tempo

    # pydubのmake_chunksを用いて音楽ファイルのデータを切り出しながら読み込む
    # 第二引数には何ミリ秒毎に読み込むかを指定
    # ここでは10ミリ秒ごとに読み込む

    size_frame_music = 10  # 10ミリ秒毎に読み込む

    idx = 0
    melody_idx = 0

    # make_chunks関数を使用して一定のフレーム毎に音楽ファイルを読み込む
    #
    # なぜ再生するだけのためにフレーム毎の処理をするのか？
    # 音楽ファイルに対しても何らかの処理を行えるようにするため（このサンプルプログラムでは行っていない）
    # おまけに，再生位置も計算することができる
    for chunk in make_chunks(audio_data, size_frame_music):
        # GUIが終了してれば，この関数の処理も終了する
        if not is_gui_running:
            break
        
        # Adjust the playback speed according to the music tempo
        chunk = chunk._spawn(chunk.raw_data, overrides={
            "frame_rate": int(chunk.frame_rate * music_tempo)
        }).set_frame_rate(chunk.frame_rate)

        # pyaudioの再生ストリームに切り出した音楽データを流し込む
        # 再生が完了するまで処理はここでブロックされる
        stream_play.write(chunk._data)

        # 現在の再生位置を計算（単位は秒）
        now_playing_sec = (idx * size_frame_music) / 1000.0

        # load first 5 seconds of melody
        if now_playing_sec >= melody_idx * 5:
            start_idx = melody_idx * int(SAMPLING_RATE / SHIFT_SIZE * 5)
            end_idx = (melody_idx + 1) * int(SAMPLING_RATE / SHIFT_SIZE * 5)

            melody_data = np.zeros(len(time_x_data))
            melody_segment = melody_computed[start_idx:end_idx]
            melody_data[-len(melody_segment) :] = melody_segment

            melody_idx += 1

        idx += 1

        #
        # 【補足】
        # 楽曲のスペクトログラムを計算する場合には下記のように楽曲のデータを受け取る
        # ただし，音声データの値は -1.0~1.0 ではなく，16bit の整数値であるので正規化を施している
        # また十分なサイズの音声データを確保してからfftを実行すること
        # 楽曲が44.1kHzの場合，44100 / (1000/10) のサイズのデータとなる
        # 以下では処理のみを行い，表示はしない．表示をするには animate 関数の中身を変更すること

        # データの取得
        data_music = np.array(chunk.get_array_of_samples())

        # 正規化
        data_music = data_music / np.iinfo(np.int32).max

        # 現在のフレームとこれまでに入力されたフレームを連結
        x_stacked_data_music = np.concatenate([x_stacked_data_music, data_music])

        # フレームサイズ分のデータがあれば処理を行う
        if len(x_stacked_data_music) >= FRAME_SIZE:
            # フレームサイズからはみ出した過去のデータは捨てる
            x_stacked_data_music = x_stacked_data_music[
                len(x_stacked_data_music) - FRAME_SIZE :
            ]

            # スペクトルを計算
            fft_spec = np.fft.rfft(x_stacked_data_music * hamming_window)
            fft_log_abs_spec = np.log10(np.abs(fft_spec) + EPSILON)[:-1]

            # ２次元配列上で列方向（時間軸方向）に１つずらし（戻し）
            # 最後の列（＝最後の時刻のスペクトルがあった位置）に最新のスペクトルデータを挿入
            spectrogram_data_music = np.roll(spectrogram_data_music, -1, axis=1)
            spectrogram_data_music[:, -1] = fft_log_abs_spec
            


# 再生時間の表示を随時更新する関数
def update_gui_text():
    global is_gui_running, now_playing_sec, text

    while True:
        # GUIが表示されていれば再生位置（秒）をテキストとしてGUI上に表示
        if is_gui_running:
            text.set("%.3f" % now_playing_sec)

        # 0.01秒ごとに更新
        time.sleep(0.01)


def update_progress_bar():
    global now_playing_sec, audio_data
    song_length = len(audio_data) / 1000.0  # Length of the song in seconds
    progress = (now_playing_sec / song_length) * 100
    song_progress["value"] = progress
    root.after(100, update_progress_bar)


# 再生時間を表す
now_playing_sec = 0.0

# Initialize lyrics at the start
lyrics_file = filename.replace(".mp3", ".txt")
lyrics_lines = []
lyrics_timings = []

if os.path.exists(lyrics_file):
    with open(lyrics_file, "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                lyrics_timings.append(float(parts[0]))
                lyrics_lines.append(parts[1])

    if len(lyrics_lines) > 0:
        lyrics_top.config(text=lyrics_lines[0])
    if len(lyrics_lines) > 1:
        lyrics_bottom.config(text=lyrics_lines[1])


def update_lyrics():
    global lyrics_top, lyrics_bottom, now_playing_sec

    current_line = 0
    for i, timing in enumerate(lyrics_timings):
        if now_playing_sec < timing:
            current_line = i
            break

    if current_line < len(lyrics_lines):
        lyrics_top.config(text=lyrics_lines[current_line])
    if current_line + 1 < len(lyrics_lines):
        lyrics_bottom.config(text=lyrics_lines[current_line + 1])

    root.after(500, update_lyrics)


# 音楽を再生するパートを関数化したので，それを別スレッドで（GUIのため）再生開始
t_play_music = threading.Thread(target=play_music)
t_play_music.setDaemon(
    True
)  # GUIが消されたときにこの別スレッドの処理も終了されるようにするため

# 再生時間の表示を随時更新する関数を別スレッドで開始
t_update_gui = threading.Thread(target=update_gui_text)
t_update_gui.setDaemon(
    True
)  # GUIが消されたときにこの別スレッドの処理も終了されるようにするため


#
# (4) 全体の処理を実行
#
if __name__ == "__main__":
    # GUIの開始フラグをTrueに
    is_gui_running = True

    # open song selection pop-up
    # select_song()
    extract_melody()

    # 上記で設定したスレッドを開始（直前のフラグを立ててから）
    t_play_music.start()
    t_update_gui.start()

    update_lyrics()
    update_progress_bar()

    # GUIを開始，GUIが表示されている間は処理はここでストップ
    tkinter.mainloop()

    # GUIの開始フラグをFalseに = 音楽再生スレッドのループを終了
    is_gui_running = False

    # 終了処理
    stream_play.stop_stream()
    stream_play.close()
    p_play.terminate()
