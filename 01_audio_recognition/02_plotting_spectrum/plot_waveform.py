#
# 計 算 機 科 学 実 験 及 演 習 4「 音 響 信 号 処 理 」
# サ ン プ ル ソ ー ス コ ー ド
#
# 音 声 フ ァ イ ル を 読 み 込 み ， 波 形 を 図 示 す る ．
#

# ラ イ ブ ラ リ の 読 み 込 み
import matplotlib.pyplot as plt
import librosa

# サ ン プ リ ン グ レ ー ト
SR = 16000

# 音 声 フ ァ イ ル の 読 み 込 み
x, _ = librosa.load("../01_recording_aiueo/aiueo_long.wav", sr=SR)

# x に 波 形 デ ー タ が 保 存 さ れ る
# 第 二 戻 り 値 は サ ン プ リ ン グ レ ー ト （ こ こ で は 必 要 な い の で _ と し て い る ）

# 波 形 デ ー タ を 標 準 出 力 し て 確 認
print(x)

#
# 波 形 を 画 像 に 表 示 ・ 保 存
#

# 画 像 と し て 保 存 す る た め の 設 定
# 画 像 サ イ ズ を 1000 x 400 に 設 定
fig = plt.figure(figsize=(10, 4))

# 波 形 を 描 画
plt.plot(x)  # 描 画 デ ー タ を 追 加
plt.xlabel("Sampling point")  # x 軸 の ラ ベ ル を 設 定
plt.show()  # 表 示

# 画 像 フ ァ イ ル に 保 存
fig.savefig("plot-waveform.png")
