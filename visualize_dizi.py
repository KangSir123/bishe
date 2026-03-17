# ===================== 完整的独立可视化脚本：visualize_dizi.py =====================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from pathlib import Path
import os

# ===================== 1. 关键配置（和原训练文件保持一致） =====================
DEVICE = torch.device("cpu")
SAMPLERATE = 44100
SEGMENT_SEC = 6
DATA_ROOT = Path("music_data")  # 你的数据集路径，和原训练一致

# 打印当前工作目录（方便找图片）
print("🔍 当前工作目录：", os.getcwd())
print("📌 图片生成后会保存在：", os.path.join(os.getcwd(), "separation_result.png"))

# ===================== 2. Demucs模型定义（完整，必须和原训练一致） =====================
class Demucs(nn.Module):
    def __init__(self, sources=["dizi"], audio_channels=2, samplerate=44100, depth=6, kernel_size=4, stride=2):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.samplerate = samplerate
        
        # 编码器
        self.encoders = nn.ModuleList()
        in_channels = audio_channels
        for i in range(depth):
            self.encoders.append(
                nn.Conv1d(in_channels, 64 * (2 ** i), kernel_size, stride, padding=kernel_size//2)
            )
            in_channels = 64 * (2 ** i)
        
        # 解码器
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            out_channels = 64 * (2 ** (i-1)) if i > 0 else audio_channels * len(sources)
            self.decoders.append(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, output_padding=stride-1)
            )
            in_channels = out_channels
        
        self.relu = nn.ReLU()

    def forward(self, x):
        enc_outs = []
        for encoder in self.encoders:
            x = self.relu(encoder(x))
            enc_outs.append(x)
        
        for i, decoder in enumerate(self.decoders):
            if i > 0:
                enc_tensor = enc_outs[-(i+1)]
                min_len = min(x.shape[-1], enc_tensor.shape[-1])
                x = x[:, :, :min_len] + enc_tensor[:, :, :min_len]
            x = self.relu(decoder(x))
        
        x = x.view(x.shape[0], len(self.sources), self.audio_channels, x.shape[-1])
        if x.shape[-1] > self.samplerate * SEGMENT_SEC:
            x = x[:, :, :, :self.samplerate * SEGMENT_SEC]
        return {self.sources[i]: x[:, i] for i in range(len(self.sources))}

# ===================== 3. 数据加载函数（仅加载单条测试音频） =====================
def load_track(track_dir):
    mix_wav, sr = torchaudio.load(track_dir / "mix.wav")
    dizi_wav, _ = torchaudio.load(track_dir / "dizi.wav")
    
    if sr != SAMPLERATE:
        mix_wav = torchaudio.transforms.Resample(sr, SAMPLERATE)(mix_wav)
        dizi_wav = torchaudio.transforms.Resample(sr, SAMPLERATE)(dizi_wav)
    if mix_wav.shape[0] != 2:
        mix_wav = mix_wav.repeat(2, 1)
        dizi_wav = dizi_wav.repeat(2, 1)
    
    target_len = SAMPLERATE * SEGMENT_SEC
    if mix_wav.shape[1] >= target_len:
        mix_wav = mix_wav[:, :target_len]
        dizi_wav = dizi_wav[:, :target_len]
    else:
        pad_len = target_len - mix_wav.shape[1]
        mix_wav = F.pad(mix_wav, (0, pad_len))
        dizi_wav = F.pad(dizi_wav, (0, pad_len))
    
    return mix_wav.to(DEVICE), dizi_wav.to(DEVICE)

# ===================== 4. 可视化函数（完整） =====================
def plot_separation_result(mix_audio, target_dizi, pred_dizi, samplerate=44100, save_path="separation_result.png"):
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    
    # 取左声道
    mix = mix_audio[0].cpu().numpy()
    target = target_dizi[0].cpu().numpy()
    pred = pred_dizi[0].cpu().numpy()
    time_axis = np.linspace(0, len(mix)/samplerate, len(mix))
    
    # 创建画布
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("笛子音频分离效果对比", fontsize=16, fontweight="bold")

    # 第一行：波形图
    axes[0,0].plot(time_axis, mix, color="#2E86AB", alpha=0.8)
    axes[0,0].set_title("混合音频（笛子+其他乐器）", fontsize=12)
    axes[0,0].set_xlabel("时间 (秒)")
    axes[0,0].set_ylabel("振幅")
    axes[0,0].set_ylim(-1, 1)
    axes[0,0].grid(alpha=0.3)
    
    axes[0,1].plot(time_axis, target, color="#A23B72", alpha=0.8)
    axes[0,1].set_title("真实笛子音频（标签）", fontsize=12)
    axes[0,1].set_xlabel("时间 (秒)")
    axes[0,1].set_ylabel("振幅")
    axes[0,1].set_ylim(-1, 1)
    axes[0,1].grid(alpha=0.3)
    
    axes[0,2].plot(time_axis, pred, color="#F18F01", alpha=0.8)
    axes[0,2].set_title("模型预测笛子音频", fontsize=12)
    axes[0,2].set_xlabel("时间 (秒)")
    axes[0,2].set_ylabel("振幅")
    axes[0,2].set_ylim(-1, 1)
    axes[0,2].grid(alpha=0.3)

    # 第二行：频谱图
    n_fft = 2048
    hop_length = 512
    
    S_mix = librosa.amplitude_to_db(np.abs(librosa.stft(mix, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    img1 = librosa.display.specshow(S_mix, sr=samplerate, hop_length=hop_length, x_axis="time", y_axis="hz", ax=axes[1,0], cmap="viridis")
    axes[1,0].set_title("混合音频频谱", fontsize=12)
    axes[1,0].set_xlabel("时间 (秒)")
    axes[1,0].set_ylabel("频率 (Hz)")
    plt.colorbar(img1, ax=axes[1,0], format="%+2.0f dB")
    
    S_target = librosa.amplitude_to_db(np.abs(librosa.stft(target, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    img2 = librosa.display.specshow(S_target, sr=samplerate, hop_length=hop_length, x_axis="time", y_axis="hz", ax=axes[1,1], cmap="viridis")
    axes[1,1].set_title("真实笛子音频频谱", fontsize=12)
    axes[1,1].set_xlabel("时间 (秒)")
    axes[1,1].set_ylabel("频率 (Hz)")
    plt.colorbar(img2, ax=axes[1,1], format="%+2.0f dB")
    
    S_pred = librosa.amplitude_to_db(np.abs(librosa.stft(pred, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    img3 = librosa.display.specshow(S_pred, sr=samplerate, hop_length=hop_length, x_axis="time", y_axis="hz", ax=axes[1,2], cmap="viridis")
    axes[1,2].set_title("模型预测笛子音频频谱", fontsize=12)
    axes[1,2].set_xlabel("时间 (秒)")
    axes[1,2].set_ylabel("频率 (Hz)")
    plt.colorbar(img3, ax=axes[1,2], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 图片已保存！路径：{os.path.abspath(save_path)}")  # 打印绝对路径

# ===================== 5. 主程序（加载模型+测试音频+生成图片） =====================
if __name__ == "__main__":
    # 加载模型（替换为你的模型路径）
    model = Demucs(
        sources=["dizi"],
        audio_channels=2,
        samplerate=SAMPLERATE,
        depth=6
    ).to(DEVICE)
    
    # 加载训练好的权重（确保dizi_best_model.pth在当前目录，或写绝对路径）
    model_path = "dizi_best_model.pth"  # 若不在当前目录，改为绝对路径，比如 "C:/code/dizi_best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件：{model_path}，请检查路径！")
    else:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
    
    # 加载测试音频（取valid集第一个样本）
    valid_track_dirs = list((DATA_ROOT / "valid").glob("track_*"))
    if not valid_track_dirs:
        print("❌ 找不到valid集的track目录，请检查DATA_ROOT路径！")
    else:
        test_track_dir = valid_track_dirs[0]
        print(f"📤 正在处理测试样本：{test_track_dir}")
        mix_audio, target_dizi = load_track(test_track_dir)
        
        # 模型预测
        with torch.no_grad():
            pred = model(mix_audio.unsqueeze(0))
            pred_dizi = pred["dizi"][0]
        
        # 生成并保存图片
        plot_separation_result(mix_audio, target_dizi, pred_dizi, samplerate=SAMPLERATE)