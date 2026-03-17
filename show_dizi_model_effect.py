# ===================== 完整可视化脚本：展示dizi_best_model.pth效果 =====================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from pathlib import Path
import os

# ===================== 1. 核心配置（必须和训练时保持一致） =====================
DEVICE = torch.device("cpu")
SAMPLERATE = 44100  # 训练时的采样率
SEGMENT_SEC = 6     # 训练时的音频长度（秒）
DATA_ROOT = Path("music_data")  # 数据集根路径（和训练一致）
MODEL_PATH = "dizi_best_model.pth"  # 最优模型路径
SAVE_IMAGE_PATH = "dizi_best_model_result.png"  # 生成的效果图路径

# 打印关键路径（方便排查）
print("🔍 关键路径信息：")
print(f"   - 当前工作目录：{os.getcwd()}")
print(f"   - 模型文件路径：{os.path.abspath(MODEL_PATH)}")
print(f"   - 效果图保存路径：{os.path.abspath(SAVE_IMAGE_PATH)}")
print(f"   - 数据集路径：{DATA_ROOT.absolute()}")

# ===================== 2. Demucs模型定义（和训练时完全一致） =====================
class Demucs(nn.Module):
    def __init__(self, sources=["dizi"], audio_channels=2, samplerate=44100, depth=6, kernel_size=4, stride=2):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.samplerate = samplerate
        
        # 编码器（下采样）
        self.encoders = nn.ModuleList()
        in_channels = audio_channels
        for i in range(depth):
            self.encoders.append(
                nn.Conv1d(in_channels, 64 * (2 ** i), kernel_size, stride, padding=kernel_size//2)
            )
            in_channels = 64 * (2 ** i)
        
        # 解码器（上采样）
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            out_channels = 64 * (2 ** (i-1)) if i > 0 else audio_channels * len(sources)
            self.decoders.append(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, output_padding=stride-1)
            )
            in_channels = out_channels
        
        self.relu = nn.ReLU()

    def forward(self, x):
        """前向传播：输入[batch, channels, length]，输出{source: 音频张量}"""
        enc_outs = []
        for encoder in self.encoders:
            x = self.relu(encoder(x))
            enc_outs.append(x)
        
        for i, decoder in enumerate(self.decoders):
            if i > 0:
                enc_tensor = enc_outs[-(i+1)]
                min_len = min(x.shape[-1], enc_tensor.shape[-1])
                x = x[:, :, :min_len] + enc_tensor[:, :, :min_len]  # 跳过连接
            x = self.relu(decoder(x))
        
        x = x.view(x.shape[0], len(self.sources), self.audio_channels, x.shape[-1])
        if x.shape[-1] > self.samplerate * SEGMENT_SEC:
            x = x[:, :, :, :self.samplerate * SEGMENT_SEC]
        return {self.sources[i]: x[:, i] for i in range(len(self.sources))}

# ===================== 3. 数据加载函数（加载单条测试音频） =====================
def load_track(track_dir):
    """加载mix.wav和dizi.wav，统一格式（2声道、44100采样率、6秒长度）"""
    # 读取音频文件
    mix_path = track_dir / "mix.wav"
    dizi_path = track_dir / "dizi.wav"
    if not mix_path.exists() or not dizi_path.exists():
        raise FileNotFoundError(f"❌ {track_dir} 中缺少mix.wav或dizi.wav")
    
    mix_wav, sr = torchaudio.load(mix_path)
    dizi_wav, _ = torchaudio.load(dizi_path)
    
    # 统一采样率
    if sr != SAMPLERATE:
        resample = torchaudio.transforms.Resample(sr, SAMPLERATE)
        mix_wav = resample(mix_wav)
        dizi_wav = resample(dizi_wav)
    
    # 统一声道（转为立体声）
    if mix_wav.shape[0] != 2:
        mix_wav = mix_wav.repeat(2, 1)
        dizi_wav = dizi_wav.repeat(2, 1)
    
    # 统一长度（6秒）
    target_len = SAMPLERATE * SEGMENT_SEC
    if mix_wav.shape[1] >= target_len:
        mix_wav = mix_wav[:, :target_len]
        dizi_wav = dizi_wav[:, :target_len]
    else:
        pad_len = target_len - mix_wav.shape[1]
        mix_wav = F.pad(mix_wav, (0, pad_len))
        dizi_wav = F.pad(dizi_wav, (0, pad_len))
    
    return mix_wav.to(DEVICE), dizi_wav.to(DEVICE)

# ===================== 4. 量化指标计算（评估模型效果） =====================
def calculate_separation_metrics(pred_audio, target_audio):
    """
    计算2个核心量化指标（评估分离效果）：
    - L1损失：预测与真实音频的平均绝对误差（越小越好）
    - SI-SDR：尺度不变信噪比（越大越好，>0说明分离有效，>10为优秀）
    """
    # 转为numpy数组（取左声道，消除batch/声道维度）
    pred = pred_audio[0].cpu().numpy()
    target = target_audio[0].cpu().numpy()
    
    # 1. 计算L1损失
    l1_loss = np.mean(np.abs(pred - target))
    
    # 2. 计算SI-SDR（标准化，避免尺度影响）
    target = target - np.mean(target)
    pred = pred - np.mean(pred)
    dot_product = np.sum(pred * target)
    s_target = dot_product * target / (np.sum(target**2) + 1e-8)  # 目标信号分量
    e_noise = pred - s_target  # 噪声分量
    si_sdr = 10 * np.log10((np.sum(s_target**2) + 1e-8) / (np.sum(e_noise**2) + 1e-8))
    
    return {
        "L1损失": round(l1_loss, 4),
        "SI-SDR (dB)": round(si_sdr, 2)
    }

# ===================== 5. 可视化函数（生成效果对比图） =====================
def plot_model_effect(mix_audio, target_dizi, pred_dizi, metrics, save_path):
    """
    生成模型效果对比图：
    - 第一行：时域波形图（混合音/真实笛子/预测笛子）
    - 第二行：频域频谱图（混合音/真实笛子/预测笛子）
    - 标题包含量化指标，直观展示模型效果
    """
    # 解决中文字体问题（通用配置）
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示
    
    # 提取单声道数据（立体声左右声道一致）
    mix = mix_audio[0].cpu().numpy()
    target = target_dizi[0].cpu().numpy()
    pred = pred_dizi[0].cpu().numpy()
    time_axis = np.linspace(0, len(mix)/SAMPLERATE, len(mix))  # 时间轴（秒）
    
    # 创建画布（2行3列，尺寸18×10）
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"dizi_best_model.pth 笛子分离效果对比\nL1损失={metrics['L1损失']} | SI-SDR={metrics['SI-SDR (dB)']} dB",
        fontsize=16, fontweight="bold", y=0.98
    )

    # --------------------- 第一行：时域波形图 ---------------------
    # 1. 混合音频波形
    axes[0,0].plot(time_axis, mix, color="#2E86AB", alpha=0.8)
    axes[0,0].set_title("混合音频（笛子+其他乐器）", fontsize=12)
    axes[0,0].set_xlabel("时间 (秒)")
    axes[0,0].set_ylabel("振幅")
    axes[0,0].set_ylim(-1, 1)
    axes[0,0].grid(alpha=0.3)
    
    # 2. 真实笛子音频波形
    axes[0,1].plot(time_axis, target, color="#A23B72", alpha=0.8)
    axes[0,1].set_title("真实笛子音频（标签）", fontsize=12)
    axes[0,1].set_xlabel("时间 (秒)")
    axes[0,1].set_ylabel("振幅")
    axes[0,1].set_ylim(-1, 1)
    axes[0,1].grid(alpha=0.3)
    
    # 3. 模型预测笛子音频波形
    axes[0,2].plot(time_axis, pred, color="#F18F01", alpha=0.8)
    axes[0,2].set_title("模型预测笛子音频", fontsize=12)
    axes[0,2].set_xlabel("时间 (秒)")
    axes[0,2].set_ylabel("振幅")
    axes[0,2].set_ylim(-1, 1)
    axes[0,2].grid(alpha=0.3)

    # --------------------- 第二行：频域频谱图 ---------------------
    n_fft = 2048  # 频率分辨率
    hop_length = 512  # 时间分辨率
    ref = np.max  # 频谱dB参考值
    
    # 1. 混合音频频谱
    S_mix = librosa.amplitude_to_db(np.abs(librosa.stft(mix, n_fft=n_fft, hop_length=hop_length)), ref=ref)
    img1 = librosa.display.specshow(S_mix, sr=SAMPLERATE, hop_length=hop_length,
                                    x_axis="time", y_axis="hz", ax=axes[1,0], cmap="viridis")
    axes[1,0].set_title("混合音频频谱", fontsize=12)
    axes[1,0].set_xlabel("时间 (秒)")
    axes[1,0].set_ylabel("频率 (Hz)")
    plt.colorbar(img1, ax=axes[1,0], format="%+2.0f dB")
    
    # 2. 真实笛子音频频谱
    S_target = librosa.amplitude_to_db(np.abs(librosa.stft(target, n_fft=n_fft, hop_length=hop_length)), ref=ref)
    img2 = librosa.display.specshow(S_target, sr=SAMPLERATE, hop_length=hop_length,
                                    x_axis="time", y_axis="hz", ax=axes[1,1], cmap="viridis")
    axes[1,1].set_title("真实笛子音频频谱", fontsize=12)
    axes[1,1].set_xlabel("时间 (秒)")
    axes[1,1].set_ylabel("频率 (Hz)")
    plt.colorbar(img2, ax=axes[1,1], format="%+2.0f dB")
    
    # 3. 模型预测笛子音频频谱
    S_pred = librosa.amplitude_to_db(np.abs(librosa.stft(pred, n_fft=n_fft, hop_length=hop_length)), ref=ref)
    img3 = librosa.display.specshow(S_pred, sr=SAMPLERATE, hop_length=hop_length,
                                    x_axis="time", y_axis="hz", ax=axes[1,2], cmap="viridis")
    axes[1,2].set_title("模型预测笛子音频频谱", fontsize=12)
    axes[1,2].set_xlabel("时间 (秒)")
    axes[1,2].set_ylabel("频率 (Hz)")
    plt.colorbar(img3, ax=axes[1,2], format="%+2.0f dB")

    # 调整布局并保存
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 给主标题留空间
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ 效果图已保存：{os.path.abspath(save_path)}")

# ===================== 6. 主程序（加载模型+生成效果对比图） =====================
if __name__ == "__main__":
    # 步骤1：检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 找不到最优模型文件：{MODEL_PATH}，请检查路径！")
    
    # 步骤2：加载最优模型
    print("\n📌 正在加载dizi_best_model.pth...")
    model = Demucs(
        sources=["dizi"],
        audio_channels=2,
        samplerate=SAMPLERATE,
        depth=6
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()  # 切换到评估模式（禁用Dropout/BatchNorm）
    print("✅ 模型加载完成！")
    
    # 步骤3：加载测试音频（验证集第一个样本）
    valid_track_dirs = list((DATA_ROOT / "valid").glob("track_*"))
    if not valid_track_dirs:
        raise FileNotFoundError(f"❌ 在 {DATA_ROOT}/valid 中未找到track_*文件夹，请检查数据集路径！")
    test_track_dir = valid_track_dirs[0]
    print(f"\n📌 正在加载测试音频：{test_track_dir}")
    mix_audio, target_dizi = load_track(test_track_dir)
    print("✅ 测试音频加载完成！")
    
    # 步骤4：用最优模型预测
    print("\n📌 正在用dizi_best_model.pth预测...")
    with torch.no_grad():  # 禁用梯度计算，节省内存
        pred_output = model(mix_audio.unsqueeze(0))  # 增加batch维度
        pred_dizi = pred_output["dizi"][0]  # 去除batch维度
    print("✅ 预测完成！")
    
    # 步骤5：计算量化指标
    metrics = calculate_separation_metrics(pred_dizi, target_dizi)
    print("\n📊 模型量化效果指标：")
    print(f"   - L1损失（越小越好）：{metrics['L1损失']}")
    print(f"   - SI-SDR（越大越好）：{metrics['SI-SDR (dB)']} dB")
    
    # 步骤6：生成并保存效果对比图
    plot_model_effect(mix_audio, target_dizi, pred_dizi, metrics, SAVE_IMAGE_PATH)