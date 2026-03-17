import torch
import torch.nn as nn  # 核心修正：提前导入torch.nn并命名为nn
import torchaudio
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")

# ===================== 1. 核心配置 =====================
DEVICE = torch.device("cpu")
SAMPLERATE = 44100
SEGMENT_SEC = 6
TARGET_LEN = SAMPLERATE * SEGMENT_SEC  # 264600
MODEL_PATH = "dizi_optimized_model.pth"  # 训练好的模型权重路径
OUTPUT_DIR = Path("dizi_separated")
OUTPUT_DIR.mkdir(exist_ok=True)

# ===================== 2. 音频后处理模块（和训练代码一致） =====================
class AudioPostProcess(nn.Module):
    """音频后处理模块：和训练时保持一致"""
    def __init__(self, samplerate=44100):
        super().__init__()
        self.samplerate = samplerate
        self.f0 = 200  # 竹笛最低频率
        self.f1 = 2000 # 竹笛最高频率

    def moving_average_filter(self, audio, window_size=11):
        """移动平均滤波：平滑尖锐脉冲噪声"""
        audio_np = audio.cpu().numpy()
        pad_width = window_size // 2
        for ch in range(audio_np.shape[0]):
            padded = np.pad(audio_np[ch], pad_width, mode='reflect')
            smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
            audio_np[ch] = smoothed[:audio_np.shape[1]]
        return torch.from_numpy(audio_np).to(DEVICE)

    def bandpass_filter(self, audio):
        """带通滤波：保留竹笛核心频率"""
        audio_np = audio.cpu().numpy()
        n_fft = 2048
        hop_length = 512
        for ch in range(audio_np.shape[0]):
            stft = librosa.stft(audio_np[ch], n_fft=n_fft, hop_length=hop_length)
            freq = librosa.fft_frequencies(sr=self.samplerate, n_fft=n_fft)
            mask = np.zeros_like(stft)
            mask[(freq >= self.f0) & (freq <= self.f1)] = 1
            filtered_stft = stft * mask
            filtered = librosa.istft(filtered_stft, hop_length=hop_length, length=audio_np.shape[1])
            audio_np[ch] = filtered
        return torch.from_numpy(audio_np).to(DEVICE)

    def harmonic_reconstruction(self, audio):
        """谐波重构：恢复竹笛音色"""
        audio_np = audio.cpu().numpy()
        for ch in range(audio_np.shape[0]):
            y = audio_np[ch]
            f0, _, _ = librosa.pyin(y, fmin=self.f0, fmax=self.f1, sr=self.samplerate)
            t = np.linspace(0, len(y)/self.samplerate, len(y))
            harmonic = np.zeros_like(y)
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                mean_f0 = np.mean(valid_f0)
                for h in range(1, 4):
                    harmonic += 0.3/h * np.sin(2 * np.pi * h * mean_f0 * t)
            audio_np[ch] = 0.7 * y + 0.3 * harmonic
        return torch.from_numpy(audio_np).to(DEVICE)

    def amplitude_normalize(self, audio):
        """幅度归一化"""
        max_amp = torch.max(torch.abs(audio))
        if max_amp > 1e-6:
            audio = audio / max_amp
        return audio

    def forward(self, audio):
        audio = self.moving_average_filter(audio)
        audio = self.bandpass_filter(audio)
        audio = self.harmonic_reconstruction(audio)
        audio = self.amplitude_normalize(audio)
        return audio

# ===================== 3. 改进版Demucs模型（和训练代码完全一致） =====================
class ImprovedDemucs(nn.Module):
    def __init__(self, sources=["dizi"], audio_channels=2, samplerate=44100, depth=6, kernel_size=4, stride=2):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.samplerate = samplerate
        
        # 编码器（Conv1d + BatchNorm1d + ReLU）
        self.encoders = nn.ModuleList()
        in_channels = audio_channels
        for i in range(depth):
            self.encoders.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, 64 * (2 ** i), kernel_size, stride, padding=kernel_size//2),
                    nn.BatchNorm1d(64 * (2 ** i)),
                    nn.ReLU()
                )
            )
            in_channels = 64 * (2 ** i)
        
        # 解码器（ConvTranspose1d + BatchNorm1d + ReLU）
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            out_channels = 64 * (2 ** (i-1)) if i > 0 else audio_channels * len(sources)
            self.decoders.append(
                nn.Sequential(
                    nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, output_padding=stride-1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU()
                )
            )
            in_channels = out_channels
        
        # 输出降噪层
        self.noise_reduce = nn.Conv1d(audio_channels * len(sources), audio_channels * len(sources), 
                                      kernel_size=7, padding=3, groups=audio_channels * len(sources))

    def forward(self, x):
        """
        输入：[batch_size, audio_channels, length]
        输出：{source: [batch_size, audio_channels, length]}
        """
        enc_outs = []
        for encoder in self.encoders:
            x = encoder(x)
            enc_outs.append(x)
        
        for i, decoder in enumerate(self.decoders):
            if i > 0:
                enc_tensor = enc_outs[-(i+1)]
                min_len = min(x.shape[-1], enc_tensor.shape[-1])
                x = x[:, :, :min_len] + enc_tensor[:, :, :min_len]
            x = decoder(x)
        
        x = self.noise_reduce(x)
        x = x.view(x.shape[0], len(self.sources), self.audio_channels, x.shape[-1])
        if x.shape[-1] > self.samplerate * SEGMENT_SEC:
            x = x[:, :, :, :self.samplerate * SEGMENT_SEC]
        
        return {self.sources[i]: x[:, i] for i in range(len(self.sources))}

# ===================== 4. 加载训练好的模型 =====================
def load_trained_model(model_path=MODEL_PATH):
    """加载和训练时结构一致的ImprovedDemucs模型"""
    # 初始化模型（结构必须和训练时完全一致）
    model = ImprovedDemucs(
        sources=["dizi"],
        audio_channels=2,
        samplerate=SAMPLERATE,
        depth=6
    ).to(DEVICE)
    
    # 加载权重（map_location适配CPU）
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()  # 切换到推理模式
    print(f"✅ 成功加载模型权重：{model_path}")
    return model

# ===================== 5. 音频预处理（和训练时一致） =====================
def preprocess_audio(audio_path):
    """预处理输入音频，匹配模型输入格式"""
    # 加载音频
    wav, sr = torchaudio.load(audio_path)
    # 统一采样率
    if sr != SAMPLERATE:
        resample = torchaudio.transforms.Resample(sr, SAMPLERATE)
        wav = resample(wav)
    # 统一声道（转为2声道立体声）
    if wav.shape[0] != 2:
        wav = wav.repeat(2, 1)
    # 统一长度（6秒）
    if wav.shape[1] >= TARGET_LEN:
        wav = wav[:, :TARGET_LEN]
    else:
        pad_len = TARGET_LEN - wav.shape[1]
        wav = torch.nn.functional.pad(wav, (0, pad_len))
    # 转到设备并添加batch维度
    wav = wav.to(DEVICE).unsqueeze(0)
    return wav, Path(audio_path).name

# ===================== 6. 核心分离函数 =====================
def separate_dizi(audio_path=None):
    """
    分离音频中的竹笛信号
    :param audio_path: 输入混合音频路径，若为None则使用命令行参数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="分离混合音频中的竹笛信号")
    parser.add_argument("--audio", type=str, default=audio_path, help="输入混合音频路径（如mix.wav）")
    args = parser.parse_args()
    
    if args.audio is None:
        raise ValueError("请指定输入音频路径！示例：python infer_dizi2.py --audio mix.wav")
    
    audio_path = args.audio
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"输入音频文件不存在：{audio_path}")
    
    # 1. 加载模型和后处理器
    model = load_trained_model()
    post_processor = AudioPostProcess(samplerate=SAMPLERATE)
    
    # 2. 预处理音频
    print(f"🔧 预处理音频：{audio_path}")
    input_wav, audio_name = preprocess_audio(audio_path)
    
    # 3. 推理（无梯度计算）
    print("🎵 开始分离竹笛信号...")
    with torch.no_grad():
        pred = model(input_wav)
        dizi_wav = pred["dizi"][0]  # 去除batch维度：[2, 264600]
    
    # 4. 音频后处理（降噪+重构）
    dizi_wav = post_processor(dizi_wav)
    
    # 5. 保存分离结果
    output_path = OUTPUT_DIR / f"dizi_separated_{audio_name}"
    # 转为numpy数组，确保格式正确
    dizi_np = dizi_wav.cpu().numpy().T  # 转置为(长度, 通道)，符合soundfile要求
    sf.write(output_path, dizi_np, SAMPLERATE)
    
    print(f"✅ 分离完成！结果保存至：{output_path}")
    print(f"📌 分离后音频参数：采样率={SAMPLERATE}Hz，时长={SEGMENT_SEC}秒，声道=2（立体声）")

# ===================== 7. 主函数 =====================
if __name__ == "__main__":
    # 执行分离
    separate_dizi()