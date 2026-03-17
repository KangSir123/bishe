import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pathlib import Path

# ===================== 1. 和训练脚本完全一致的Demucs模型类（depth=6+通道数64*(2^i)） =====================
class Demucs(nn.Module):
    """必须和训练脚本的Demucs类完全一致！"""
    def __init__(self, sources=["dizi"], audio_channels=2, samplerate=44100, depth=6, kernel_size=4, stride=2):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.samplerate = samplerate
        
        # 编码器（depth=6，通道数64*(2^i)）
        self.encoders = nn.ModuleList()
        in_channels = audio_channels
        for i in range(depth):
            self.encoders.append(
                nn.Conv1d(in_channels, 64 * (2 ** i), kernel_size, stride, padding=kernel_size//2)
            )
            in_channels = 64 * (2 ** i)
        
        # 解码器（对应depth=6）
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            out_channels = 64 * (2 ** (i-1)) if i > 0 else audio_channels * len(sources)
            self.decoders.append(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, output_padding=stride-1)
            )
            in_channels = out_channels
        
        self.relu = nn.ReLU()

    def forward(self, x):
        """前向传播：和训练脚本完全一致"""
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
        return {self.sources[i]: x[:, i] for i in range(len(self.sources))}

# ===================== 2. 配置参数（和训练脚本一致） =====================
DEVICE = torch.device("cpu")
SAMPLERATE = 44100  # 必须和训练一致
MODEL_PATH = "dizi_best_model.pth"  # 训练好的最优模型
INPUT_AUDIO_PATH = "test1.wav"  # 待分离的混合音频
OUTPUT_AUDIO_PATH = "separated_dizi1.wav"  # 分离后的笛子音频

# ===================== 3. 加载训练好的模型 =====================
def load_trained_model():
    """加载模型：确保结构和训练100%匹配"""
    # 初始化模型（depth=6，和训练一致）
    model = Demucs(
        sources=["dizi"],
        audio_channels=2,
        samplerate=SAMPLERATE,
        depth=6  # 核心：和训练脚本的depth完全一致
    ).to(DEVICE)
    
    # 加载权重（map_location确保CPU加载）
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    
    # 切换到推理模式
    model.eval()
    print(f"✅ 模型加载成功：{MODEL_PATH}")
    return model

# ===================== 4. 处理输入音频 =====================
def process_input_audio(audio_path):
    """预处理：和训练脚本的音频处理逻辑一致"""
    wav, sr = torchaudio.load(audio_path)
    
    # 统一采样率
    if sr != SAMPLERATE:
        wav = torchaudio.transforms.Resample(sr, SAMPLERATE)(wav)
    
    # 统一双声道
    if wav.shape[0] != 2:
        wav = wav.repeat(2, 1)
    
    # 增加批次维度（模型输入要求：[batch, channels, length]）
    wav = wav.unsqueeze(0).to(DEVICE)
    print(f"✅ 音频处理完成：输入格式 {wav.shape}")
    return wav

# ===================== 5. 核心推理 =====================
def separate_dizi():
    # 检查文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误：未找到模型文件 {MODEL_PATH}")
        return
    if not os.path.exists(INPUT_AUDIO_PATH):
        print(f"❌ 错误：未找到输入音频 {INPUT_AUDIO_PATH}")
        return
    
    # 加载模型
    model = load_trained_model()
    
    # 处理输入音频
    input_wav = process_input_audio(INPUT_AUDIO_PATH)
    
    # 推理（禁用梯度计算）
    with torch.no_grad():
        pred = model(input_wav)
        dizi_wav = pred["dizi"]
    
    # 后处理：放大幅值+裁剪
    dizi_wav = dizi_wav.squeeze(0)  # 去掉批次维度
    dizi_wav = dizi_wav * 10  # 放大幅值，解决没声音问题
    dizi_wav = torch.clamp(dizi_wav, -1.0, 1.0)  # 限制幅值避免爆音
    
    # 保存结果
    torchaudio.save(OUTPUT_AUDIO_PATH, dizi_wav.cpu(), SAMPLERATE)
    print(f"🎉 分离完成！笛子音频已保存为：{OUTPUT_AUDIO_PATH}")

# ===================== 运行推理 =====================
if __name__ == "__main__":
    separate_dizi()