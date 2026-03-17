import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from tqdm import tqdm
from pathlib import Path

# ===================== 内置 Demucs 核心模型代码（修复维度不匹配） =====================
class Demucs(nn.Module):
    """极简版Demucs模型（适配单源分离，CPU友好，修复维度不匹配）"""
    def __init__(self, sources=["dizi"], audio_channels=2, samplerate=44100, depth=4, kernel_size=4, stride=2):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.samplerate = samplerate
        
        # 编码器（下采样）- 调整kernel_size=4, stride=2，减少维度误差
        self.encoders = nn.ModuleList()
        in_channels = audio_channels
        for i in range(depth):
            self.encoders.append(
                nn.Conv1d(in_channels, 64 * (2 ** i), kernel_size, stride, padding=kernel_size//2)
            )
            in_channels = 64 * (2 ** i)
        
        # 解码器（上采样）- 匹配编码器参数
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            out_channels = 64 * (2 ** (i-1)) if i > 0 else audio_channels * len(sources)
            self.decoders.append(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, output_padding=stride-1)
            )
            in_channels = out_channels
        
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        """前向传播：x.shape = [batch, channels, length]"""
        # 编码器前向+保存中间结果
        enc_outs = []
        for encoder in self.encoders:
            x = self.relu(encoder(x))
            enc_outs.append(x)
        
        # 解码器前向（修复跳过连接维度不匹配）
        for i, decoder in enumerate(self.decoders):
            if i > 0:
                # 核心修复：裁剪到相同长度（取更短的维度）
                enc_tensor = enc_outs[-(i+1)]
                min_len = min(x.shape[-1], enc_tensor.shape[-1])
                x = x[:, :, :min_len] + enc_tensor[:, :, :min_len]  # 跳过连接
            x = self.relu(decoder(x))
        
        # 拆分到各个源
        x = x.view(x.shape[0], len(self.sources), self.audio_channels, x.shape[-1])
        # 裁剪到原始输入长度（可选，避免长度偏差）
        if x.shape[-1] > self.samplerate * SEGMENT_SEC:
            x = x[:, :, :, :self.samplerate * SEGMENT_SEC]
        # 返回字典：{source: wav}
        return {self.sources[i]: x[:, i] for i in range(len(self.sources))}

# ===================== 手动配置参数（适配你的环境） =====================
DEVICE = torch.device("cpu")  # 强制用CPU
DATA_ROOT = Path("music_data")  # 你的数据集路径
SAMPLERATE = 44100  # 音频采样率
SEGMENT_SEC = 6  # 调整为6秒，使长度=44100*6=264600（更易被2整除）
BATCH_SIZE = 2  # CPU批次大小
EPOCHS = 10  # 先验证训练流程
LR = 3e-4  # 学习率

# ===================== 加载自定义笛子数据集（统一长度） =====================
def load_track(track_dir):
    """加载单条轨道的mix和dizi音频"""
    mix_wav, sr = torchaudio.load(track_dir / "mix.wav")
    dizi_wav, _ = torchaudio.load(track_dir / "dizi.wav")
    
    # 统一采样率+声道
    if sr != SAMPLERATE:
        mix_wav = torchaudio.transforms.Resample(sr, SAMPLERATE)(mix_wav)
        dizi_wav = torchaudio.transforms.Resample(sr, SAMPLERATE)(dizi_wav)
    if mix_wav.shape[0] != 2:
        mix_wav = mix_wav.repeat(2, 1)
        dizi_wav = dizi_wav.repeat(2, 1)
    
    # 核心修复：统一长度为SEGMENT_SEC*SAMPLERATE（无余数）
    target_len = SAMPLERATE * SEGMENT_SEC
    # 强制裁剪（不再随机，避免维度不一致）
    if mix_wav.shape[1] >= target_len:
        mix_wav = mix_wav[:, :target_len]
        dizi_wav = dizi_wav[:, :target_len]
    else:
        # 补零到目标长度
        pad_len = target_len - mix_wav.shape[1]
        mix_wav = F.pad(mix_wav, (0, pad_len))
        dizi_wav = F.pad(dizi_wav, (0, pad_len))
    
    return mix_wav.to(DEVICE), dizi_wav.to(DEVICE)

def get_dataloader(split="train"):
    """构建数据加载器"""
    track_dirs = list((DATA_ROOT / split).glob("track_*"))
    if not track_dirs:
        print(f"⚠️ 警告：{split}集未找到track_*目录，请检查数据集路径！")
        return []
    
    dataloader = []
    for i in range(0, len(track_dirs), BATCH_SIZE):
        batch_dirs = track_dirs[i:i+BATCH_SIZE]
        batch_mix, batch_dizi = [], []
        for dir in batch_dirs:
            mix, dizi = load_track(dir)
            batch_mix.append(mix)
            batch_dizi.append(dizi)
        dataloader.append((torch.stack(batch_mix), torch.stack(batch_dizi)))
    return dataloader

# 加载数据
train_loader = get_dataloader("train")
valid_loader = get_dataloader("valid")
print(f"✅ 数据集加载完成：训练集{len(train_loader)}批次，验证集{len(valid_loader)}批次")

# ===================== 初始化模型+优化器+损失函数 =====================
model = Demucs(
    sources=["dizi"],
    audio_channels=2,
    samplerate=SAMPLERATE,
    depth=6  # 适配CPU的模型深度
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.L1Loss()

# ===================== 核心训练循环 =====================
best_valid_loss = float("inf")
for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Train)")
    for mix_batch, dizi_batch in pbar:
        optimizer.zero_grad()
        pred = model(mix_batch)
        # 裁剪预测结果到和标签相同长度（避免最后维度偏差）
        min_len = min(pred["dizi"].shape[-1], dizi_batch.shape[-1])
        loss = loss_fn(pred["dizi"][:, :, :min_len], dizi_batch[:, :, :min_len])
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # 验证阶段
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for mix_batch, dizi_batch in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Valid)"):
            pred = model(mix_batch)
            min_len = min(pred["dizi"].shape[-1], dizi_batch.shape[-1])
            loss = loss_fn(pred["dizi"][:, :, :min_len], dizi_batch[:, :, :min_len])
            valid_loss += loss.item()
    
    # 计算平均损失
    avg_train_loss = train_loss / len(train_loader) if train_loader else 0
    avg_valid_loss = valid_loss / len(valid_loader) if valid_loader else 0
    
    # 保存最优模型
    if avg_valid_loss < best_valid_loss and valid_loader:
        best_valid_loss = avg_valid_loss
        torch.save(model.state_dict(), "dizi_best_model.pth")
        print(f"✅ 保存最优模型：验证损失 {best_valid_loss:.4f}")
    
    # 打印结果
    print(f"Epoch {epoch+1} | 训练损失：{avg_train_loss:.4f} | 验证损失：{avg_valid_loss:.4f}")

# ===================== 保存最终模型 =====================
#python train_dizi_simple.py
#python infer_dizi.py
#deactivate
torch.save(model.state_dict(), "dizi_final_model.pth")
print("\n🎉 训练完成！模型保存为：")
print("- 最优模型：dizi_best_model.pth")
print("- 最终模型：dizi_final_model.pth") 