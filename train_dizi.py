import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from pathlib import Path
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")

# ===================== 1. 核心配置 =====================
DEVICE = torch.device("cpu")
SAMPLERATE = 44100
SEGMENT_SEC = 6
TARGET_LEN = SAMPLERATE * SEGMENT_SEC  # 264600
DATA_ROOT = Path("music_data")

# 训练参数（优化后）
BATCH_SIZE = 1  # CPU环境稳定优先
LR = 1e-4       # 降低学习率避免震荡
EPOCHS = 10     # 增加训练轮数
PATIENCE = 5    # 早停耐心值
SAVE_PATH = "dizi_optimized_model.pth"  # 最优模型保存路径

# ===================== 2. 音频后处理模块（新增） =====================
class AudioPostProcess(nn.Module):
    """音频后处理模块：滤波+谐波重构+归一化，解决噪声/失真"""
    def __init__(self, samplerate=44100):
        super().__init__()
        self.samplerate = samplerate
        # 竹笛核心频率区间：200-2000Hz
        self.f0 = 200
        self.f1 = 2000

    def moving_average_filter(self, audio, window_size=11):
        """移动平均滤波：平滑尖锐脉冲噪声"""
        audio_np = audio.cpu().numpy()
        pad_width = window_size // 2
        # 通道维度单独滤波
        for ch in range(audio_np.shape[0]):
            padded = np.pad(audio_np[ch], pad_width, mode='reflect')
            smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
            audio_np[ch] = smoothed[:audio_np.shape[1]]  # 对齐长度
        return torch.from_numpy(audio_np).to(DEVICE)

    def bandpass_filter(self, audio):
        """带通滤波：保留竹笛核心频率，滤除高频噪声"""
        audio_np = audio.cpu().numpy()
        n_fft = 2048
        hop_length = 512
        # 对每个通道做滤波
        for ch in range(audio_np.shape[0]):
            # STFT转换到频域
            stft = librosa.stft(audio_np[ch], n_fft=n_fft, hop_length=hop_length)
            freq = librosa.fft_frequencies(sr=self.samplerate, n_fft=n_fft)
            # 构建带通掩码：保留200-2000Hz
            mask = np.zeros_like(stft)
            mask[(freq >= self.f0) & (freq <= self.f1)] = 1
            # 应用掩码+逆STFT
            filtered_stft = stft * mask
            filtered = librosa.istft(filtered_stft, hop_length=hop_length, length=audio_np.shape[1])
            audio_np[ch] = filtered
        return torch.from_numpy(audio_np).to(DEVICE)

    def harmonic_reconstruction(self, audio):
        """谐波重构：恢复竹笛谐波特征，解决音色失真"""
        audio_np = audio.cpu().numpy()
        for ch in range(audio_np.shape[0]):
            # 提取基频与谐波
            y = audio_np[ch]
            f0, _, _ = librosa.pyin(y, fmin=self.f0, fmax=self.f1, sr=self.samplerate)
            # 重构谐波
            t = np.linspace(0, len(y)/self.samplerate, len(y))
            harmonic = np.zeros_like(y)
            # 填充有效基频的谐波
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                mean_f0 = np.mean(valid_f0)
                for h in range(1, 4):  # 1-3次谐波（竹笛核心）
                    harmonic += 0.3/h * np.sin(2 * np.pi * h * mean_f0 * t)
            # 融合原始信号与谐波
            audio_np[ch] = 0.7 * y + 0.3 * harmonic
        return torch.from_numpy(audio_np).to(DEVICE)

    def amplitude_normalize(self, audio):
        """幅度归一化：控制振幅在[-1,1]，避免削波失真"""
        max_amp = torch.max(torch.abs(audio))
        if max_amp > 1e-6:
            audio = audio / max_amp
        return audio

    def forward(self, audio):
        """后处理流程：滤波→谐波重构→归一化"""
        audio = self.moving_average_filter(audio)
        audio = self.bandpass_filter(audio)
        audio = self.harmonic_reconstruction(audio)
        audio = self.amplitude_normalize(audio)
        return audio

# ===================== 3. 改进版Demucs模型 =====================
class ImprovedDemucs(nn.Module):
    def __init__(self, sources=["dizi"], audio_channels=2, samplerate=44100, depth=6, kernel_size=4, stride=2):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.samplerate = samplerate
        
        # 编码器（新增BatchNorm1d层）
        self.encoders = nn.ModuleList()
        in_channels = audio_channels
        for i in range(depth):
            self.encoders.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, 64 * (2 ** i), kernel_size, stride, padding=kernel_size//2),
                    nn.BatchNorm1d(64 * (2 ** i)),  # 缓解梯度消失
                    nn.ReLU()
                )
            )
            in_channels = 64 * (2 ** i)
        
        # 解码器（新增BatchNorm1d层）
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
        输入：[batch_size, audio_channels, length]（3维）
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
                x = x[:, :, :min_len] + enc_tensor[:, :, :min_len]  # 跳过连接
            x = decoder(x)
        
        # 输出降噪
        x = self.noise_reduce(x)
        
        # 调整输出形状
        x = x.view(x.shape[0], len(self.sources), self.audio_channels, x.shape[-1])
        if x.shape[-1] > self.samplerate * SEGMENT_SEC:
            x = x[:, :, :, :self.samplerate * SEGMENT_SEC]
        
        return {self.sources[i]: x[:, i] for i in range(len(self.sources))}

# ===================== 4. SI-SDR损失函数 =====================
class SISDRLoss(nn.Module):
    """音频分离专用损失函数：SI-SDR Loss（转为损失需取负）"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # 去除均值（中心化）
        pred = pred - torch.mean(pred, dim=-1, keepdim=True)
        target = target - torch.mean(target, dim=-1, keepdim=True)
        
        # 计算SI-SDR
        dot = torch.sum(pred * target, dim=-1, keepdim=True)
        target_power = torch.sum(target ** 2, dim=-1, keepdim=True) + 1e-8
        s_target = (dot / target_power) * target
        e_noise = pred - s_target
        
        sdr = 10 * torch.log10(torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + 1e-8) + 1e-8)
        # 转为损失（越小越好）
        loss = -torch.mean(sdr)
        return loss

# ===================== 5. 数据加载与数据集类 =====================
def load_track(track_dir):
    """加载单条音频，统一格式"""
    mix_path = track_dir / "mix.wav"
    dizi_path = track_dir / "dizi.wav"
    
    mix_wav, sr = torchaudio.load(mix_path)
    dizi_wav, _ = torchaudio.load(dizi_path)
    
    # 统一采样率
    if sr != SAMPLERATE:
        resample = torchaudio.transforms.Resample(sr, SAMPLERATE)
        mix_wav = resample(mix_wav)
        dizi_wav = resample(dizi_wav)
    
    # 统一声道
    if mix_wav.shape[0] != 2:
        mix_wav = mix_wav.repeat(2, 1)
        dizi_wav = dizi_wav.repeat(2, 1)
    
    # 统一长度
    if mix_wav.shape[1] >= TARGET_LEN:
        mix_wav = mix_wav[:, :TARGET_LEN]
        dizi_wav = dizi_wav[:, :TARGET_LEN]
    else:
        pad_len = TARGET_LEN - mix_wav.shape[1]
        mix_wav = F.pad(mix_wav, (0, pad_len))
        dizi_wav = F.pad(dizi_wav, (0, pad_len))
    
    return mix_wav.to(DEVICE), dizi_wav.to(DEVICE)

class DiziDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train"):
        self.track_dirs = list((Path(root) / split).glob("track_*"))
        if len(self.track_dirs) == 0:
            raise ValueError(f"未找到{split}集的track文件夹，请检查路径：{root}/{split}")

    def __len__(self):
        return len(self.track_dirs)

    def __getitem__(self, idx):
        track_dir = self.track_dirs[idx]
        mix, dizi = load_track(track_dir)
        return mix, dizi

# ===================== 6. 训练与验证函数（核心修正） =====================
def train_one_epoch(model, dataloader, criterion, optimizer, epoch):
    """单轮训练"""
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    
    for mix, target in pbar:
        # mix.shape = [1, 2, 264600]（3维，符合Conv1d要求）
        optimizer.zero_grad()
        # 修正：去掉多余的unsqueeze(0)，直接传入mix
        pred = model(mix)  
        # 修正：pred["dizi"].shape = [1, 2, 264600]，无需取[0]
        pred_dizi = pred["dizi"]     
        # 计算损失（需匹配维度：pred_dizi [1,2,264600]，target [1,2,264600]）
        loss = criterion(pred_dizi, target)
        # 反向传播
        loss.backward()
        optimizer.step()
        # 累计损失
        total_loss += loss.item()
        pbar.set_postfix({"Train Loss": f"{loss.item():.6f}"})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate(model, dataloader, criterion, post_processor):
    """验证（含后处理+指标计算）"""
    model.eval()
    total_loss = 0.0
    total_si_sdr = 0.0
    total_l1 = 0.0
    pbar = tqdm(dataloader, desc="Validation")
    
    with torch.no_grad():
        for mix, target in pbar:
            # 修正：直接传入mix，无需unsqueeze(0)
            pred = model(mix)
            pred_dizi = pred["dizi"]  # [1,2,264600]
            # 去除batch维度后做后处理：[2,264600]
            pred_dizi_post = post_processor(pred_dizi[0])
            # 计算损失（恢复batch维度匹配target）
            loss = criterion(pred_dizi, target)
            total_loss += loss.item()
            
            # 计算L1损失（用后处理后的结果）
            l1 = torch.mean(torch.abs(pred_dizi_post - target[0])).item()
            total_l1 += l1
            
            # 计算SI-SDR（用后处理后的结果）
            pred_np = pred_dizi_post[0].cpu().numpy()
            target_np = target[0][0].cpu().numpy()
            pred_np = pred_np - np.mean(pred_np)
            target_np = target_np - np.mean(target_np)
            dot = np.sum(pred_np * target_np)
            s_target = dot * target_np / (np.sum(target_np**2) + 1e-8)
            e_noise = pred_np - s_target
            si_sdr = 10 * np.log10((np.sum(s_target**2) + 1e-8) / (np.sum(e_noise**2) + 1e-8))
            total_si_sdr += si_sdr
            
            pbar.set_postfix({"Val Loss": f"{loss.item():.6f}", "SI-SDR": f"{si_sdr:.2f}dB"})
    
    avg_loss = total_loss / len(dataloader)
    avg_l1 = total_l1 / len(dataloader)
    avg_si_sdr = total_si_sdr / len(dataloader)
    return avg_loss, avg_l1, avg_si_sdr

# ===================== 7. 主训练流程 =====================
if __name__ == "__main__":
    # 步骤1：创建数据集和数据加载器
    train_dataset = DiziDataset(DATA_ROOT, split="train")
    valid_dataset = DiziDataset(DATA_ROOT, split="valid")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    # 步骤2：初始化模型、损失、优化器
    model = ImprovedDemucs(
        sources=["dizi"],
        audio_channels=2,
        samplerate=SAMPLERATE,
        depth=6
    ).to(DEVICE)
    
    criterion = SISDRLoss()  # 音频分离专用损失
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)  # 加权重衰减防过拟合
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)  # 动态学习率
    post_processor = AudioPostProcess(samplerate=SAMPLERATE)  # 后处理器
    
    # 步骤3：早停初始化
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 步骤4：开始训练
    print("="*50)
    print("开始训练改进版Demucs模型（修复维度错误）")
    print(f"训练参数：LR={LR}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}")
    print("="*50)
    
    for epoch in range(EPOCHS):
        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        # 验证
        val_loss, val_l1, val_si_sdr = validate(model, valid_loader, criterion, post_processor)
        # 学习率调度
        scheduler.step(val_loss)
        
        # 打印本轮结果
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Val L1 Loss: {val_l1:.6f}")
        print(f"  Val SI-SDR: {val_si_sdr:.2f} dB")
        
        # 早停判断
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最优模型
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✅ 保存最优模型至：{SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"  ⚠️  验证损失未下降，耐心值：{patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("  🛑 触发早停，停止训练")
                break
    
    # 步骤5：加载最优模型，测试最终效果
    print("\n" + "="*50)
    print("训练完成！加载最优模型测试最终效果")
    print("="*50)
    
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()
    
    # 取验证集第一个样本测试（test_mix.shape = [2,264600]，需加batch维度）
    test_mix, test_target = valid_dataset[0]
    with torch.no_grad():
        pred = model(test_mix.unsqueeze(0))  # 这里需要unsqueeze(0)，因为是单条数据
        pred_dizi = pred["dizi"][0]  # 去除batch维度：[2,264600]
        pred_dizi = post_processor(pred_dizi)
    
    # 计算最终指标
    final_l1 = torch.mean(torch.abs(pred_dizi - test_target)).item()
    pred_np = pred_dizi[0].cpu().numpy()
    target_np = test_target[0].cpu().numpy()
    pred_np = pred_np - np.mean(pred_np)
    target_np = target_np - np.mean(target_np)
    dot = np.sum(pred_np * target_np)
    s_target = dot * target_np / (np.sum(target_np**2) + 1e-8)
    e_noise = pred_np - s_target
    final_si_sdr = 10 * np.log10((np.sum(s_target**2) + 1e-8) / (np.sum(e_noise**2) + 1e-8))
    
    # 打印最终结果
    print("\n📊 最优模型最终效果：")
    print(f"  L1损失（越小越好）：{final_l1:.6f}")
    print(f"  SI-SDR（越大越好）：{final_si_sdr:.2f} dB")
    
    # 生成可视化对比图
    def plot_final_result(mix, target, pred):
        plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
        plt.rcParams["axes.unicode_minus"] = False
        
        mix_np = mix[0].cpu().numpy()
        target_np = target[0].cpu().numpy()
        pred_np = pred[0].cpu().numpy()
        time_axis = np.linspace(0, len(mix_np)/SAMPLERATE, len(mix_np))
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"优化后模型效果（L1={final_l1:.4f} | SI-SDR={final_si_sdr:.2f}dB）", fontsize=16, fontweight="bold")
        
        # 波形图
        axes[0,0].plot(time_axis, mix_np, color="#2E86AB", alpha=0.8)
        axes[0,0].set_title("混合音频", fontsize=12)
        axes[0,0].set_xlabel("时间 (秒)")
        axes[0,0].set_ylabel("振幅")
        axes[0,0].set_ylim(-1, 1)
        axes[0,0].grid(alpha=0.3)
        
        axes[0,1].plot(time_axis, target_np, color="#A23B72", alpha=0.8)
        axes[0,1].set_title("真实笛子音频", fontsize=12)
        axes[0,1].set_xlabel("时间 (秒)")
        axes[0,1].set_ylabel("振幅")
        axes[0,1].set_ylim(-1, 1)
        axes[0,1].grid(alpha=0.3)
        
        axes[0,2].plot(time_axis, pred_np, color="#F18F01", alpha=0.8)
        axes[0,2].set_title("模型预测（后处理后）", fontsize=12)
        axes[0,2].set_xlabel("时间 (秒)")
        axes[0,2].set_ylabel("振幅")
        axes[0,2].set_ylim(-1, 1)
        axes[0,2].grid(alpha=0.3)
        
        # 频谱图
        n_fft = 2048
        hop_length = 512
        S_mix = librosa.amplitude_to_db(np.abs(librosa.stft(mix_np, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
        img1 = librosa.display.specshow(S_mix, sr=SAMPLERATE, hop_length=hop_length,
                                        x_axis="time", y_axis="hz", ax=axes[1,0], cmap="viridis")
        axes[1,0].set_title("混合音频频谱", fontsize=12)
        plt.colorbar(img1, ax=axes[1,0], format="%+2.0f dB")
        
        S_target = librosa.amplitude_to_db(np.abs(librosa.stft(target_np, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
        img2 = librosa.display.specshow(S_target, sr=SAMPLERATE, hop_length=hop_length,
                                        x_axis="time", y_axis="hz", ax=axes[1,1], cmap="viridis")
        axes[1,1].set_title("真实笛子频谱", fontsize=12)
        plt.colorbar(img2, ax=axes[1,1], format="%+2.0f dB")
        
        S_pred = librosa.amplitude_to_db(np.abs(librosa.stft(pred_np, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
        img3 = librosa.display.specshow(S_pred, sr=SAMPLERATE, hop_length=hop_length,
                                        x_axis="time", y_axis="hz", ax=axes[1,2], cmap="viridis")
        axes[1,2].set_title("预测笛子频谱（后处理后）", fontsize=12)
        plt.colorbar(img3, ax=axes[1,2], format="%+2.0f dB")
        
        plt.tight_layout()
        plt.savefig("dizi_optimized_result.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ 优化后效果对比图已保存：dizi_optimized_result.png")
    
    plot_final_result(test_mix, test_target, pred_dizi)