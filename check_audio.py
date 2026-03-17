# check_audio.py
import soundfile as sf
import os

# 测试你的音频文件路径
audio_path = "D:/cn_music/demucs/music_data/train/dizi/dizi.wav"  # 改为重命名后的文件

# 1. 检查文件是否存在
if not os.path.exists(audio_path):
    print(f"❌ 文件不存在：{audio_path}")
else:
    # 2. 尝试读取文件信息
    try:
        info = sf.info(audio_path)
        print(f"✅ 文件正常！")
        print(f"   采样率：{info.samplerate}，时长：{info.duration}秒，格式：{info.format}")
    except Exception as e:
        print(f"❌ 文件损坏/格式错误：{str(e)}")
        print("   解决方案：重新转换为标准 WAV 格式（44100Hz，16bit）")