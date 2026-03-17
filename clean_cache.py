# clean_cache.py
import os
import shutil

# 1. 清除Demucs输出缓存
outputs_dir = "D:/cn_music/demucs/demucs-main/outputs"
if os.path.exists(outputs_dir):
    shutil.rmtree(outputs_dir)
    print(f"✅ 已删除Demucs输出缓存：{outputs_dir}")

# 2. 清除数据集目录的缓存文件
data_dir = "D:/cn_music/demucs/music_data"
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith((".json", ".pkl", ".cache")):
            os.remove(os.path.join(root, file))
            print(f"✅ 已删除缓存文件：{os.path.join(root, file)}")

# 3. 刷新Python文件系统缓存
import importlib
import sys
for module in list(sys.modules.keys()):
    if "demucs" in module:
        importlib.reload(sys.modules[module])
print("✅ 已刷新Python模块缓存")