import os

# 这里写你刚重命名好的文件夹名字
folder_path = "ssdd_100" 

# 获取所有的 jpg/png 文件名
files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

# 直接覆盖官方的读取列表
with open("exp/imagenet_val_1k.txt", "w") as f:
    for file in files:
        f.write(f"{file}\n")
        
print(f"✅ 成功找到 {len(files)} 张图片，已更新到任务列表！")