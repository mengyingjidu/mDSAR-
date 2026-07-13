import os
import glob

def safe_rename_0_to_99(folder_path):
    print(f"\n📂 正在处理文件夹: {folder_path}")
    
    # 获取所有的 png 文件并按当前名称字母排序
    files = sorted([f for f in glob.glob(os.path.join(folder_path, "*.png")) if os.path.isfile(f)])
    
    total_files = len(files)
    if total_files == 0:
        print("❌ 没找到任何 png 文件，跳过此文件夹！")
        return

    print(f"✅ 找到 {total_files} 张图片，开始洗牌重命名...")

    # 第一步：先全部重命名为 temp_0.png 等，防止相互覆盖
    temp_files = []
    for i, file_path in enumerate(files):
        temp_name = os.path.join(folder_path, f"temp_safe_{i}.png")
        os.rename(file_path, temp_name)
        temp_files.append(temp_name)

    # 第二步：把 temp_X.png 正式重命名为 000000.png, 000001.png...
    for i, temp_path in enumerate(temp_files):
        # 格式化为 6位数字，例如 000000.png
        final_name = os.path.join(folder_path, f"{i:06d}.png")
        os.rename(temp_path, final_name)

    print(f"🎉 搞定！已完美重命名为 000000.png 到 {total_files-1:06d}.png")

if __name__ == "__main__":
    # ================= 修改这里 =================
    # 指向包含所有 SSDD_ 文件夹的父级目录
    # 按照你原本的代码，应该就是 "./exp/image_samples"
    PARENT_DIR = "./exp/image_samples"
    # ==========================================

    if not os.path.exists(PARENT_DIR):
        print(f"🚨 找不到指定的父文件夹: {PARENT_DIR}，请检查路径！")
    else:
        # 获取父目录下的所有项
        for item_name in os.listdir(PARENT_DIR):
            folder_path = os.path.join(PARENT_DIR, item_name)
            
            # 检查该项是否为文件夹
            # 你也可以加上 `and item_name.startswith("SSDD_")` 来严格限制只处理 SSDD 开头的文件夹
            if os.path.isdir(folder_path):
                safe_rename_0_to_99(folder_path)
                
        print("\n🚀 批处理完成！所有文件夹已处理。")