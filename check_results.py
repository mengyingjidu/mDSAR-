import os
import re
import glob
import numpy as np
import cv2
from tqdm import tqdm

base_path = "./exp/image_samples"

# 💡 已经根据你截图里的文件夹名字修改好了
tasks = {
    "Cs_DDNM": "SSDD_CS_Baseline_100",
    "Cs_DPS": "SSDD_CS25_DPS",
    "Cs_Baseline": "SSDD_CS25_EQuS_Only",
    "Cs_Ours": "SSDD_CS25_EquS_Ours_Final"
}

# 💡 GT 真值图的路径 (假设存在 Ours 的 Apy 文件夹下，请确保路径正确！)
GT_DIR = os.path.join(base_path, "SSDD_CS25_EquS_Ours_Final", "Apy")

def compute_sar_enl(image_np, patch_size=16):
    """计算 SAR 图像的局部等效视数 (ENL) - 滑动窗口法"""
    img = image_np.astype(np.float32) / 255.0
    h, w = img.shape
    max_enl = 0.0
    
    step = max(1, patch_size // 2)
    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            patch = img[y:y+patch_size, x:x+patch_size]
            mean_val = np.mean(patch)
            var_val = np.var(patch)
            if var_val > 1e-5 and mean_val > 0.01:
                enl = (mean_val ** 2) / var_val
                if enl > max_enl:
                    max_enl = enl
    return max_enl

def calculate_epi(img_pred, img_gt):
    """计算舰船区域的边缘保持指数 (EPI)"""
    lap_pred = cv2.Laplacian(img_pred.astype(np.float64), cv2.CV_64F)
    lap_gt = cv2.Laplacian(img_gt.astype(np.float64), cv2.CV_64F)
    
    # 自动提取船只区域进行比对
    _, mask = cv2.threshold(img_gt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    edges_pred = np.abs(lap_pred[mask > 0])
    edges_gt = np.abs(lap_gt[mask > 0])
    
    sum_gt = np.sum(edges_gt)
    if sum_gt == 0: return 0.0
    return np.sum(edges_pred) / sum_gt

def find_best_match(gt_img, candidate_images):
    """暴力寻找最匹配的图片，防止文件名错乱"""
    best_mse = float('inf')
    best_match = None
    for cand_img in candidate_images:
        if cand_img.shape != gt_img.shape:
            temp_cand = cv2.resize(cand_img, (gt_img.shape[1], gt_img.shape[0]))
        else:
            temp_cand = cand_img
        mse = np.mean((gt_img.astype(np.float32) - temp_cand.astype(np.float32)) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_match = temp_cand
    return best_match, best_mse

def main():
    print("🚀 正在启动全自动五项全能评估 (读取 results.txt + 计算物理指标)...")
    
    # 1. 预先加载 GT 图像用于计算 EPI
    gt_files = glob.glob(os.path.join(GT_DIR, "orig_*.png"))
    gt_images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in gt_files if cv2.imread(f, cv2.IMREAD_GRAYSCALE) is not None]
    
    if not gt_images:
        print(f"❌ 警告：在 {GT_DIR} 下找不到 GT 真值图，无法计算 EPI！请检查路径。")
        return

    # --- 打印表头 ---
    print("\n" + "="*85)
    print(f"{'Task (模型)':<15} | {'PSNR ↑':<8} | {'SSIM ↑':<8} | {'LPIPS ↓':<8} | {'ENL (平滑) ↑':<12} | {'EPI (边缘) ↑':<12}")
    print("-" * 85)

    # --- 遍历 4 个任务进行评估 ---
    for task_name, folder in tasks.items():
        recon_dir = os.path.join(base_path, folder)
        file_path = os.path.join(recon_dir, "results.txt")
        
        # 1. 提取 results.txt 里的基础指标
        p_val, s_val, l_val = "N/A", "N/A", "N/A"
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            psnr = re.search(r"Total Average PSNR:\s*(\d+\.\d+)", content)
            ssim = re.search(r"Total Average SSIM:\s*(\d+\.\d+)", content)
            lpips = re.search(r"Total Average LPIPS:\s*(\d+\.\d+)", content)
            if psnr: p_val = psnr.group(1)[:5]   # 截断保留两位小数
            if ssim: s_val = ssim.group(1)[:6]
            if lpips: l_val = lpips.group(1)[:6]
        
        # 2. 读取图片并计算 ENL & EPI
        image_paths = [f for f in glob.glob(os.path.join(recon_dir, "*.png")) if os.path.isfile(f)]
        pred_images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_paths if cv2.imread(f, cv2.IMREAD_GRAYSCALE) is not None]
        
        enl_list = []
        epi_list = []
        
        if pred_images:
            # 遍历 GT 图，去生成图里找匹配项
            for gt_img in gt_images:
                best_img, best_mse = find_best_match(gt_img, pred_images)
                
                # 匹配成功 (MSE < 1000)
                if best_mse <= 1000 and best_img is not None:
                    # 算 ENL (滑动窗口)
                    enl_list.append(compute_sar_enl(best_img, patch_size=16))
                    # 算 EPI (拉普拉斯)
                    epi_list.append(calculate_epi(best_img, gt_img))
        
        # 3. 计算物理指标平均值
        avg_enl = np.mean(enl_list) if enl_list else 0.0
        avg_epi = np.mean(epi_list) if epi_list else 0.0
        
        # 4. 打印当前行数据
        print(f"{task_name:<15} | {p_val:<8} | {s_val:<8} | {l_val:<8} | {avg_enl:<12.3f} | {avg_epi:.4f}")

    print("="*85)

if __name__ == "__main__":
    main()