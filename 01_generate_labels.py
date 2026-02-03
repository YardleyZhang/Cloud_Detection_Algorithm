import cv2
import numpy as np
import os
import glob
import sys
from tqdm import tqdm

# 解决控制台中文乱码
sys.stdout.reconfigure(encoding='utf-8')

# ================= 路径配置 =================
# 根目录 (代码会去这个目录下的 sunny_glare 和 cloudy_thick 找图)
# 保持了你原本的拼写 'could_pic_test'
BASE_DIR = r"C:\Users\掌义霖\Desktop\Cloud_Detection_Algorithm\could_pic_test"

# 标签输出目录 (所有生成的标签会自动汇集到这里)
OUTPUT_DIR = r"C:\Users\掌义霖\Desktop\Cloud_Detection_Algorithm\could_pic_test_masks"
# ===========================================

def compute_texture_mask(img_gray, high_thresh=15):
    """纹理计算辅助函数 (用于V29)"""
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    _, mask_texture = cv2.threshold(laplacian, high_thresh, 255, cv2.THRESH_BINARY)
    return mask_texture

# =========================================================
# 专家 A: V29 (抗光晕专家)
# 适用：sunny_glare (直射、强光晕)
# 特点：宁可错杀薄云，绝不放过光晕，保证太阳周围是黑的
# =========================================================
def algo_v29_anti_glare(img):
    height, width = img.shape[:2]
    img_radius = min(height, width) / 2
    center = (int(width/2), int(height/2))
    
    # FOV 掩膜
    mask_fov = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask_fov, center, int(img_radius * 0.90), 255, -1)
    
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask_fov)
    _, maxVal, _, sun_loc = cv2.minMaxLoc(gray_masked)

    # 1. 基础识别
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    _, mask_low_sat = cv2.threshold(s_channel, 40, 255, cv2.THRESH_BINARY_INV)
    _, mask_high_val = cv2.threshold(v_channel, 80, 255, cv2.THRESH_BINARY)
    mask_base = cv2.bitwise_and(mask_low_sat, mask_high_val)

    # 2. 自适应细节
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    s_inv = 255 - s_channel
    s_enhanced = clahe.apply(s_inv)
    mask_adaptive = cv2.adaptiveThreshold(
        s_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, int(width/8)|1, -5
    )
    mask_fusion = cv2.bitwise_or(mask_base, mask_adaptive)

    # 3. 强力抗光晕逻辑
    if maxVal > 240:
        glare_radius = int(img_radius * 0.20)
        mask_glare_zone = np.zeros_like(gray)
        cv2.circle(mask_glare_zone, sun_loc, glare_radius, 255, -1)
        
        # 计算纹理
        mask_texture = compute_texture_mask(gray, high_thresh=12)
        
        # 逻辑：光晕区内 + 无纹理 = 剔除
        mask_to_remove = cv2.bitwise_and(mask_glare_zone, cv2.bitwise_not(mask_texture))
        mask_fusion = cv2.bitwise_and(mask_fusion, cv2.bitwise_not(mask_to_remove))
        
        # 物理核心保留 (防空洞)
        cv2.circle(mask_fusion, sun_loc, 15, 255, -1)

    # 后处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_final = cv2.morphologyEx(mask_fusion, cv2.MORPH_CLOSE, kernel)
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_and(mask_final, mask_fov)

# =========================================================
# 专家 B: V28 (CLAHE + Otsu 高敏专家) - 已修改 (方案A: 暴力填缝版)
# 适用：cloudy_thick (阴天、厚云、满天云)
# 特点：极其贪婪，且强制填充内部空洞，确保标签实心
# =========================================================
def algo_v28_clahe_otsu(img):
    height, width = img.shape[:2]
    img_radius = min(height, width) / 2
    center = (int(width/2), int(height/2))
    
    # 1. FOV 掩膜
    mask_fov = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask_fov, center, int(img_radius * 0.90), 255, -1)
    
    # 2. 预处理
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:,:,1]
    
    # 3. CLAHE 增强 (V28的核心)
    # 反转S通道：云变亮，天变暗
    s_inv = 255 - s_channel
    
    # 限制对比度自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    s_enhanced = clahe.apply(s_inv)
    
    # 4. Otsu 自动阈值
    # 只在 FOV 内计算阈值，避开黑色背景干扰
    valid_pixels = s_enhanced[mask_fov > 0]
    
    if len(valid_pixels) > 0:
        thresh_val, _ = cv2.threshold(valid_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask_base = cv2.threshold(s_enhanced, thresh_val, 255, cv2.THRESH_BINARY)
    else:
        mask_base = np.zeros_like(s_channel)

    # 5. NRBR 辅助 (针对极黑雨云)
    img_float = img.astype(np.float32)
    B, R = img_float[:,:,0], img_float[:,:,2]
    denom = B + R + 1.0
    nrbr = (B - R) / denom
    mask_nrbr = np.zeros_like(s_channel)
    mask_nrbr[nrbr < 0.25] = 255 
    
    # 6. 融合
    mask_final = cv2.bitwise_or(mask_base, mask_nrbr)
    
    # ================= 修改处：暴力填缝 =================
    # 7. 闭运算 (把厚云连成一片)
    # 将核大小从 (7,7) 增加到 (35,35)，强力填补由于 Otsu 阈值导致的内部空洞
    kernel_giant = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel_giant)
    
    # 增加一次开运算，去除背景中细小的噪点
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel_small)
    # ===================================================
    
    return cv2.bitwise_and(mask_final, mask_fov)

def process_folder(subfolder_name, algo_func, algo_name):
    # 拼接完整路径
    folder_path = os.path.join(BASE_DIR, subfolder_name)
    
    if not os.path.exists(folder_path):
        print(f"⚠️ 找不到文件夹: {folder_path}")
        print(f"   请确认 {BASE_DIR} 下存在 {subfolder_name} 文件夹！")
        return

    # 寻找所有图片
    img_paths = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                glob.glob(os.path.join(folder_path, "*.png"))
    
    print(f"🚀 正在处理 [{subfolder_name}] ({len(img_paths)}张) -> 使用算法: {algo_name}")
    
    count = 0
    for path in tqdm(img_paths):
        try:
            # 读取图片
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
            if img is None: continue
            
            # 执行对应的专家算法
            mask = algo_func(img)
            
            # 保存到 OUTPUT_DIR
            name = os.path.splitext(os.path.basename(path))[0]
            save_path = os.path.join(OUTPUT_DIR, name + ".png")
            cv2.imencode('.png', mask)[1].tofile(save_path)
            count += 1
        except Exception as e:
            print(f"Error processing {path}: {e}")
    print(f"   已生成 {count} 张标签。")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 创建输出目录: {OUTPUT_DIR}")
    
    print(f"📂 根目录: {BASE_DIR}")
    print("-" * 50)
    
    # 1. 对 'sunny_glare' 文件夹使用 V29
    process_folder("sunny_glare", algo_v29_anti_glare, "V29 (抗光晕专家)")
    
    print("-" * 50)
    
    # 2. 对 'cloudy_thick' 文件夹使用 V28 (暴力填缝版)
    process_folder("cloudy_thick", algo_v28_clahe_otsu, "V28 (CLAHE+Otsu 暴力填缝版)")
    
    print("-" * 50)
    print(f"✅ 全部标签生成完毕! \n📁 请检查: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()