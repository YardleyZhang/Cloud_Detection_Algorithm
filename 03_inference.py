import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import os
import glob
from tqdm import tqdm

# ================= 配置 =================
TEST_IMG_DIR = r"C:\Users\掌义霖\Desktop\Cloud_Detection_Algorithm\clould_pic_eg"
RESULT_DIR = r"C:\Users\掌义霖\Desktop\Cloud_Detection_Algorithm\clould_pic_eg_results_smart" 
MODEL_PATH = "best_cloud_model.pth"
IMG_SIZE = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 【核心参数】置信度阈值
# 调整此值以平衡召回率和精确率
# 较低的值（如0.3）会检测更多云，但可能包含更多误报
# 较高的值（如0.7）会减少误报，但可能漏检薄云
CONFIDENCE_THRESHOLD = 0.4 

# 【核心参数】最小连通域面积
# 小于这个像素数的独立白块会被当做噪点删掉
# 建议值：30 ~ 100。
MIN_AREA_THRESHOLD = 30

# 【新增参数】形态学操作
# 是否应用形态学闭合操作来填充云内部的小洞
APPLY_MORPHOLOGY = True
MORPHOLOGY_KERNEL_SIZE = 3

# 【新增参数】边缘平滑
# 是否对结果进行边缘平滑处理
APPLY_EDGE_SMOOTHING = True
SMOOTHING_KERNEL_SIZE = 3
# =======================================

def remove_small_holes_and_islands(mask, min_size):
    """
    智能过滤：只删除面积小于 min_size 的独立噪点
    """
    # 1. 寻找所有连通域
    # connectivity=8 表示 8 邻域（对角线相连也算连在一起）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # 创建一个新掩膜
    cleaned_mask = np.zeros_like(mask)
    
    # 2. 遍历所有连通域
    # 注意：label=0 是背景，从 1 开始是前景对象
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # 只有面积足够大的块才保留
        if area >= min_size:
            cleaned_mask[labels == i] = 255
            
    return cleaned_mask

def post_process_mask(mask):
    """
    高级后处理：应用形态学操作和边缘平滑
    """
    processed_mask = mask.copy()
    
    # 应用形态学闭合操作填充小洞
    if APPLY_MORPHOLOGY:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE))
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)
    
    # 应用边缘平滑
    if APPLY_EDGE_SMOOTHING:
        processed_mask = cv2.GaussianBlur(processed_mask, (SMOOTHING_KERNEL_SIZE, SMOOTHING_KERNEL_SIZE), 0)
        _, processed_mask = cv2.threshold(processed_mask, 127, 255, cv2.THRESH_BINARY)
    
    return processed_mask

def predict_folder():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型: {MODEL_PATH}")
        return
    if not os.path.exists(RESULT_DIR): os.makedirs(RESULT_DIR)

    print(f"🚀 加载模型: {MODEL_PATH}")
    model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, activation=None).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    img_paths = glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")) + \
                glob.glob(os.path.join(TEST_IMG_DIR, "*.png"))
    
    print(f"📂 开始智能推理 (过滤面积 < {MIN_AREA_THRESHOLD} 的噪点)...")

    for path in tqdm(img_paths):
        try:
            original_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
            if original_img is None: continue
            h, w = original_img.shape[:2]
            
            # 预处理
            img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
            input_tensor = torch.from_numpy(img_resized.astype('float32') / 255.0)
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            # 推理
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.sigmoid(logits)
                pred_mask_raw = (probs > CONFIDENCE_THRESHOLD).float().cpu().numpy()[0, 0]
                pred_mask_uint8 = (pred_mask_raw * 255).astype(np.uint8)

            # ================= 核心修改：智能面积过滤 =================
            # 不再使用腐蚀/开运算，改用连通域分析
            # 这能保留极细的云丝，只删除孤立的噪点
            pred_mask_cleaned = remove_small_holes_and_islands(pred_mask_uint8, min_size=MIN_AREA_THRESHOLD)
            
            # 应用高级后处理
            pred_mask_final = post_process_mask(pred_mask_cleaned)
            # =======================================================

            # 还原尺寸
            result_mask = cv2.resize(pred_mask_final, (w, h), interpolation=cv2.INTER_NEAREST)

            # 可视化
            edges = cv2.Canny(result_mask, 50, 150)
            overlay = original_img.copy()
            overlay[edges > 0] = [0, 255, 0] 
            combined = np.hstack([original_img, cv2.cvtColor(result_mask, cv2.COLOR_GRAY2BGR), overlay])
            
            name = os.path.splitext(os.path.basename(path))[0]
            save_path = os.path.join(RESULT_DIR, f"Smart_{name}.jpg")
            cv2.imencode('.jpg', combined)[1].tofile(save_path)

        except Exception as e:
            print(f"Error processing {path}: {e}")

    print(f"✅ 推理完成! 结果保存在: {RESULT_DIR}")

if __name__ == "__main__":
    predict_folder()