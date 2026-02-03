import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import glob
import colorama
import sys

# 初始化
colorama.init()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================= 路径配置 =================
# 原图根目录 (代码会自动搜索里面的子文件夹)
TRAIN_IMG_DIR = r"C:\Users\掌义霖\Desktop\Cloud_Detection_Algorithm\could_pic_test"
# 标签目录 (上一步生成的标签都在这里)
TRAIN_MASK_DIR = r"C:\Users\掌义霖\Desktop\Cloud_Detection_Algorithm\could_pic_test_masks"
MODEL_SAVE_PATH = "best_cloud_model.pth"

IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 40
LR = 0.0001
# ===========================================

class CloudDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = []
        
        # 递归搜索所有子文件夹中的图片
        # 这样能同时找到 sunny_glare 和 cloudy_thick 里的图片
        valid_ext = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
        all_img_paths = []
        for ext in valid_ext:
            all_img_paths.extend(glob.glob(os.path.join(img_dir, '**', ext), recursive=True))

        for img_path in all_img_paths:
            # 获取文件名 (不含路径)
            filename = os.path.basename(img_path)
            basename = os.path.splitext(filename)[0]
            # 对应的标签名
            mask_name = basename + ".png"
            mask_path = os.path.join(mask_dir, mask_name)
            
            # 只有当标签存在时才加入训练
            if os.path.exists(mask_path):
                self.images.append((img_path, mask_path))
        
        print(f"📊 数据集加载: 找到 {len(self.images)} 组配对数据")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path, mask_path = self.images[index]

        # 1. 读取并强制缩放
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), 0)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        # 2. 简单增强
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        # 3. 转 Tensor (手动控制维度)
        # Image: [3, H, W]
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        image_tensor = torch.from_numpy(image)
        
        # Mask: [H, W] -> [1, H, W]
        mask = (mask > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask)
        mask_tensor = mask_tensor.view(1, IMG_SIZE, IMG_SIZE) # 暴力对齐

        return image_tensor, mask_tensor

def train():
    print(f"🚀 设备: {device}")
    
    dataset = CloudDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    
    if len(dataset) == 0:
        print("❌ 错误: 未找到数据，请检查路径或文件名！")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1,
        activation=None
    ).to(device)

    loss_fn_bce = torch.nn.BCEWithLogitsLoss()
    loss_fn_dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    best_loss = float('inf')

    print(f"🔥 开始训练 ({EPOCHS} Epochs)...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, (images, masks) in enumerate(loop):
            images = images.to(device)
            masks = masks.to(device)
            
            # 【终极保险】强制对齐维度
            outputs = model(images)
            masks = masks.reshape(outputs.shape)
            
            loss = loss_fn_bce(outputs, masks) + loss_fn_dice(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_loss = train_loss / len(loader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   💾 模型保存 (Loss: {best_loss:.4f})")
            
    print(f"\n🏆 训练完成! 最佳模型: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()