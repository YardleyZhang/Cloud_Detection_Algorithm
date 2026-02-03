# 云检测算法项目

## 项目概述

这是一个基于深度学习的云检测算法项目，用于自动识别和分割图像中的云层。该项目使用U-Net模型结合ResNet34编码器，实现了高精度的云分割功能，特别适用于处理复杂天气条件下的云图像。

### 主要功能

- **自动标签生成**：使用两种专业算法分别处理晴天强光晕和阴天厚云场景
- **模型训练**：基于生成的标签训练U-Net深度学习模型
- **智能推理**：支持批量处理图像，提供高精度的云分割结果
- **后处理优化**：包含连通域分析、形态学操作和边缘平滑等高级后处理功能
- **可视化结果**：生成包含原始图像、分割结果和边缘叠加的组合图像

## 目录结构

```
Cloud_Detection_Algorithm/
├── Cloud_Algorith_DL/           # 核心算法目录
│   ├── 01_generate_labels.py    # 标签生成脚本
│   ├── 02_train_unet.py         # 模型训练脚本
│   └── 03_inference.py          # 推理脚本
├── clould_pic_eg/               # 示例图像目录
├── clould_pic_eg_results_smart/ # 推理结果目录
├── could_pic_test/              # 训练测试数据目录
│   ├── cloudy_thick/            # 阴天厚云场景
│   └── sunny_glare/             # 晴天强光晕场景
├── could_pic_test_masks/        # 生成的标签目录
└── best_cloud_model.pth         # 训练好的模型文件
```

## 技术栈

- **编程语言**：Python 3.8+
- **深度学习框架**：PyTorch
- **图像处理**：OpenCV
- **模型库**：segmentation_models_pytorch
- **其他依赖**：numpy, tqdm, colorama

## 安装指南

### 1. 克隆项目

```bash
git clone <项目地址>
cd Cloud_Detection_Algorithm
```

### 2. 创建虚拟环境（可选但推荐）

```bash
# 使用conda创建虚拟环境
conda create -n cloud-detection python=3.8
conda activate cloud-detection

# 或使用pip创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install torch torchvision
pip install opencv-python numpy tqdm colorama
pip install segmentation-models-pytorch
```

## 使用说明

### 1. 生成标签

**功能**：为训练图像生成云分割标签

**使用方法**：

```bash
python Cloud_Algorith_DL/01_generate_labels.py
```

**说明**：
- 脚本会自动处理 `could_pic_test` 目录下的 `sunny_glare` 和 `cloudy_thick` 文件夹
- 对晴天强光晕场景使用 V29 抗光晕算法
- 对阴天厚云场景使用 V28 CLAHE+Otsu 暴力填缝算法
- 生成的标签会保存在 `could_pic_test_masks` 目录

### 2. 训练模型

**功能**：基于生成的标签训练U-Net模型

**使用方法**：

```bash
python Cloud_Algorith_DL/02_train_unet.py
```

**核心参数**（可在脚本中修改）：
- `IMG_SIZE`：输入图像大小（默认512）
- `BATCH_SIZE`：批量大小（默认4）
- `EPOCHS`：训练轮数（默认40）
- `LR`：学习率（默认0.0001）

**说明**：
- 模型使用ResNet34作为编码器，预加载ImageNet权重
- 使用BCE+Dice组合损失函数
- 训练过程中会自动保存最佳模型到 `best_cloud_model.pth`

### 3. 智能推理

**功能**：使用训练好的模型对新图像进行云分割

**使用方法**：

```bash
python Cloud_Algorith_DL/03_inference.py
```

**核心参数**（可在脚本中修改）：
- `CONFIDENCE_THRESHOLD`：置信度阈值（默认0.4）
- `MIN_AREA_THRESHOLD`：最小连通域面积（默认30）
- `APPLY_MORPHOLOGY`：是否应用形态学操作（默认True）
- `APPLY_EDGE_SMOOTHING`：是否应用边缘平滑（默认True）

**说明**：
- 脚本会处理 `clould_pic_eg` 目录下的所有图像
- 结果会保存在 `clould_pic_eg_results_smart` 目录
- 每个结果包含三部分：原始图像、分割结果、边缘叠加图像

## 核心算法说明

### 1. 标签生成算法

#### V29（抗光晕专家）
- **适用场景**：晴天强光晕场景
- **特点**：宁可错杀薄云，绝不放过光晕，保证太阳周围是黑的
- **核心技术**：FOV掩膜、自适应阈值、纹理计算、光晕区域处理

#### V28（CLAHE+Otsu 暴力填缝版）
- **适用场景**：阴天厚云场景
- **特点**：极其贪婪，且强制填充内部空洞，确保标签实心
- **核心技术**：CLAHE增强、Otsu自动阈值、NRBR辅助、大型闭运算

### 2. 深度学习模型

- **架构**：U-Net + ResNet34编码器
- **输入**：RGB图像（512×512）
- **输出**：二值分割掩码（云/非云）
- **损失函数**：BCEWithLogitsLoss + DiceLoss
- **优化器**：AdamW

### 3. 推理后处理

- **连通域分析**：过滤面积小于阈值的独立噪点
- **形态学操作**：使用闭合操作填充云内部的小洞
- **边缘平滑**：应用高斯模糊使云的边界更自然

## 结果解释

推理结果图像包含三个部分：

1. **左侧**：原始输入图像
2. **中间**：云分割结果（白色表示云，黑色表示非云）
3. **右侧**：边缘叠加图像（绿色边缘标记云的边界）

## 优化建议

### 1. 推理速度优化

- **批量推理**：修改 `predict_folder` 函数，支持批量加载和处理图片
- **预处理缓存**：预计算并缓存图像处理结果
- **模型量化**：使用 `torch.quantization` 进行模型量化，减少内存使用
- **半精度推理**：在推理时使用 `torch.cuda.amp.autocast()`，提高GPU利用率

### 2. 模型性能优化

- **数据增强**：添加更多的数据增强策略，如旋转、亮度调整、对比度调整
- **学习率调度**：使用 `torch.optim.lr_scheduler` 实现学习率衰减
- **模型架构**：尝试使用更深的编码器（如ResNet50）或更先进的模型架构
- **损失函数**：调整BCE和Dice损失的权重比例

### 3. 后处理优化

- **自适应阈值**：根据图像的亮度和对比度自动调整阈值
- **多尺度处理**：使用多尺度输入提高分割精度
- **边缘优化**：使用更高级的边缘检测算法，如Canny的参数自动调整

## 示例

### 输入输出示例

#### 晴天强光晕场景

**输入**：`clould_pic_eg/20251202100000_11.jpg`
**输出**：`clould_pic_eg_results_smart/Smart_20251202100000_11.jpg`

#### 阴天厚云场景

**输入**：`clould_pic_eg/20251205123000_11.jpg`
**输出**：`clould_pic_eg_results_smart/Smart_20251205123000_11.jpg`

## 常见问题

### 1. 推理速度慢

**原因**：模型在CPU上运行时可能非常缓慢
**解决方案**：
- 确保使用GPU加速（检查CUDA是否可用）
- 减小 `IMG_SIZE` 参数
- 禁用或简化后处理操作

### 2. 分割结果不准确

**原因**：可能是阈值设置不合适或模型训练不足
**解决方案**：
- 调整 `CONFIDENCE_THRESHOLD` 参数
- 调整 `MIN_AREA_THRESHOLD` 参数
- 增加模型训练轮数
- 确保训练数据包含足够的目标场景样本

### 3. 内存不足

**原因**：处理高分辨率图像时可能会占用大量内存
**解决方案**：
- 减小 `IMG_SIZE` 参数
- 减小 `BATCH_SIZE` 参数
- 确保系统有足够的RAM（推荐至少8GB）

## 系统要求

- **操作系统**：Windows、Linux或macOS
- **Python版本**：3.8或更高
- **硬件要求**：
  - CPU：至少4核
  - 内存：至少8GB
  - GPU（推荐）：支持CUDA的NVIDIA显卡，至少4GB显存

## 许可证

本项目仅供研究和学习使用。

## 联系方式

如有任何问题或建议，请联系项目维护者。

---

**项目更新时间**：2026-02-03
**版本**：1.0
