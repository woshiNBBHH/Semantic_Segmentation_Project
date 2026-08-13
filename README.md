
# Semantic_Segmentation_Project
基于 PyTorch 的模块化语义分割实验框架。
本项目用于学习、复现和扩展经典语义分割网络，包括 FCN、UNet、PSPNet、DeepLab、DANet、CCNet 等模型。
项目采用模块化设计，将 **数据集、模型、损失函数、评价指标、实验配置、训练流程** 完全解耦，方便进行多组语义分割实验和论文模型复现。
---
# 1. 项目运行环境
## 开发环境
- 操作系统：
  - macOS
- 开发工具：
  - PyCharm
- Python环境：

Python >= 3.8

- 虚拟环境：

.venv

---
## 训练环境
本项目不在本地 MacBook 上进行训练。
训练统一使用：
- Kaggle Notebook
- Kaggle GPU Runtime
数据集通过 Kaggle 用户上传 Dataset 提供。
因此项目代码：
- 不包含本地数据路径
- 不保存训练数据
- 所有路径通过 Config 文件管理
---
# 2. 项目特点
## 模块化设计
项目按照功能划分：

Dataset
|
|
Model
|
|
Loss
|
|
Metric
|
|
Experiment

每个模块独立，方便替换和扩展。
---
## 多实验管理
每一次实验对应一个 Config 文件。
例如：

Config/

unet_modify1.py
deeplabv3_aspp.py
ccnet_rcca.py
运行后自动生成：

Experiments/

unet_modify1/
deeplabv3_aspp/
ccnet_rcca/
每个实验独立保存：
- 日志
- 模型权重
- 指标
- 预测结果
---
# 3. 项目结构

Semantic_Segmentation_Project/

├── Config/
│   ├── init.py
│   ├── unet_modify1.py
│   └── deeplabv3_aspp.py
│
├── Datasets/
│   ├── init.py
│   ├── dataset.py
│   └── transforms.py
│
├── Experiments/
│
│   └── experiment_name/
│
│       ├── config.py
│       ├── log.txt
│       ├── metrics.csv
│       ├── best.pth
│       ├── latest.pth
│       └── pred_images/
│
├── Losses/
│   ├── cross_entropy.py
│   └── dice_loss.py
│
├── Metrics/
│   ├── miou.py
│   ├── pixel_accuracy.py
│   └── dice.py
│
├── Models/
│   ├── Backbone/
│   │   ├── resnet.py
│   │   └── vgg.py
│   │
│   │
│   ├── Modules/
│   │   ├── aspp.py
│   │   ├── ppm.py
│   │   ├── cem.py
│   │   ├── pm.py
│   │   └── rcca.py
│   │
│   │
│   └── Segmentation/
│       ├── unet.py
│       ├── fcn.py
│       ├── deeplabv3.py
│       └── ccnet.py
│
├── train.py
├── test.py
├── requirements.txt
└── README.md

---
# 4. Config配置说明
所有实验参数均由 Config 文件管理。
示例：
```python
class Config:
    EXP_NAME = "unet_modify1"
    DATA_ROOT = "/kaggle/input/dataset"
    NUM_CLASSES = 21
    IMAGE_SIZE = 512
    BATCH_SIZE = 8
    EPOCHS = 100
    LR = 1e-4

Config负责保存：

* 数据路径
* 类别数量
* 输入尺寸
* Batch Size
* Epoch数量
* 学习率
* 优化器参数
* Loss选择
* 模型名称

⸻

5. Dataset模块

数据集不会存储在项目中。

训练时：

Kaggle Dataset
        |
        ↓
Datasets/dataset.py
        |
        ↓
PyTorch Dataset
        |
        ↓
DataLoader

支持：

* image读取
* mask读取
* resize
* normalization
* 数据增强

⸻

6. Models模型结构

模型目录：

Models/
├── Backbone
    主干网络
├── Modules
    通用模块
└── Segmentation
    完整语义分割网络

目前规划支持：

Backbone

* ResNet
* VGG
* MobileNet

Modules

包含：

* ASPP
* Pyramid Pooling Module(PPM)
* Context Enhancement Module(CEM)
* Position Module(PM)
* RCCA

Segmentation

包含：

* FCN
* UNet
* SegNet
* DeepLab
* CCNet

⸻

7. Loss函数

目录：

Losses/

目前支持：

Cross Entropy Loss

用于标准多分类语义分割。

Dice Loss

提高类别不平衡情况下的小目标分割能力。

未来支持：

* Focal Loss
* Boundary Loss
* Lovasz Loss

⸻

8. Evaluation Metrics

目录：

Metrics/

实现：

Pixel Accuracy

像素分类正确比例。

⸻

mIoU

Mean Intersection over Union。

计算：

[
IoU=\frac{TP}{TP+FP+FN}
]

⸻

Dice Score

用于评价预测区域和真实区域重叠程度。

⸻

9. Training

训练入口：

train.py

训练流程：

Config
 ↓
Dataset
 ↓
Model
 ↓
Loss
 ↓
Optimizer
 ↓
Training
 ↓
Validation
 ↓
Metrics
 ↓
Save Result

⸻

10. 实验结果保存

每次运行实验自动生成：

Experiments/
└── unet_modify1/
        config.py
        log.txt
        metrics.csv
        best.pth
        latest.pth
        pred_images/

说明：

best.pth

验证集指标最高模型。

latest.pth

最近一次训练状态。

用于断点续训。

⸻

11. Resume训练

支持训练中断恢复。

例如：

python train.py \
--config Config/unet_modify1.py \
--resume Experiments/unet_modify1/latest.pth

恢复：

* epoch
* model参数
* optimizer状态
* 最优指标

⸻

12. Testing

测试入口：

test.py

功能：

* 加载训练权重
* 测试数据集
* 计算评价指标
* 保存预测结果

输出：

Experiments/
    experiment_name/
        pred_images/

⸻

13. Kaggle运行方式

1. 上传项目

将：

Semantic_Segmentation_Project

上传到 Kaggle。

⸻

2. 安装依赖

pip install -r requirements.txt

⸻

3. 开始训练

python train.py \
--config Config/unet_modify1.py

⸻

4. 测试

python test.py \
--config Config/unet_modify1.py

⸻

14. Import规范

项目采用 Python Package 结构。

所有目录包含：

__init__.py

使用绝对导入：

from Models.Segmentation.unet import UNet
from Datasets.dataset import SegDataset
from Losses.dice_loss import DiceLoss

⸻

15. Future Work

计划加入：

* FCN
* SegNet
* PSPNet
* DeepLabV3+
* DANet
* CCNet
* OCRNet
* Transformer-based Segmentation

⸻

License

This project is for learning, research and academic experimentation.

这个 README 和你的项目定位一致：**不是一个单模型代码，而是一个用于语义分割研究实验的平台框架**。后续你往里面添加 PSPNet、DANet、CCNet 时，只需要更新 Future Work 和 Models 部分即可。