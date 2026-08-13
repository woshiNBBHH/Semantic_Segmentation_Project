# Kaggle 运行指南

本指南说明如何把 `Semantic_Segmentation_Project` 放到 Kaggle Notebook 上训练。

---

## 1. 准备 Kaggle Dataset（数据）

你的代码不保存数据，需要从 Kaggle Dataset 读取。

### 数据集要求

项目中的 `Datasets/dataset.py` 目前使用 **Pascal VOC 2012** 官方目录结构：

```
VOC2012/
├── JPEGImages/
├── SegmentationClass/
└── ImageSets/
    └── Segmentation/
        ├── train.txt
        └── val.txt
```

如果你使用其他数据集，需要修改 `Datasets/dataset.py` 或新建一个 Dataset 类。

### 上传数据集

1. 打开 [Kaggle Datasets](https://www.kaggle.com/datasets)
2. 点击 **New Dataset**
3. 上传你的 VOC2012 文件夹
4. 给数据集起个名字，例如 `pascal-voc-2012`
5. 点击 **Create**

上传后，数据集在 Kaggle Notebook 中的挂载路径为：

```
/kaggle/input/pascal-voc-2012/VOC2012
```

> 注意：路径中的 `pascal-voc-2012` 就是你创建 Dataset 时起的名字。如果你的名字不同，请修改 `Config/fcn_baseline.py` 中的 `DATA_ROOT`。

---

## 2. 准备代码（两种方式）

### 方式一：直接 git clone（推荐，最简单）

如果你的 GitHub 仓库是公开的，直接在 Kaggle Notebook 里克隆即可：

```python
!git clone https://github.com/woshiNBBHH/Semantic_Segmentation_Project.git
```

不需要把代码打包成 Kaggle Dataset。

### 方式二：打包上传为 Kaggle Dataset

如果你希望代码随 Notebook 一起保存，或者仓库是私有的，可以打包上传。

#### 打包代码

在本地项目根目录下：

```bash
cd /Users/woshizbw/Downloads/Semantic_Segmentation_Project
zip -r semantic_segmentation_code.zip . -x "*.git*" -x "Experiments/*" -x "__pycache__/*"
```

#### 上传代码数据集

1. 打开 [Kaggle Datasets](https://www.kaggle.com/datasets)
2. 点击 **New Dataset**
3. 上传 `semantic_segmentation_code.zip`
4. 数据集名称例如 `semantic-segmentation-code`
5. 点击 **Create**

---

## 3. 创建 Kaggle Notebook

1. 打开 [Kaggle Notebooks](https://www.kaggle.com/code)
2. 点击 **New Notebook**
3. 右侧 **Settings** → 开启 **GPU**（建议选 T4 x2 或 P100）
4. 右侧 **Add Data**，添加：
   - 你的代码 Dataset：`semantic-segmentation-code`
   - 你的数据 Dataset：`pascal-voc-2012`

---

## 4. Notebook 代码

### 如果你用 git clone（推荐）

在 Notebook 的第一个 Cell 中粘贴：

```python
# 克隆代码
!git clone https://github.com/woshiNBBHH/Semantic_Segmentation_Project.git

# 切换到项目目录
import os
os.chdir("/kaggle/working/Semantic_Segmentation_Project")

# 安装依赖
!pip install -q -r requirements.txt
```

### 如果你用 zip 数据集

```python
# 解压代码到工作目录
import zipfile
import os

os.makedirs("/kaggle/working/project", exist_ok=True)

with zipfile.ZipFile("/kaggle/input/semantic-segmentation-code/semantic_segmentation_code.zip", "r") as f:
    f.extractall("/kaggle/working/project")

# 切换到项目目录
os.chdir("/kaggle/working/project")

# 安装依赖
!pip install -q -r requirements.txt
```

在第二个 Cell 中训练：

```python
!python train.py --config Config.fcn_baseline
```

训练过程中，输出会保存到：

```
/kaggle/working/Semantic_Segmentation_Project/Experiments/fcn_baseline/
    ├── best.pth
    ├── latest.pth
    ├── log.txt
    └── pred_images/
```

（如果你用 zip 方式解压到了 `/kaggle/working/project/`，则对应路径为 `/kaggle/working/project/Experiments/fcn_baseline/`）

---

## 5. 断点续训

Kaggle 会话有 9~12 小时运行时长限制。如果训练被中断，可以保存 `latest.pth`，下次启动时继续训练：

```python
!python train.py \
    --config Config.fcn_baseline \
    --resume Experiments/fcn_baseline/latest.pth
```

---

## 6. 测试

训练完成后，运行测试：

```python
!python test.py \
    --config Config.fcn_baseline \
    --split val \
    --num_vis 10
```

---

## 7. 下载结果

训练结束后，在 Notebook 右侧 **Output** 标签页：

- 找到 `/kaggle/working/Semantic_Segmentation_Project/Experiments/fcn_baseline/`
- 下载 `best.pth`、`log.txt`、`pred_images/` 等文件

或者打包下载：

```python
import shutil
shutil.make_archive(
    "/kaggle/working/fcn_baseline_results",
    "zip",
    "/kaggle/working/Semantic_Segmentation_Project/Experiments/fcn_baseline"
)
```

---

## 8. 常见问题

### `ModuleNotFoundError: No module named 'Config'`

确保 Notebook 中执行了 `os.chdir("/kaggle/working/project")`，让 Python 能识别项目根目录。

### `FileNotFoundError: dataset path not found`

检查 `Config/fcn_baseline.py` 中的 `DATA_ROOT` 是否和你上传的 Kaggle Dataset 名称一致。

例如：

- Dataset 名称：`pascal-voc-2012`
- 路径应为：`/kaggle/input/pascal-voc-2012/VOC2012`

### `RuntimeError: CUDA out of memory`

在 Config 中减小 `BATCH_SIZE`，例如从 8 改为 4 或 2。

### Kaggle 下载预训练权重失败

Kaggle Notebook 默认可以访问互联网。如果失败，检查 Notebook 设置中 **Internet** 是否开启。

---

## 9. 下一步

目前项目已实现 FCN + VGG16 baseline。你可以：

1. 在 `Config/` 下新建更多实验配置
2. 在 `Models/Segmentation/` 下添加 UNet、DeepLabV3 等模型
3. 在 `Datasets/dataset.py` 中适配其他数据集
