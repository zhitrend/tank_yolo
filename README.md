# YOLOv8 物体检测训练

本项目使用 YOLOv8s 模型进行物体检测训练。

## 数据集

数据集已准备好，包含以下类别：
- Car
- Cow
- Hamer
- Merkava_high
- Nagmash
- Person
- Tank
- Tender

## 环境配置

1. 创建并激活虚拟环境：
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate    # Windows
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 训练模型

运行以下命令开始训练：
```bash
python train.py
```

训练参数说明：
- 模型：YOLOv8s
- 输入尺寸：640x640
- Batch size: 16
- 早停耐心值：20个epoch
- 训练日志和模型权重将保存在 `runs/train/` 目录下

## 使用训练好的模型进行推理

训练完成后，可以使用以下代码进行推理：

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/train/weights/best.pt')

# 进行预测
results = model('path/to/your/image.jpg')

# 显示结果
results[0].show()
```

## 评估模型

训练完成后，可以使用以下命令评估模型性能：

```bash
python -m ultralytics.yolo val model=runs/train/weights/best.pt data=data.yaml
```

## 导出模型

导出为ONNX格式：
```bash
python -m ultralytics.export model=runs/train/weights/best.pt format=onnx
```

## 注意事项

1. 确保有足够的GPU内存进行训练
2. 训练过程中可以通过tensorboard监控训练进度：
   ```bash
   tensorboard --logdir=runs/train
   ```
3. 如果遇到显存不足的问题，可以减小batch size

## 数据集来源

本数据集来自 Roboflow: [Tank Detection Dataset](https://universe.roboflow.com/zhitrend/tank-2k5pb/dataset/1)
