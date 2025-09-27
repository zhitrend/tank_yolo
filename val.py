from ultralytics import YOLO
import os

def validate_model():
    # 加载最佳模型
    model_path = 'runs/train_simple/weights/best.pt'
    model = YOLO(model_path)
    
    # 在测试集上评估
    results = model.val(
        data='data.yaml',
        split='test',  # 使用测试集
        imgsz=640,
        batch=4,
        conf=0.25,  # 置信度阈值
        iou=0.6,    # IOU阈值
        device='cpu',
        save_json=True,  # 保存JSON格式的结果
        save_conf=True,  # 保存置信度分数
        plots=True,      # 生成评估图表
        name='val'       # 结果保存到runs/val
    )
    
    # 打印详细结果
    print("\n验证结果:")
    print(f"mAP@0.5: {results.box.map:.3f}")
    print(f"mAP@0.5:0.95: {results.box.map75:.3f}")
    
    # 打印每个类别的AP
    print("\n各类别AP@0.5:")
    for i, ap in enumerate(results.box.ap50):
        print(f"  {model.names[i]}: {ap:.3f}")

if __name__ == '__main__':
    validate_model()
