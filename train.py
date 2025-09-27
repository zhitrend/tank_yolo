from ultralytics import YOLO
import torch
import numpy as np
import os
import yaml
import glob
from PIL import Image

def validate_and_fix_dataset(data_yaml_path):
    """验证并修复数据集问题"""
    print("=== 开始数据集验证 ===")
    
    # 读取data.yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"数据集配置: {data_config}")
    
    # 检查关键字段
    required_fields = ['nc', 'names', 'train', 'val']
    for field in required_fields:
        if field not in data_config:
            raise ValueError(f"data.yaml 缺少必要字段: {field}")
    
    nc = data_config['nc']
    print(f"类别数量: {nc}")
    print(f"类别名称: {data_config['names']}")
    
    # 验证每个分割集
    splits = ['train', 'val', 'test']
    
    for split in splits:
        if split not in data_config:
            print(f"⚠️ 跳过 {split} 分割集")
            continue
            
        print(f"\n--- 验证 {split} 分割集 ---")
        
        # 获取图像和标签目录
        images_path = data_config[split]
        labels_path = images_path.replace('images', 'labels')
        
        if not os.path.exists(images_path):
            raise ValueError(f"图像目录不存在: {images_path}")
        if not os.path.exists(labels_path):
            raise ValueError(f"标签目录不存在: {labels_path}")
        
        # 检查图像文件
        image_files = glob.glob(os.path.join(images_path, "*.jpg")) + \
                     glob.glob(os.path.join(images_path, "*.png")) + \
                     glob.glob(os.path.join(images_path, "*.jpeg"))
        
        print(f"找到 {len(image_files)} 张图像")
        
        if len(image_files) < 2:
            raise ValueError(f"{split} 分割集需要至少2张图像，当前只有 {len(image_files)} 张")
        
        # 检查标签文件并修复类别索引
        fixed_count = 0
        for image_file in image_files:
            base_name = os.path.splitext(os.path.basename(image_file))[0]
            label_file = os.path.join(labels_path, base_name + ".txt")
            
            if not os.path.exists(label_file):
                print(f"⚠️ 图像缺少对应标签: {base_name}")
                continue
            
            # 检查并修复标签文件
            if fix_label_file(label_file, nc):
                fixed_count += 1
        
        print(f"修复了 {fixed_count} 个标签文件")
        
        # 验证图像可读性
        valid_images = 0
        for image_file in image_files[:5]:  # 抽样检查前5张
            try:
                with Image.open(image_file) as img:
                    img.verify()
                valid_images += 1
            except Exception as e:
                print(f"❌ 图像文件损坏: {image_file} - {e}")
        
        print(f"图像文件检查: {valid_images}/5 张有效")
    
    print("✅ 数据集验证完成")
    return True

def fix_label_file(label_file, nc):
    """修复单个标签文件中的类别索引"""
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        fixed_lines = []
        needs_fix = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 5:
                print(f"⚠️ {label_file} 第{i+1}行格式错误: {line}")
                needs_fix = True
                continue
            
            try:
                class_id = int(parts[0])
                # 检查类别索引是否有效
                if class_id < 0 or class_id >= nc:
                    print(f"❌ {label_file} 第{i+1}行: 无效类别索引 {class_id} (有效范围: 0-{nc-1})")
                    # 修复：限制在有效范围内
                    fixed_class_id = max(0, min(class_id, nc-1))
                    parts[0] = str(fixed_class_id)
                    fixed_line = " ".join(parts)
                    fixed_lines.append(fixed_line + "\n")
                    needs_fix = True
                    print(f"  修复为: {fixed_class_id}")
                else:
                    fixed_lines.append(line + "\n")
                    
            except ValueError as e:
                print(f"❌ {label_file} 第{i+1}行: 无法解析类别索引 - {e}")
                needs_fix = True
        
        # 如果需要修复，写入修正后的文件
        if needs_fix:
            with open(label_file, 'w') as f:
                f.writelines(fixed_lines)
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ 处理标签文件错误 {label_file}: {e}")
        return False

def get_device():
    if torch.backends.mps.is_available():
        return 'mps'  # Use MPS for M1/M2 Macs
    elif torch.cuda.is_available():
        return 'cuda'  # Use CUDA if available
    else:
        return 'cpu'  # Fallback to CPU

def create_safe_data_yaml(original_path, backup_path='data_safe.yaml'):
    """创建安全的data.yaml备份，确保路径正确"""
    with open(original_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # 确保使用绝对路径
    if 'path' in data:
        base_path = os.path.abspath(data['path'])
    else:
        base_path = os.path.dirname(os.path.abspath(original_path))
    
    # 更新路径为绝对路径
    for split in ['train', 'val', 'test']:
        if split in data:
            if not os.path.isabs(data[split]):
                data[split] = os.path.join(base_path, data[split])
    
    # 保存备份
    with open(backup_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"✅ 创建安全配置文件: {backup_path}")
    return backup_path

def train_yolov8s():
    # 首先验证和修复数据集
    try:
        # 创建安全的配置文件
        safe_data_yaml = create_safe_data_yaml('data.yaml')
        
        # 验证数据集
        validate_and_fix_dataset(safe_data_yaml)
        
    except Exception as e:
        print(f"❌ 数据集验证失败: {e}")
        print("请先修复数据集问题再重新训练")
        return None
    
    # Set device
    device = get_device()
    print(f"使用设备: {device}")
    
    # Load dataset configuration
    with open(safe_data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"数据集配置: {data_config}")
    
    # 简化训练配置 - 先确保能正常运行
    try:
        # 使用更小的模型开始测试
        model = YOLO('yolov8n.pt')  # 先用nano版本测试
        
        # 简化训练配置
        results = model.train(
            data=safe_data_yaml,
            epochs=50,  # 先训练少量epochs测试
            imgsz=640,
            batch=4,  # 小批量确保稳定
            device=device,
            workers=0,  # 避免多进程问题
            patience=10,
            save=True,
            exist_ok=True,
            val=True,
            plots=True,
            verbose=True,
            # 简化数据增强
            hsv_h=0.0,
            hsv_s=0.0, 
            hsv_v=0.0,
            degrees=0.0,
            flipud=0.0,
            fliplr=0.0,
            mosaic=0.0,
            mixup=0.0
        )
        
        print("✅ 训练成功完成!")
        return results
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        
        # 尝试更简化的配置
        print("尝试使用最简配置...")
        try:
            model = YOLO('yolov8n.pt')
            results = model.train(
                data=safe_data_yaml,
                epochs=10,
                imgsz=320,  # 更小的图像尺寸
                batch=2,
                device=device,
                workers=0,
                val=False,  # 暂时关闭验证
                verbose=True
            )
            return results
        except Exception as e2:
            print(f"❌ 最简配置也失败: {e2}")
            return None

if __name__ == '__main__':
    # 先运行数据检查工具
    print("=== YOLOv8 训练脚本 ===")
    
    # 可选：单独运行数据检查
    if not os.path.exists('data.yaml'):
        print("❌ 找不到 data.yaml 文件")
        print("请确保 data.yaml 文件存在于当前目录")
    else:
        results = train_yolov8s()
        if results is not None:
            print("训练完成！")
        else:
            print("训练失败，请检查上述错误信息")