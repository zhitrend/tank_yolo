import os
import yaml
import cv2
import numpy as np
from tqdm import tqdm

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def check_image_file(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False, "Failed to load image"
        if img.size == 0:
            return False, "Empty image"
        return True, "OK"
    except Exception as e:
        return False, str(e)

def check_label_file(label_path, img_shape=None):
    try:
        if not os.path.exists(label_path):
            return False, "File not found"
            
        with open(label_path, 'r') as f:
            content = f.read().strip()
            
        if not content:
            return False, "Empty file"
            
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 5:
                return False, f"Line {i}: Expected 5 values, got {len(parts)}"
                
            try:
                class_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
                
                if class_id < 0:
                    return False, f"Line {i}: Negative class ID: {class_id}"
                    
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    return False, f"Line {i}: Invalid bbox: x={x}, y={y}, w={w}, h={h}"
                    
                if img_shape is not None:
                    img_h, img_w = img_shape[:2]
                    x_abs = int(x * img_w)
                    y_abs = int(y * img_h)
                    w_abs = int(w * img_w)
                    h_abs = int(h * img_h)
                    
                    if not (0 <= x_abs < img_w and 0 <= y_abs < img_h):
                        return False, f"Line {i}: Bounding box center outside image: ({x_abs}, {y_abs}) for image size {img_w}x{img_h}"
                        
            except ValueError as e:
                return False, f"Line {i}: {str(e)}"
                
        return True, "OK"
        
    except Exception as e:
        return False, str(e)

def check_dataset_split(split_name, data_dir, img_dir_name='images', label_dir_name='labels'):
    print(f"\nChecking {split_name} set...")
    
    img_dir = os.path.join(data_dir, split_name, img_dir_name)
    label_dir = os.path.join(data_dir, split_name, label_dir_name)
    
    if not os.path.exists(img_dir):
        print(f"  [ERROR] Image directory not found: {img_dir}")
        return False
        
    if not os.path.exists(label_dir):
        print(f"  [ERROR] Label directory not found: {label_dir}")
        return False
    
    # Get image and label files
    img_extensions = ['.jpg', '.jpeg', '.png']
    img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in img_extensions]
    label_files = [f.replace(os.path.splitext(f)[1], '.txt') for f in img_files]
    
    print(f"  Found {len(img_files)} images and {len(os.listdir(label_dir))} label files")
    
    if len(img_files) == 0:
        print("  [ERROR] No image files found")
        return False
        
    if len(img_files) != len(os.listdir(label_dir)):
        print(f"  [WARNING] Mismatch between number of images ({len(img_files)}) and labels ({len(os.listdir(label_dir))})")
    
    # Check a sample of images and labels
    sample_size = min(10, len(img_files))
    print(f"\nChecking {sample_size} random samples:")
    
    np.random.shuffle(img_files)
    all_ok = True
    
    for img_file in img_files[:sample_size]:
        img_path = os.path.join(img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        
        # Check image
        img_ok, img_msg = check_image_file(img_path)
        if not img_ok:
            print(f"  [ERROR] {img_file}: {img_msg}")
            all_ok = False
            continue
            
        # Get image shape for label validation
        img = cv2.imread(img_path)
        img_shape = img.shape
        
        # Check label
        label_ok, label_msg = check_label_file(label_path, img_shape)
        if not label_ok:
            print(f"  [ERROR] {label_file}: {label_msg}")
            all_ok = False
            continue
            
        print(f"  [OK] {img_file} and {label_file} are valid")
    
    return all_ok

def main():
    # Load data config
    try:
        config = load_yaml('data.yaml')
        print("Dataset configuration:")
        print(f"  Train: {config.get('train', 'Not specified')}")
        print(f"  Val: {config.get('val', 'Not specified')}")
        print(f"  Test: {config.get('test', 'Not specified')}")
        print(f"  Classes: {config.get('nc', 'Not specified')}")
        print(f"  Class names: {config.get('names', 'Not specified')}")
    except Exception as e:
        print(f"[ERROR] Failed to load data.yaml: {str(e)}")
        return
    
    # Get dataset directory
    data_dir = os.path.dirname(os.path.abspath(config.get('train', '')))
    if not os.path.exists(data_dir):
        print(f"[ERROR] Dataset directory not found: {data_dir}")
        return
    
    print(f"\nDataset directory: {data_dir}")
    
    # Check train and validation sets
    train_ok = check_dataset_split('train', data_dir)
    val_ok = check_dataset_split('valid', data_dir)
    
    print("\nSummary:")
    print(f"  Train set: {'OK' if train_ok else 'Issues found'}")
    print(f"  Validation set: {'OK' if val_ok else 'Issues found'}")
    
    if train_ok and val_ok:
        print("\nDataset appears to be valid. You should be able to start training.")
    else:
        print("\nIssues were found in the dataset. Please fix them before training.")

if __name__ == "__main__":
    main()
