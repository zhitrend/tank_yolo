import os
import yaml
import cv2
import numpy as np

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
                return False, "Line %d: Expected 5 values, got %d" % (i, len(parts))
                
            try:
                class_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
                
                if class_id < 0:
                    return False, "Line %d: Negative class ID: %d" % (i, class_id)
                    
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    return False, "Line %d: Invalid bbox: x=%.4f, y=%.4f, w=%.4f, h=%.4f" % (i, x, y, w, h)
                    
                if img_shape is not None:
                    img_h, img_w = img_shape[:2]
                    x_abs = int(x * img_w)
                    y_abs = int(y * img_h)
                    w_abs = int(w * img_w)
                    h_abs = int(h * img_h)
                    
                    if not (0 <= x_abs < img_w and 0 <= y_abs < img_h):
                        return False, "Line %d: Bounding box center outside image: (%d, %d) for image size %dx%d" % (i, x_abs, y_abs, img_w, img_h)
                        
            except ValueError as e:
                return False, "Line %d: %s" % (i, str(e))
                
        return True, "OK"
        
    except Exception as e:
        return False, str(e)

def check_dataset():
    print("Checking dataset structure...")
    
    # Check data.yaml
    if not os.path.exists('data.yaml'):
        print("[ERROR] data.yaml not found")
        return False
    
    try:
        config = load_yaml('data.yaml')
        print("\nDataset configuration:")
        print("  Train:", config.get('train', 'Not specified'))
        print("  Val:", config.get('val', 'Not specified'))
        print("  Test:", config.get('test', 'Not specified'))
        print("  Classes:", config.get('nc', 'Not specified'))
        print("  Class names:", config.get('names', 'Not specified'))
    except Exception as e:
        print("[ERROR] Failed to load data.yaml:", str(e))
        return False
    
    # Check train set
    print("\nChecking training set...")
    train_img_dir = 'train/images'
    train_label_dir = 'train/labels'
    
    if not os.path.exists(train_img_dir):
        print("  [ERROR] Training images directory not found:", train_img_dir)
        return False
    if not os.path.exists(train_label_dir):
        print("  [ERROR] Training labels directory not found:", train_label_dir)
        return False
    
    # Check validation set
    print("\nChecking validation set...")
    val_img_dir = 'valid/images'
    val_label_dir = 'valid/labels'
    
    if not os.path.exists(val_img_dir):
        print("  [ERROR] Validation images directory not found:", val_img_dir)
        return False
    if not os.path.exists(val_label_dir):
        print("  [ERROR] Validation labels directory not found:", val_label_dir)
        return False
    
    # Check a few samples
    print("\nChecking samples...")
    
    # Check training sample
    train_imgs = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not train_imgs:
        print("  [ERROR] No training images found in", train_img_dir)
        return False
    
    sample_img = os.path.join(train_img_dir, train_imgs[0])
    sample_label = os.path.join(train_label_dir, os.path.splitext(train_imgs[0])[0] + '.txt')
    
    print("\nSample training image:", sample_img)
    img_ok, img_msg = check_image_file(sample_img)
    print("  Image check:", "OK" if img_ok else "ERROR: " + img_msg)
    
    print("\nSample training label:", sample_label)
    if os.path.exists(sample_label):
        label_ok, label_msg = check_label_file(sample_label)
        print("  Label check:", "OK" if label_ok else "ERROR: " + label_msg)
    else:
        print("  [ERROR] Label file not found:", sample_label)
    
    return True

if __name__ == "__main__":
    if check_dataset():
        print("\nDataset structure appears to be valid.")
    else:
        print("\nIssues were found in the dataset.")
