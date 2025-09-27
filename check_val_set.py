import os
import yaml
import numpy as np
from tqdm import tqdm

def check_validation_set():
    # Load data config
    with open('data.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    val_img_dir = 'valid/images'
    val_label_dir = 'valid/labels'
    
    print("Checking validation set...")
    print(f"Images directory: {val_img_dir}")
    print(f"Labels directory: {val_label_dir}")
    
    # Get all image files
    img_files = [f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(img_files)} validation images")
    
    # Check each image and its corresponding label
    for img_file in tqdm(img_files, desc="Checking validation samples"):
        img_path = os.path.join(val_img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(val_label_dir, label_file)
        
        # Check if label file exists
        if not os.path.exists(label_path):
            print(f"\n[WARNING] Label file not found: {label_path}")
            continue
            
        # Read label file
        with open(label_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        # Check each bounding box
        for i, line in enumerate(lines):
            try:
                parts = line.split()
                if len(parts) != 5:
                    print(f"\n[ERROR] Invalid format in {label_file}, line {i+1}: {line}")
                    continue
                    
                class_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
                
                if class_id < 0:
                    print(f"\n[ERROR] Negative class ID in {label_file}, line {i+1}: {class_id}")
                    
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    print(f"\n[ERROR] Invalid bbox in {label_file}, line {i+1}: x={x}, y={y}, w={w}, h={h}")
                    
            except Exception as e:
                print(f"\n[ERROR] Error processing {label_file}, line {i+1} ({line}): {str(e)}")
    
    print("\nValidation set check complete!")

if __name__ == "__main__":
    check_validation_set()
