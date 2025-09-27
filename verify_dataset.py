import os
import yaml
from pathlib import Path

def verify_dataset(data_yaml_path):
    # Load dataset configuration
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print("Verifying dataset configuration...")
    print("Dataset config: {}".format(data_config))
    
    # Check paths
    base_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    train_img_dir = os.path.join(base_dir, data_config['train'].lstrip('./'))
    val_img_dir = os.path.join(base_dir, data_config['val'].lstrip('./'))
    
    # Verify directories exist
    print("\nChecking directories...")
    for name, path in [('train', train_img_dir), ('val', val_img_dir)]:
        print("{} images: {} - ".format(name, path)),
        if os.path.exists(path):
            print("OK")
        else:
            print("MISSING")
    
    # Check images and labels
    print("\nChecking training set...")
    check_image_label_pairs(train_img_dir, base_dir)
    
    print("\nChecking validation set...")
    check_image_label_pairs(val_img_dir, base_dir)

def check_image_label_pairs(img_dir, base_dir):
    """Verify that each image has a corresponding label file with valid content."""
    img_exts = {'.jpg', '.jpeg', '.png'}
    
    # Get all image files
    image_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in img_exts]
    print("Found {} images in {}".format(len(image_files), img_dir))
    
    # Check corresponding label files
    missing_labels = 0
    empty_labels = 0
    invalid_labels = 0
    
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        label_file = os.path.join(os.path.dirname(img_dir).replace('images', 'labels'), "{}.txt".format(base_name))
        
        # Check if label file exists
        if not os.path.exists(label_file):
            print("  Missing label: {}".format(label_file))
            missing_labels += 1
            continue
            
        # Check if label file is empty
        if os.path.getsize(label_file) == 0:
            print("  Empty label file: {}".format(label_file))
            empty_labels += 1
            continue
            
        # Check label file content
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    print("  Invalid line in {}: {}".format(label_file, line.strip()))
                    invalid_labels += 1
                    break
                try:
                    class_id, x, y, w, h = map(float, parts)
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        print("  Invalid coordinates in {}: {}".format(label_file, line.strip()))
                        invalid_labels += 1
                        break
                except ValueError:
                    print("  Invalid number format in {}: {}".format(label_file, line.strip()))
                    invalid_labels += 1
                    break
    
    # Print summary
    if missing_labels == 0 and empty_labels == 0 and invalid_labels == 0:
        print("  All label files are valid!")
    else:
        print("  Issues found: {} missing, {} empty, {} invalid labels".format(missing_labels, empty_labels, invalid_labels))

if __name__ == "__main__":
    verify_dataset('data.yaml')
