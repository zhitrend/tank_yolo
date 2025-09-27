import os
import sys

def check_directory_structure(base_dir):
    print("Checking directory structure...")
    base_dir = os.path.abspath(base_dir)  # Ensure we have absolute path
    
    # Check required directories
    required_dirs = [
        'train/images',
        'train/labels',
        'valid/images',
        'valid/labels'
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        full_path = os.path.join(base_dir, dir_path)
        if not os.path.exists(full_path):
            print("[ERROR] Directory not found: {}".format(full_path))
            all_ok = False
        else:
            print("[OK] Found directory: {}".format(full_path))
    
    return all_ok

def check_files(directory, extension):
    """Check files in directory with given extension"""
    try:
        files = [f for f in os.listdir(directory) if f.lower().endswith(extension)]
        return len(files)
    except Exception as e:
        print("Error checking %s: %s" % (directory, str(e)))
        return 0

def verify_dataset(base_dir):
    print("\nVerifying dataset...")
    
    # Check directory structure first
    if not check_directory_structure(base_dir):
        print("\nPlease fix the directory structure first.")
        return False
    
    print("\nChecking files...")
    
    # Check training set
    train_img_count = check_files(os.path.join(base_dir, 'train/images'), ('.jpg', '.jpeg', '.png'))
    train_label_count = check_files(os.path.join(base_dir, 'train/labels'), '.txt')
    
    print("\nTraining set:")
    print("  Images: {}".format(train_img_count))
    print("  Labels: {}".format(train_label_count))
    
    if train_img_count != train_label_count:
        print("  [WARNING] Number of images and labels don't match in training set!")
    
    # Check validation set
    val_img_count = check_files(os.path.join(base_dir, 'valid/images'), ('.jpg', '.jpeg', '.png'))
    val_label_count = check_files(os.path.join(base_dir, 'valid/labels'), '.txt')
    
    print("\nValidation set:")
    print("  Images: {}".format(val_img_count))
    print("  Labels: {}".format(val_label_count))
    
    if val_img_count != val_label_count:
        print("  [WARNING] Number of images and labels don't match in validation set!")
    
    # Check data.yaml
    yaml_path = os.path.join(base_dir, 'data.yaml')
    if not os.path.exists(yaml_path):
        print("\n[ERROR] data.yaml not found!")
        return False
    
    print("\n[SUCCESS] Basic verification completed.")
    print("\nIf you see any warnings or errors, please fix them before training.")
    return True

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    verify_dataset(base_dir)
