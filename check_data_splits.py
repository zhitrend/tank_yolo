import os
import yaml

def check_data_splits():
    # Load data config
    with open('data.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Dataset configuration:")
    print(f"Train: {config.get('train')}")
    print(f"Val: {config.get('val')}")
    print(f"Test: {config.get('test')}")
    print(f"Number of classes: {config.get('nc')}")
    print(f"Class names: {config.get('names')}")
    
    # Check training set
    print("\nChecking training set...")
    train_img_dir = 'train/images'
    train_label_dir = 'train/labels'
    
    if not os.path.exists(train_img_dir):
        print(f"[ERROR] Training images directory not found: {train_img_dir}")
    else:
        train_imgs = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(train_imgs)} training images")
        
        # Check a sample of training labels
        sample_count = min(3, len(train_imgs))
        print(f"\nChecking {sample_count} training samples:")
        for img_file in train_imgs[:sample_count]:
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(train_label_dir, label_file)
            
            if not os.path.exists(label_path):
                print(f"  [ERROR] Label file not found: {label_path}")
            else:
                with open(label_path, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    print(f"  {label_file}: {len(lines)} objects")
    
    # Check validation set
    print("\nChecking validation set...")
    val_img_dir = 'valid/images'
    val_label_dir = 'valid/labels'
    
    if not os.path.exists(val_img_dir):
        print(f"[ERROR] Validation images directory not found: {val_img_dir}")
    else:
        val_imgs = [f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(val_imgs)} validation images")
        
        if val_imgs:
            print(f"\nChecking validation labels:")
            for img_file in val_imgs:
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(val_label_dir, label_file)
                
                if not os.path.exists(label_path):
                    print(f"  [ERROR] Label file not found: {label_path}")
                else:
                    with open(label_path, 'r') as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                        print(f"  {label_file}: {len(lines)} objects")
                            
                            # Check for negative class IDs
                            for i, line in enumerate(lines):
                                try:
                                    class_id = int(float(line.split()[0]))
                                    if class_id < 0:
                                        print(f"    [ERROR] Negative class ID in {label_file}, line {i+1}: {class_id}")
                                except (ValueError, IndexError) as e:
                                    print(f"    [ERROR] Error parsing {label_file}, line {i+1}: {line}")
    
    print("\nData split check complete!")

if __name__ == "__main__":
    check_data_splits()
