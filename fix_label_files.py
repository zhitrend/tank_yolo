import os
import glob

def check_and_fix_labels(label_dir):
    """Check and fix label files in the given directory"""
    total_fixed = 0
    
    # Get all .txt files in the directory
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    
    for label_file in label_files:
        fixed_lines = []
        needs_fix = False
        
        # Read the label file
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # Check each line for issues
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 5:  # Should be: class x y w h
                print("Warning: Invalid format in {}: {}".format(os.path.basename(label_file), line))
                needs_fix = True
                continue
                
            try:
                class_id = int(float(parts[0]))  # Handle potential float strings
                x, y, w, h = map(float, parts[1:])
                
                # Check for negative class ID
                if class_id < 0:
                    print("Found negative class ID in {}: {}".format(os.path.basename(label_file), class_id))
                    class_id = 0  # Set to first class (0-indexed)
                    needs_fix = True
                
                # Check bounding box coordinates
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    print("Warning: Invalid bbox in {}: x={}, y={}, w={}, h={}".format(os.path.basename(label_file), x, y, w, h))
                    needs_fix = True
                
                fixed_lines.append("{} {} {} {} {}\n".format(class_id, x, y, w, h))
                
            except ValueError as e:
                print("Error parsing line in {}: {} - {}".format(os.path.basename(label_file), line, str(e)))
                needs_fix = True
        
        # Write fixed file if needed
        if needs_fix:
            with open(label_file, 'w') as f:
                f.writelines(fixed_lines)
            total_fixed += 1
            print("Fixed: {}".format(os.path.basename(label_file)))
    
    return total_fixed

def main():
    # Check and fix training labels
    train_label_dir = 'train/labels'
    print("Checking training labels in {}...".format(train_label_dir))
    fixed_train = check_and_fix_labels(train_label_dir)
    
    # Check and fix validation labels
    val_label_dir = 'valid/labels'
    print("\nChecking validation labels in {}...".format(val_label_dir))
    fixed_val = check_and_fix_labels(val_label_dir)
    
    print("\nSummary:")
    print("- Fixed {} training label files".format(fixed_train))
    print("- Fixed {} validation label files".format(fixed_val))
    print("\nPlease try training again after fixing the label files.")

if __name__ == "__main__":
    main()
