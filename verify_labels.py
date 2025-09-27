import os
import glob

def verify_label_file(filepath):
    """Verify the contents of a single label file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
        
        if not content:
            print("[EMPTY] {}".format(os.path.basename(filepath)))
            return False
            
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue
                
            parts = line.strip().split()
            if len(parts) != 5:
                print("[INVALID FORMAT] {} - Line {}: Expected 5 values, got {}".format(os.path.basename(filepath), i, len(parts)))
                return False
                
            try:
                class_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
                
                if class_id < 0:
                    print("[NEGATIVE CLASS] {} - Line {}: Class ID is negative".format(os.path.basename(filepath), i))
                    return False
                    
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    print("[INVALID BBOX] {} - Line {}: x={}, y={}, w={}, h={}".format(os.path.basename(filepath), i, x, y, w, h))
                    return False
                    
            except ValueError as e:
                print("[PARSING ERROR] {} - Line {}: {}".format(os.path.basename(filepath), i, str(e)))
                return False
                
        return True
        
    except Exception as e:
        print("[ERROR] {}: {}".format(os.path.basename(filepath), str(e)))
        return False

def verify_all_labels(directory):
    """Verify all label files in a directory"""
    if not os.path.exists(directory):
        print("Directory not found: {}".format(directory))
        return
        
    txt_files = glob.glob(os.path.join(directory, '*.txt'))
    if not txt_files:
        print("No .txt files found in {}".format(directory))
        return
        
    print("\nVerifying {} label files in {}...".format(len(txt_files), directory))
    
    valid_count = 0
    for filepath in txt_files:
        if verify_label_file(filepath):
            valid_count += 1
    
    print("\nResults for {}:".format(directory))
    print("- Total files: {}".format(len(txt_files)))
    print("- Valid files: {}".format(valid_count))
    print("- Invalid files: {}".format(len(txt_files) - valid_count))
    
    return valid_count == len(txt_files)

if __name__ == "__main__":
    # Check training labels
    train_labels = os.path.join('train', 'labels')
    val_labels = os.path.join('valid', 'labels')
    
    train_ok = verify_all_labels(train_labels)
    val_ok = verify_all_labels(val_labels)
    
    if train_ok and val_ok:
        print("\nAll label files are valid!")
    else:
        print("\nSome label files have issues. Please check the output above.")
