import os
import glob

def fix_label_file(filepath):
    """Fix a single label file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
        
        if not content:
            print(f"[EMPTY] {os.path.basename(filepath)}")
            return False
            
        lines = content.split('\n')
        fixed_lines = []
        needs_fix = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 5:
                print(f"[INVALID FORMAT] {os.path.basename(filepath)}: {line}")
                needs_fix = True
                continue
                
            try:
                # Parse values
                class_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
                
                # Validate values
                if class_id < 0:
                    print(f"[NEGATIVE CLASS] {os.path.basename(filepath)}: {class_id}")
                    class_id = 0  # Fix negative class ID
                    needs_fix = True
                
                # Clamp values to valid range
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                w = max(0.0001, min(1.0, w))
                h = max(0.0001, min(1.0, h))
                
                # Format the fixed line
                fixed_line = f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                fixed_lines.append(fixed_line)
                
            except ValueError as e:
                print(f"[PARSING ERROR] {os.path.basename(filepath)}: {line} - {str(e)}")
                needs_fix = True
                continue
        
        # Write fixed content back to file if needed
        if needs_fix or len(fixed_lines) != len(lines):
            with open(filepath, 'w') as f:
                f.write('\n'.join(fixed_lines) + '\n')
            return True
            
        return False
        
    except Exception as e:
        print(f"[ERROR] {os.path.basename(filepath)}: {str(e)}")
        return False

def fix_all_labels(directory):
    """Fix all label files in a directory"""
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return 0
        
    txt_files = glob.glob(os.path.join(directory, '*.txt'))
    if not txt_files:
        print(f"No .txt files found in {directory}")
        return 0
    
    print(f"\nFixing label files in {directory}...")
    fixed_count = 0
    
    for filepath in txt_files:
        if fix_label_file(filepath):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} out of {len(txt_files)} files in {directory}")
    return fixed_count

if __name__ == "__main__":
    # Fix training labels
    train_label_dir = 'train/labels'
    val_label_dir = 'valid/labels'
    
    train_fixed = fix_all_labels(train_label_dir)
    val_fixed = fix_all_labels(val_label_dir)
    
    print("\nSummary:")
    print(f"- Fixed {train_fixed} training label files")
    print(f"- Fixed {val_fixed} validation label files")
    
    if train_fixed > 0 or val_fixed > 0:
        print("\nLabel files have been fixed. Please try training again.")
    else:
        print("\nNo issues found in label files. Please check your dataset configuration.")
