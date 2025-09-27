import os
import torch
from ultralytics import YOLO
import yaml

def debug_training():
    # Set device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print("Using device:", device)
    
    # Load dataset configuration
    with open('data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    print("Dataset configuration:", data_config)
    
    # Load a pretrained YOLOv8s model
    print("Loading model...")
    model = YOLO('yolov8s.pt')
    
    # Print model class names
    print("Model class names:", model.names)
    
    # Enable debug mode
    os.environ['YOLO_DEBUG'] = '1'
    
    try:
        print("\nStarting training with debug mode...")
        results = model.train(
            data=os.path.abspath('data.yaml'),
            epochs=3,  # Just a few epochs for testing
            imgsz=640,
            batch=2,   # Small batch size
            device=device,
            workers=1,  # Minimal workers
            project='runs',
            name='debug_run',
            verbose=True,
            val=True,
            plots=True,
            save=False,  # Don't save checkpoints during test
            exist_ok=True
        )
        print("\nTraining completed successfully!")
        return True
        
    except Exception as e:
        print("\n[ERROR] Training failed:")
        print("Error type:", type(e).__name__)
        print("Error message:", str(e))
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_training()
