import os
import sys
import torch
from ultralytics import YOLO

def main():
    # Set device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = YOLO('yolov8s.pt')
    
    # Print model info
    print("\nModel info:")
    print(f"Number of classes: {len(model.names)}")
    print(f"Class names: {model.names}")
    
    # Training configuration
    print("\nStarting training...")
    try:
        results = model.train(
            data=os.path.abspath('data.yaml'),
            epochs=5,  # Just a few epochs for testing
            imgsz=640,
            batch=2,  # Small batch size to avoid memory issues
            device=device,
            workers=1,  # Minimal workers for stability
            project='runs',
            name='test_run',
            verbose=True,
            val=True,
            plots=True,
            save=False,  # Don't save checkpoints during test
            exist_ok=True
        )
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nError type:", type(e).__name__)
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
