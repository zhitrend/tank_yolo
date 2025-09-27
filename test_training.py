import os
import torch
from ultralytics import YOLO

def test_training():
    # Check PyTorch and CUDA
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("MPS available:", torch.backends.mps.is_available())
    
    # Set device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print("\nUsing device: {}".format(device))
    
    # Verify data.yaml exists
    if not os.path.exists('data.yaml'):
        print("\n[ERROR] data.yaml not found!")
        return False
    
    # Load a small YOLOv8 model
    print("\nLoading YOLOv8 model...")
    try:
        model = YOLO('yolov8s.pt')
        print("Model loaded successfully!")
    except Exception as e:
        print("\n[ERROR] Failed to load model: {}".format(str(e)))
        return False
    
    # Try a very small training run
    print("\nStarting test training run...")
    try:
        results = model.train(
            data='data.yaml',
            epochs=3,  # Just 3 epochs for testing
            imgsz=640,
            batch=2,   # Small batch size
            device=device,
            workers=1,  # Minimal workers
            project='runs',
            name='test_run',
            exist_ok=True,
            verbose=True
        )
        print("\nTest training completed successfully!")
        return True
        
    except Exception as e:
        print("\n[ERROR] Training failed: {}".format(str(e)))
        print("\nError type: {}".format(type(e).__name__))
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_training()
