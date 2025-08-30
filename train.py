from ultralytics import YOLO

def main():
    """
    This script trains a YOLOv8 model on a custom dataset.
    """
    model = YOLO('yolov8s.pt')

    print("Starting model training...")
    results = model.train(
        data='dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        name='yolov8s_carsdd_fine_tuned'
    )

    print("\nTraining complete.")
    print("Best model weights saved to 'runs/detect/yolov8s_carsdd_fine_tuned/weights/best.pt'")

if __name__ == '__main__':
    main()
