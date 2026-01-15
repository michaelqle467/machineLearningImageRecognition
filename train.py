from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")
    model.train(
        data="dataset/data.yaml",
        epochs=60,
        imgsz=640,
        batch=16,
        device="cpu",   # <-- FIX
        patience=20
    )

if __name__ == "__main__":
    main()
