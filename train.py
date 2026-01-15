from ultralytics import YOLO

def main():
    # Small model is a good fit for ~1k images; change to yolo11s.pt if needed
    model = YOLO("yolo11n.pt")
    model.train(
        data="dataset/data.yaml",
        epochs=60,
        imgsz=640,
        batch=16,
        device="auto",
        patience=20
    )

if __name__ == "__main__":
    main()
