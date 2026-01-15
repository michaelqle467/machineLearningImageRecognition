# Blackjack YOLO (Realtime + On-screen decision)

## 1) Put your labeled dataset into:
- dataset/images/{train,val,test}
- dataset/labels/{train,val,test}

Classes are rank-only:
A,2,3,4,5,6,7,8,9,10,J,Q,K

## 2) Install
pip install -r requirements.txt

## 3) Train
python train.py

Weights end up at:
runs/detect/train/weights/best.pt

## 4) Webcam (shows decision on screen)
python webcam.py
Press Q to quit.

## 5) FastAPI (optional)
uvicorn api_fastapi:app --reload --port 8000
- GET /health
- POST /infer (multipart form file)
