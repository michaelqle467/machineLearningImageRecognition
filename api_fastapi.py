from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from ultralytics import YOLO
from blackjack_strategy_full import strategy

MODEL_PATH = "runs/detect/train/weights/best.pt"
CONF_THRES = 0.40

app = FastAPI()
model = YOLO(MODEL_PATH)

def auto_split_y(ys):
    ys = sorted(ys)
    mid = len(ys)//2
    return ys[mid] if len(ys)%2==1 else (ys[mid-1]+ys[mid])/2.0

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode image"}

    results = model(img, conf=CONF_THRES, verbose=False)
    r = results[0]

    dets = []
    for box in r.boxes:
        if float(box.conf) < CONF_THRES:
            continue
        card = model.names[int(box.cls)]
        x,y,w,h = map(float, box.xywhn[0])
        dets.append((card, float(box.conf), y))

    if len(dets) < 3:
        return {"error": "Not enough detections", "detections": dets}

    split = auto_split_y([d[2] for d in dets])
    dealer = sorted([d for d in dets if d[2] < split], key=lambda t: t[1], reverse=True)
    player = sorted([d for d in dets if d[2] >= split], key=lambda t: t[1], reverse=True)

    if not dealer or len(player) < 2:
        return {"error": "Could not split dealer/player reliably", "detections": dets, "split": split}

    dealer_up = dealer[0][0]
    player_cards = [player[0][0], player[1][0]]
    decision = strategy(player_cards, dealer_up)

    return {
        "player_cards": player_cards,
        "dealer_upcard": dealer_up,
        "decision": decision,
        "split_y": split,
        "detections": dets
    }
