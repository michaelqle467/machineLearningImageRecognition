from ultralytics import YOLO
from blackjack_strategy_full import strategy

MODEL_PATH = "runs/detect/train12/weights/best.pt"
CONF_THRES = 0.15

def auto_split_y(ys):
    ys = sorted(ys)
    mid = len(ys)//2
    return ys[mid] if len(ys)%2==1 else (ys[mid-1]+ys[mid])/2.0

def main():
    model = YOLO(MODEL_PATH)
    img_path = "images/IMG_8751.jpg"  # change if needed

    results = model(img_path, conf=CONF_THRES, verbose=False)
    r = results[0]

    dets = []
    for box in r.boxes:
        if float(box.conf) < CONF_THRES:
            continue
        card = model.names[int(box.cls)]
        x,y,w,h = map(float, box.xywhn[0])
        dets.append((card, float(box.conf), y))

    if len(dets) < 3:
        print("Not enough detections (need 2 player + 1 dealer).")
        return

    split = auto_split_y([d[2] for d in dets])
    dealer = sorted([d for d in dets if d[2] < split], key=lambda t: t[1], reverse=True)
    player = sorted([d for d in dets if d[2] >= split], key=lambda t: t[1], reverse=True)

    if not dealer or len(player) < 2:
        print("Could not split dealer/player reliably.")
        return

    dealer_up = dealer[0][0]
    player_cards = [player[0][0], player[1][0]]

    print("Player:", player_cards, "Dealer:", dealer_up, "=>", strategy(player_cards, dealer_up))

if __name__ == "__main__":
    main()
