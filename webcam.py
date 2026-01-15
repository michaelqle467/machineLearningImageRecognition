import cv2
from ultralytics import YOLO
from blackjack_strategy_full import strategy

# ---- Config ----
MODEL_PATH = "runs/detect/train/weights/best.pt"
CONF_THRES = 0.40

SPLIT_MODE = "auto"   # 'auto' uses median Y each frame; 'fixed' uses FIXED_SPLIT
FIXED_SPLIT = 0.50

MAX_PLAYER_CARDS = 2  # your basic-strategy use-case: 2 cards
SHOW_BOXES = True
SHOW_SPLIT_LINE = True
# ----------------

def auto_split_y(dets):
    ys = sorted([d["y"] for d in dets])
    if not ys:
        return None
    mid = len(ys) // 2
    return ys[mid] if len(ys) % 2 == 1 else (ys[mid - 1] + ys[mid]) / 2.0

def parse_dets(result, model, conf_thres):
    dets = []
    if result.boxes is None:
        return dets

    for box in result.boxes:
        conf = float(box.conf)
        if conf < conf_thres:
            continue

        cls = int(box.cls)
        card = model.names[cls]

        # normalized xywh (0..1)
        x, y, w, h = map(float, box.xywhn[0])

        # pixel xyxy for drawing
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        dets.append({
            "card": card,
            "conf": conf,
            "x": x, "y": y, "w": w, "h": h,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })

    return dets

def split_cards(dets, split_mode="auto", fixed_split=0.5):
    if not dets:
        return [], None, None, [], []

    split = auto_split_y(dets) if split_mode == "auto" else fixed_split

    dealer = []
    player = []
    for d in dets:
        if d["y"] < split:
            dealer.append(d)
        else:
            player.append(d)

    dealer.sort(key=lambda d: d["conf"], reverse=True)
    player.sort(key=lambda d: d["conf"], reverse=True)

    dealer_up = dealer[0]["card"] if dealer else None
    player_cards = [d["card"] for d in player[:MAX_PLAYER_CARDS]]

    return player_cards, dealer_up, split, player, dealer

def draw_overlay(frame, dets, player_cards, dealer_up, decision, split_y_norm, player_group, dealer_group):
    h, w = frame.shape[:2]

    # Split line
    if SHOW_SPLIT_LINE and split_y_norm is not None:
        y_px = int(split_y_norm * h)
        cv2.line(frame, (0, y_px), (w, y_px), (255, 255, 255), 2)

    # Boxes
    if SHOW_BOXES:
        for d in dets:
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            # Decide owner for label
            owner = "D" if d in dealer_group else "P"
            label = f"{owner}:{d['card']} {d['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Decision text
    if dealer_up and len(player_cards) >= 2:
        text = f"Player {player_cards[:2]} vs Dealer {dealer_up} -> {decision}"
    else:
        text = "Show 2 player cards + 1 dealer upcard"

    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF_THRES, verbose=False)
        dets = parse_dets(results[0], model, CONF_THRES)

        player_cards, dealer_up, split_y, player_group, dealer_group = split_cards(
            dets, split_mode=SPLIT_MODE, fixed_split=FIXED_SPLIT
        )

        decision = None
        if dealer_up and len(player_cards) >= 2:
            decision = strategy(player_cards[:2], dealer_up)

        draw_overlay(frame, dets, player_cards, dealer_up, decision, split_y, player_group, dealer_group)

        cv2.imshow("Blackjack YOLO (press Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
