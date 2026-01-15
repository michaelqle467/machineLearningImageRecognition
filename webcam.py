import cv2
from ultralytics import YOLO
from blackjack_strategy_full import strategy

# ---------------- CONFIG ----------------
MODEL_PATH = r"runs/detect/train4/weights/best.pt"
CONF_THRES = 0.40

FIXED_SPLIT_Y = 0.50   # dealer (top) / player (bottom)
FIXED_SPLIT_X = 0.50   # player left / right

MAX_PLAYER_CARDS = 2
# ----------------------------------------


# ---------- UI HELPERS ----------
def draw_panel(frame, x, y, w, h, alpha=0.65):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_axis_overlay(frame, split_y, split_x):
    h, w = frame.shape[:2]
    y_px = int(split_y * h)
    x_px = int(split_x * w)

    # Axis lines
    cv2.line(frame, (0, y_px), (w, y_px), (180, 180, 180), 1)
    cv2.line(frame, (x_px, 0), (x_px, h), (180, 180, 180), 1)

    # Labels
    cv2.putText(frame, "DEALER", (10, y_px - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

    cv2.putText(frame, "PLAYER", (10, y_px + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, "LEFT", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

    cv2.putText(frame, "RIGHT", (x_px + 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)


# ---------- YOLO PARSING ----------
def parse_dets(result, model):
    dets = []
    if result.boxes is None:
        return dets

    for box in result.boxes:
        conf = float(box.conf)
        if conf < CONF_THRES:
            continue

        cls = int(box.cls)
        card = model.names[cls]

        x, y, w, h = map(float, box.xywhn[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        dets.append({
            "card": card,
            "conf": conf,
            "x": x, "y": y,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })
    return dets


def split_dealer_player(dets):
    dealer, player = [], []
    for d in dets:
        (dealer if d["y"] < FIXED_SPLIT_Y else player).append(d)

    dealer.sort(key=lambda d: d["conf"], reverse=True)
    player.sort(key=lambda d: d["conf"], reverse=True)

    dealer_up = dealer[0]["card"] if dealer else None
    return player, dealer_up


def split_player_hands(player):
    left, right = [], []
    for d in player:
        (left if d["x"] < FIXED_SPLIT_X else right).append(d)

    left.sort(key=lambda d: d["conf"], reverse=True)
    right.sort(key=lambda d: d["conf"], reverse=True)

    return (
        [d["card"] for d in left[:MAX_PLAYER_CARDS]],
        [d["card"] for d in right[:MAX_PLAYER_CARDS]],
        left, right
    )


# ---------- DRAW HUD ----------
def draw_hud(frame, dealer_up, left_cards, right_cards, active_hand, decision):
    draw_panel(frame, 10, 10, 260, 150)

    cv2.putText(frame, "BLACKJACK", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    cv2.putText(frame, f"Dealer: {dealer_up or '??'}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"L: {left_cards}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    cv2.putText(frame, f"R: {right_cards}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    cv2.putText(frame, decision, (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.putText(frame, "Q quit | N next | R reset", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)


# ---------- MAIN ----------
def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Webcam failed.")
        return

    active_hand = "right"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        draw_axis_overlay(frame, FIXED_SPLIT_Y, FIXED_SPLIT_X)

        results = model(frame, conf=CONF_THRES, verbose=False)
        dets = parse_dets(results[0], model)

        player, dealer_up = split_dealer_player(dets)
        left_cards, right_cards, left_group, right_group = split_player_hands(player)

        decision = "Waiting for cards..."
        if dealer_up:
            cards = right_cards if active_hand == "right" else left_cards
            if len(cards) >= 2:
                dec = strategy(cards[:2], dealer_up)
                decision = f"{active_hand.upper()} → {dec}"

        draw_hud(frame, dealer_up, left_cards, right_cards, active_hand, decision)

        # Draw boxes
        for d in dets:
            if d in left_group:
                color = (255, 200, 0)
            elif d in right_group:
                color = (0, 200, 255)
            else:
                color = (0, 140, 255)

            cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), color, 2)

        cv2.imshow("Blackjack YOLO", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('n'), ord('N')):
            active_hand = "left" if active_hand == "right" else "right"
        elif key in (ord('r'), ord('R')):
            active_hand = "right"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
