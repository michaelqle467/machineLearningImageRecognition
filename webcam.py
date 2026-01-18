import cv2
from collections import Counter, deque
from ultralytics import YOLO
# from blackjack_strategy_full import strategy  # optional later

# ---------------- CONFIG ----------------
MODEL_PATH = r"runs/detect/train12/weights/best.pt"

# Start low so it "tries" on webcam. Increase with key controls.
CONF_THRES = 0.20

# Bigger helps tiny rank symbols; CPU slower. Toggle with I key.
IMGSZ = 640

# Speed: run YOLO every N frames, reuse cached detections.
FRAME_SKIP = 2

# Smoothing: card must appear this many times (in recent frames) to be "accepted"
SMOOTH_WINDOW = 8
SMOOTH_MIN_HITS = 3

# Splits (kept for UI)
FIXED_SPLIT_Y = 0.50   # dealer (top) / player (bottom)
FIXED_SPLIT_X = 0.50   # player left / right

MAX_PLAYER_CARDS = 2

# Crop (helps a LOT for webcam): toggle with C key
USE_CROP = False
CROP_Y_START = 0.35  # keep bottom 65% of frame when crop enabled
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

    cv2.line(frame, (0, y_px), (w, y_px), (180, 180, 180), 1)
    cv2.line(frame, (x_px, 0), (x_px, h), (180, 180, 180), 1)

    cv2.putText(frame, "DEALER", (10, y_px - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

    cv2.putText(frame, "PLAYER", (10, y_px + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, "LEFT", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

    cv2.putText(frame, "RIGHT", (x_px + 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)


def put_small(frame, text, x, y, color=(200, 200, 200), scale=0.45, thickness=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


# ---------- YOLO PARSING ----------
def parse_dets(result, model, conf_thres):
    dets = []
    if result.boxes is None:
        return dets

    for box in result.boxes:
        conf = float(box.conf)
        if conf < conf_thres:
            continue

        cls = int(box.cls)
        card = str(model.names[cls])

        # normalized xywh (0..1)
        x, y, w, h = map(float, box.xywhn[0])
        # pixel xyxy
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        dets.append({
            "card": card,
            "conf": conf,
            "x": x, "y": y,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })

    # highest confidence first
    dets.sort(key=lambda d: d["conf"], reverse=True)
    return dets


def split_dealer_player(dets):
    dealer, player = [], []
    for d in dets:
        (dealer if d["y"] < FIXED_SPLIT_Y else player).append(d)

    dealer.sort(key=lambda d: d["conf"], reverse=True)
    player.sort(key=lambda d: d["conf"], reverse=True)

    dealer_up = dealer[0]["card"] if dealer else None
    return player, dealer_up, dealer


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


# ---------- SMOOTHING ----------
class StableCards:
    """
    Tracks recent detected card labels and returns stable top-K.
    """
    def __init__(self, window=8, min_hits=3):
        self.window = window
        self.min_hits = min_hits
        self.buf = deque(maxlen=window)

    def update(self, labels):
        # labels: list[str] for current frame
        self.buf.append(labels)

    def stable_topk(self, k=5):
        flat = [x for frame in self.buf for x in frame]
        if not flat:
            return []
        counts = Counter(flat)
        # keep only labels with enough hits
        stable = [(lbl, c) for lbl, c in counts.items() if c >= self.min_hits]
        stable.sort(key=lambda t: t[1], reverse=True)
        return [lbl for lbl, _ in stable[:k]]

    def debug_counts(self):
        flat = [x for frame in self.buf for x in frame]
        return Counter(flat)


# ---------- DRAW HUD ----------
def draw_hud(frame, dets, dealer_up, left_cards, right_cards, active_hand,
             conf_thres, imgsz, fps, use_crop, stable_labels):
    draw_panel(frame, 10, 10, 330, 190)

    cv2.putText(frame, "CARD DETECTOR", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    put_small(frame, f"Detections: {len(dets)}", 20, 60, (255, 255, 255), scale=0.5, thickness=1)
    put_small(frame, f"Stable: {stable_labels}", 20, 80, (200, 200, 200), scale=0.45, thickness=1)

    # keep these for your layout style
    put_small(frame, f"Dealer(top): {dealer_up or '??'}", 20, 105, (0, 180, 255), scale=0.5, thickness=1)
    put_small(frame, f"L(bottom-left): {left_cards}", 20, 125, (255, 200, 0), scale=0.5, thickness=1)
    put_small(frame, f"R(bottom-right): {right_cards}", 20, 145, (0, 200, 255), scale=0.5, thickness=1)

    put_small(frame, f"Active: {active_hand.upper()}", 20, 165, (200, 200, 200), scale=0.5, thickness=1)

    # Controls / status
    put_small(frame, f"conf={conf_thres:.2f} imgsz={imgsz} crop={'ON' if use_crop else 'OFF'} fps={fps:.1f}", 20, 185, (120, 120, 120), scale=0.45, thickness=1)
    put_small(frame, "Q quit | N swap | [ / ] conf | I imgsz | C crop", 20, 205, (120, 120, 120), scale=0.45, thickness=1)


# ---------- MAIN ----------
def main():
    global CONF_THRES, IMGSZ, USE_CROP

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Webcam failed.")
        return

    # reduce camera load (helps FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    active_hand = "right"

    frame_i = 0
    cached_dets = []
    stable = StableCards(window=SMOOTH_WINDOW, min_hits=SMOOTH_MIN_HITS)

    prev_tick = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_i += 1

        # compute FPS
        tick = cv2.getTickCount()
        fps = cv2.getTickFrequency() / max(1, (tick - prev_tick))
        prev_tick = tick

        # Optional crop (helps a LOT when you only care about table area)
        crop_offset_y = 0
        frame_infer = frame
        if USE_CROP:
            H, W = frame.shape[:2]
            crop_offset_y = int(CROP_Y_START * H)
            frame_infer = frame[crop_offset_y:H, 0:W].copy()

        # Axis overlay (draw on the displayed image later)
        # Run YOLO only every FRAME_SKIP frames
        if frame_i % FRAME_SKIP == 0:
            results = model.predict(
                frame_infer,
                conf=CONF_THRES,
                iou=0.35,        # lower = more aggressive merging
                max_det=10,
                imgsz=IMGSZ,
                verbose=False
            )
            dets = parse_dets(results[0], model, CONF_THRES)
            cached_dets = dets
            # update stable labels buffer with current detections
            stable.update([d["card"] for d in dets])
        else:
            dets = cached_dets
            # still update buffer with cached labels so stability doesn't "freeze"
            stable.update([d["card"] for d in dets])

        # Now split groups (kept for your UI/box colors)
        player, dealer_up, dealer_group = split_dealer_player(dets)
        left_cards, right_cards, left_group, right_group = split_player_hands(player)

        # Show stable labels (what it "really" saw recently)
        stable_labels = stable.stable_topk(k=8)

        # Prepare display frame (if cropped, paste back for a clean UI)
        if USE_CROP:
            frame_show = frame.copy()
            H, W = frame.shape[:2]
            frame_show[crop_offset_y:H, 0:W] = frame_infer
            # crop boundary line
            cv2.line(frame_show, (0, crop_offset_y), (W, crop_offset_y), (255, 255, 255), 2)
        else:
            frame_show = frame_infer

        # Draw axis on the shown frame
        draw_axis_overlay(frame_show, FIXED_SPLIT_Y, FIXED_SPLIT_X)

        # Draw HUD
        draw_hud(frame_show, dets, dealer_up, left_cards, right_cards,
                 active_hand, CONF_THRES, IMGSZ, fps, USE_CROP, stable_labels)

        # Draw boxes (with your color scheme)
        for d in dets:
            # adjust boxes if we cropped (boxes are relative to frame_infer)
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            if USE_CROP:
                y1 += crop_offset_y
                y2 += crop_offset_y

            if d in left_group:
                color = (255, 200, 0)
            elif d in right_group:
                color = (0, 200, 255)
            else:
                color = (0, 140, 255)

            cv2.rectangle(frame_show, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_show, f"{d['card']} {d['conf']:.2f}", (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        cv2.imshow("Card Detector", frame_show)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('n'), ord('N')):
            active_hand = "left" if active_hand == "right" else "right"
        elif key in (ord('['),):
            CONF_THRES = max(0.01, CONF_THRES - 0.05)
        elif key in (ord(']'),):
            CONF_THRES = min(0.95, CONF_THRES + 0.05)
        elif key in (ord('i'), ord('I')):
            # Toggle imgsz between 640 and 960 (CPU-friendly)
            IMGSZ = 960 if IMGSZ == 640 else 640
        elif key in (ord('c'), ord('C')):
            USE_CROP = not USE_CROP

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
