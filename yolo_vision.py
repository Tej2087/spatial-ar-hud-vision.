import cv2
import math
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import os

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

# 1. Initialize YOLO
model = YOLO("yolov8m.pt")

# 2. Initialize MediaPipe
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def open_camera():
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap
            cap.release()
    return None

cap = open_camera()
if cap is None:
    raise SystemExit("Error: No camera feed accessible.")

tip_ids = [4, 8, 12, 16, 20]

# Memory States
canvas = None
prev_write_pts = {"Left": None, "Right": None}
draw_grace = {"Left": 0, "Right": 0} # 5-frame grace period for fast drawing

locked_track_id = None
locked_obj_name = None

# Targeting mechanics
pinch_hover_id = None
pinch_frames = 0
unlock_frames = 0
PINCH_THRESHOLD = 12 
is_pinch_cooling_down = False
cooldown_counter = 0

print("JARVIS V5: High-FPS Mode. 360-Degree Dual-Hand Drawing Active.")

while True:
    success, frame = cap.read()
    if not success or frame is None:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # --- 1. YOLO Object Tracking (OPTIMIZED) ---
    # imgsz=480 drastically cuts inference time on Apple Silicon
    results = model.track(frame, persist=True, verbose=False, conf=0.4, device="mps", imgsz=480)
    annotated_frame = results[0].plot(line_width=1)

    # --- 2. Hand Tracking ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hand_detector.process(rgb_frame)

    if is_pinch_cooling_down:
        cooldown_counter += 1
        if cooldown_counter > 15:
            is_pinch_cooling_down = False
            cooldown_counter = 0
            pinch_frames = 0
            unlock_frames = 0

    active_hands = set()
    pinching_hand_pos = None

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            label = handedness.classification[0].label
            active_hands.add(label)
            
            lm = hand_landmarks.landmark
            fingers = []

            idx_tip = (int(lm[8].x * w), int(lm[8].y * h))
            thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))

            # --- ANGLE-INDEPENDENT FINGER MATH ---
            # Thumb: Distance from tip to pinky base vs joint to pinky base
            dist_thumb_tip = math.dist((lm[4].x, lm[4].y), (lm[17].x, lm[17].y))
            dist_thumb_joint = math.dist((lm[2].x, lm[2].y), (lm[17].x, lm[17].y))
            fingers.append(1 if dist_thumb_tip > dist_thumb_joint else 0)

            # 4 Fingers: Distance from fingertip to wrist vs knuckle to wrist
            wrist = (lm[0].x, lm[0].y)
            for tid in tip_ids[1:]:
                dist_tip = math.dist((lm[tid].x, lm[tid].y), wrist)
                dist_pip = math.dist((lm[tid - 2].x, lm[tid - 2].y), wrist)
                fingers.append(1 if dist_tip > dist_pip else 0)

            is_pinching = math.dist(idx_tip, thumb_tip) < 40
            is_writing = fingers[1] == 1 and sum(fingers[2:]) == 0 and not is_pinching
            is_fist = sum(fingers[1:]) == 0

            # --- DUAL-HAND AIR WRITING (ANTI-LAG) ---
            if is_fist:
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                prev_write_pts["Left"] = None
                prev_write_pts["Right"] = None
                draw_grace["Left"] = 0
                draw_grace["Right"] = 0
            elif is_writing:
                draw_grace[label] = 5 # Reset grace memory when writing is detected
                if prev_write_pts[label] is not None:
                    cv2.line(canvas, prev_write_pts[label], idx_tip, (0, 165, 255), 15)
                prev_write_pts[label] = idx_tip
            else:
                # Use grace frames to connect lines even if hand moves too fast
                if draw_grace[label] > 0:
                    draw_grace[label] -= 1
                    if prev_write_pts[label] is not None:
                        cv2.line(canvas, prev_write_pts[label], idx_tip, (0, 165, 255), 15)
                    prev_write_pts[label] = idx_tip
                else:
                    prev_write_pts[label] = None

            if is_pinching and pinching_hand_pos is None:
                pinching_hand_pos = idx_tip

    for hand_key in ["Left", "Right"]:
        if hand_key not in active_hands:
            prev_write_pts[hand_key] = None

    # --- 3. TARGET LOCK & UNLOCK ---
    if pinching_hand_pos is not None and not is_pinch_cooling_down:
        px, py = pinching_hand_pos
        hovered_box_id = None
        hovered_box_name = None

        if results[0].boxes and results[0].boxes.id is not None:
            for box, track_id, cls_id in zip(results[0].boxes.xyxy, results[0].boxes.id, results[0].boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                if x1 < px < x2 and y1 < py < y2:
                    hovered_box_id = int(track_id)
                    hovered_box_name = results[0].names[int(cls_id)]
                    break

        if locked_track_id is not None:
            if hovered_box_id is not None and hovered_box_id != locked_track_id:
                unlock_frames = 0
                if pinch_hover_id == hovered_box_id:
                    pinch_frames += 1
                else:
                    pinch_hover_id = hovered_box_id
                    pinch_frames = 1

                angle = int(360 * (pinch_frames / PINCH_THRESHOLD))
                cv2.ellipse(annotated_frame, (px, py), (25, 25), 270, 0, angle, (0, 255, 0), 4)

                if pinch_frames >= PINCH_THRESHOLD:
                    locked_track_id = hovered_box_id
                    locked_obj_name = hovered_box_name
                    is_pinch_cooling_down = True
                    pinch_frames = 0
            else:
                pinch_frames = 0
                unlock_frames += 1
                angle = int(360 * (unlock_frames / PINCH_THRESHOLD))
                cv2.ellipse(annotated_frame, (px, py), (25, 25), 270, 0, angle, (0, 0, 255), 4)

                if unlock_frames >= PINCH_THRESHOLD:
                    locked_track_id = None
                    locked_obj_name = None
                    is_pinch_cooling_down = True
                    unlock_frames = 0
        else:
            unlock_frames = 0
            if hovered_box_id is not None:
                if pinch_hover_id == hovered_box_id:
                    pinch_frames += 1
                else:
                    pinch_hover_id = hovered_box_id
                    pinch_frames = 1

                angle = int(360 * (pinch_frames / PINCH_THRESHOLD))
                cv2.ellipse(annotated_frame, (px, py), (25, 25), 270, 0, angle, (0, 255, 0), 4)

                if pinch_frames >= PINCH_THRESHOLD:
                    locked_track_id = hovered_box_id
                    locked_obj_name = hovered_box_name
                    is_pinch_cooling_down = True
                    pinch_frames = 0
            else:
                pinch_frames = 0
    else:
        pinch_frames = 0
        unlock_frames = 0
        pinch_hover_id = None

    # --- Render Canvas ---
    annotated_frame = cv2.addWeighted(annotated_frame, 1, canvas, 1, 0)

    # --- Render Target Lock & PiP ---
    if locked_track_id is not None and results[0].boxes and results[0].boxes.id is not None:
        for box, track_id, cls_id in zip(results[0].boxes.xyxy, results[0].boxes.id, results[0].boxes.cls):
            if int(track_id) == locked_track_id:
                x1, y1, x2, y2 = map(int, box)

                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.75, annotated_frame, 0.25, 0, annotated_frame)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 4)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.drawMarker(annotated_frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 40, 2)
                cv2.putText(annotated_frame, f"LOCKED: {locked_obj_name.upper()}",
                            (x1, max(y1 - 15, 30)), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)

                if locked_obj_name == "person":
                    fw = x2 - x1
                    fh = y2 - y1

                    zx1 = max(0, x1 + int(fw * 0.08))
                    zx2 = min(w, x2 - int(fw * 0.08))
                    zy1 = max(0, y1)
                    zy2 = min(h, y1 + int(fh * 0.58))

                    if zx2 > zx1 and zy2 > zy1:
                        face_crop = frame[zy1:zy2, zx1:zx2]
                        pip_size = 260
                        face_resized = cv2.resize(face_crop, (pip_size, pip_size))

                        blue_tint = np.full_like(face_resized, (255, 150, 0), dtype=np.uint8)
                        face_resized = cv2.addWeighted(face_resized, 0.7, blue_tint, 0.3, 0)

                        px1, py1 = w - pip_size - 30, 30
                        px2, py2 = px1 + pip_size, py1 + pip_size

                        annotated_frame[py1:py2, px1:px2] = face_resized

                        cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (255, 200, 0), 3)
                        cv2.putText(annotated_frame, "FACIAL TRACKING UPLINK",
                                    (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

                        cv2.line(annotated_frame, (px1 + pip_size//2, py1 + pip_size//4),
                                 (px1 + pip_size//2, py2 - pip_size//4), (0, 255, 0), 1)
                        cv2.line(annotated_frame, (px1 + pip_size//4, py1 + pip_size//2),
                                 (px2 - pip_size//4, py1 + pip_size//2), (0, 255, 0), 1)
                break

    cv2.imshow("Teja's JARVIS Interface", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
