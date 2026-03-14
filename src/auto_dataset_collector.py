import cv2
import mediapipe as mp
import os
import csv
import time
import math

# -------- CONFIG --------
gesture_name = "point"     # CHANGE FOR EACH GESTURE
dataset_path = f"dataset/{gesture_name}"
os.makedirs(dataset_path, exist_ok=True)
# ------------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

window_name = "GestureX Dataset Collector"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1200, 800)

recording = False
sample_count = len(os.listdir(dataset_path))


def normalize_landmarks(hand_landmarks):

    landmarks = []

    # reference point (wrist)
    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y

    # scale reference (distance wrist → middle finger tip)
    mid_tip = hand_landmarks.landmark[12]

    scale = math.sqrt(
        (mid_tip.x - wrist_x) ** 2 +
        (mid_tip.y - wrist_y) ** 2
    )

    for lm in hand_landmarks.landmark:

        norm_x = (lm.x - wrist_x) / scale
        norm_y = (lm.y - wrist_y) / scale

        landmarks.append(norm_x)
        landmarks.append(norm_y)

    return landmarks


while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    landmarks = []

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = normalize_landmarks(hand_landmarks)

    status = "RECORDING..." if recording else "Press R to Record"

    cv2.putText(
        frame,
        f"Gesture: {gesture_name}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Samples: {sample_count}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        status,
        (20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0,0,255) if recording else (255,255,0),
        2
    )

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1)

    # Toggle recording
    if key == ord("r"):
        recording = not recording
        time.sleep(0.3)

    # Save samples automatically
    if recording and len(landmarks) == 42:

        sample_file = f"{dataset_path}/{sample_count}.csv"

        with open(sample_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(landmarks)

        sample_count += 1

    # Exit if window closed
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()