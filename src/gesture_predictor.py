import cv2
import mediapipe as mp
import joblib
import math
from collections import deque
from statistics import mode

# Load trained model
model = joblib.load("models/gesture_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

window_name = "GestureX - Live Gesture Recognition"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1400, 900)

# For smoothing predictions
prediction_buffer = deque(maxlen=10)


def normalize_landmarks(hand_landmarks):

    landmarks = []

    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y

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

    gesture = ""

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = normalize_landmarks(hand_landmarks)

            if len(landmarks) == 42:

                prediction = model.predict([landmarks])[0]

                prediction_buffer.append(prediction)

                # Smooth prediction
                if len(prediction_buffer) == prediction_buffer.maxlen:
                    gesture = mode(prediction_buffer)

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow(window_name, frame)

    cv2.waitKey(1)

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()