import cv2
import mediapipe as mp
import joblib
import math
import pyautogui
import time
import keyboard

# Load trained gesture model
model = joblib.load("models/gesture_model.pkl")

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

window_name = "GestureX Controller"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1400, 900)

screen_w, screen_h = pyautogui.size()

# Cursor smoothing
smooth_factor = 0.3
prev_x, prev_y = 0, 0

# Gesture stability
current_gesture = ""
gesture_start_time = time.time()
stable_time_required = 2

# Cooldown
last_action_time = 0
cooldown = 1

last_action = "None"


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

    predicted_gesture = ""
    confidence = 0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = normalize_landmarks(hand_landmarks)

            if len(landmarks) == 42:

                probs = model.predict_proba([landmarks])[0]

                idx = probs.argmax()

                confidence = probs[idx]

                predicted_gesture = model.classes_[idx]

            # Cursor movement
            index_finger = hand_landmarks.landmark[8]

            x = int(index_finger.x * screen_w)
            y = int(index_finger.y * screen_h)

            curr_x = prev_x + (x - prev_x) * smooth_factor
            curr_y = prev_y + (y - prev_y) * smooth_factor

            if predicted_gesture == "point":
                pyautogui.moveTo(curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

    current_time = time.time()

    # Check gesture stability
    if predicted_gesture == current_gesture:
        gesture_duration = current_time - gesture_start_time
    else:
        current_gesture = predicted_gesture
        gesture_start_time = current_time
        gesture_duration = 0

    # Trigger actions
    if gesture_duration > stable_time_required and current_time - last_action_time > cooldown:

        if current_gesture == "fist":

            print("Click")
            pyautogui.click()

            last_action = "Click"
            last_action_time = current_time


        elif current_gesture == "peace":

            print("Double Click")
            pyautogui.doubleClick()

            last_action = "Double Click"
            last_action_time = current_time


        elif current_gesture == "thumbs_up":
            

            print("Volume Up")
            keyboard.send("volume up")

            last_action = "Volume Up"
            last_action_time = current_time


        elif current_gesture == "palm":

            print("Scroll Down")
            pyautogui.scroll(-300)

            last_action = "Scroll Down"
            last_action_time = current_time

    # UI Dashboard
    cv2.rectangle(frame,(0,0),(420,160),(40,40,40),-1)

    cv2.putText(
        frame,
        f"Gesture: {current_gesture}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {round(confidence,2)}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Last Action: {last_action}",
        (20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.imshow(window_name, frame)

    cv2.waitKey(1)

    if cv2.getWindowProperty(window_name,cv2.WND_PROP_VISIBLE)<1:
        break

cap.release()
cv2.destroyAllWindows()