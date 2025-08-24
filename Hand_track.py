import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Open webcam
capture = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = None, None

while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror the frame

    if canvas is None:
        canvas = np.zeros_like(frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, c = frame.shape
            index_finger_tip = hand_landmarks.landmark[8]
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            if prev_x is not None and prev_y is not None:
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), (0, 0, 255), 5)

            prev_x, prev_y = cx, cy
    else:
        prev_x, prev_y = None, None

    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow("Hand Drawing", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()
