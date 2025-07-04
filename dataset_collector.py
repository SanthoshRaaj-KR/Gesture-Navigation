import cv2
import os
import numpy as np
import mediapipe as mp
import time

GESTURES = ["restore_browser", "minimize_browser", "tab_left", "tab_right"]
DATA_DIR = "dataset"
SAMPLES_PER_CLASS = 50
FRAMES_PER_SEQUENCE = 45

os.makedirs(DATA_DIR, exist_ok=True)
for gesture in GESTURES:
    os.makedirs(os.path.join(DATA_DIR, gesture), exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

current_gesture = 0
sequence_count = 0

while cap.isOpened():
    sequence = []
    print(f"\nPrepare to perform gesture: {GESTURES[current_gesture]} (Sequence {sequence_count+1}/{SAMPLES_PER_CLASS})")
    time.sleep(2.5)  # give user time to prepare

    while len(sequence) < FRAMES_PER_SEQUENCE:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            sequence.append(keypoints)

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            # Use zeros if no hand detected (optional)
            sequence.append(np.zeros(63))

        cv2.putText(
            image,
            f"{GESTURES[current_gesture]} | Frame: {len(sequence)}/{FRAMES_PER_SEQUENCE}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        cv2.imshow("Recording Gesture Sequence", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save the sequence
    sequence = np.array(sequence)  # shape: (frames, 63)
    filename = f"{sequence_count}.npy"
    np.save(os.path.join(DATA_DIR, GESTURES[current_gesture], filename), sequence)

    sequence_count += 1
    if sequence_count >= SAMPLES_PER_CLASS:
        sequence_count = 0
        current_gesture += 1
        if current_gesture >= len(GESTURES):
            print("🎉 Data collection complete!")
            break

cap.release()
cv2.destroyAllWindows()
