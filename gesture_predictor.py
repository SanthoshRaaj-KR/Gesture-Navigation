# === gesture_predictor.py ===
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model
import pickle
import time
import queue

# === Constants ===
FRAMES_PER_SEQUENCE = 45
FEATURES_PER_FRAME = 63
MODEL_PATH = "models/gesture_model.keras"
ENCODER_PATH = "models/label_encoder.pkl"

# === Load Model and Label Encoder ===
model = load_model(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# === MediaPipe Hands Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# === Prediction Function ===
def start_prediction(pred_queue: queue.Queue):
    cap = cv2.VideoCapture(0)
    sequence = deque(maxlen=FRAMES_PER_SEQUENCE)

    print("📷 Webcam started. Press 'q' to quit.")

    while cap.isOpened():
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
            sequence.append(np.zeros(FEATURES_PER_FRAME))

        display_text = "Waiting..."

        if len(sequence) == FRAMES_PER_SEQUENCE:
            input_seq = np.expand_dims(sequence, axis=0)
            prediction = model.predict(input_seq, verbose=0)[0]
            pred_index = np.argmax(prediction)
            confidence = prediction[pred_index]

            if confidence > 0.8:
                pred_label = label_encoder.inverse_transform([pred_index])[0]
                pred_queue.put(pred_label)
                display_text = f"{pred_label} ({confidence:.2f})"
            else:
                display_text = "No gesture"

            cv2.putText(image, display_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("🖐 Gesture Prediction", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    q = queue.Queue()
    start_prediction(q)
