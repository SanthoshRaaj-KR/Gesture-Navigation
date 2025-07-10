# === gesture_predictor.py ===
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import pickle
import time
import queue
import sys
import os

# Add error handling for TensorFlow/Keras import
try:
    from tensorflow.keras.models import load_model
    print("[✅ SUCCESS] TensorFlow/Keras imported successfully")
except ImportError as e:
    print(f"[❌ ERROR] Failed to import TensorFlow/Keras: {e}")
    sys.exit(1)

# === Constants ===
FRAMES_PER_SEQUENCE = 45
FEATURES_PER_FRAME = 63
MODEL_PATH = "models/gesture_model.keras"
ENCODER_PATH = "models/label_encoder.pkl"

# === Global Variables ===
model = None
label_encoder = None
mp_hands = None
hands = None
mp_drawing = None

def load_model_and_encoder():
    """Load the trained model and label encoder"""
    global model, label_encoder
    
    try:
        # Check if files exist
        if not os.path.exists(MODEL_PATH):
            print(f"[❌ ERROR] Model file not found: {MODEL_PATH}")
            return False
            
        if not os.path.exists(ENCODER_PATH):
            print(f"[❌ ERROR] Label encoder file not found: {ENCODER_PATH}")
            return False
        
        # Load model
        print(f"[📦 LOADING] Loading model from: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        print("[✅ SUCCESS] Model loaded successfully")
        
        # Load label encoder
        print(f"[📦 LOADING] Loading label encoder from: {ENCODER_PATH}")
        with open(ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)
        print("[✅ SUCCESS] Label encoder loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"[❌ ERROR] Failed to load model/encoder: {e}")
        import traceback
        traceback.print_exc()
        return False

def initialize_mediapipe():
    """Initialize MediaPipe hands detection"""
    global mp_hands, hands, mp_drawing
    
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.85,
            min_tracking_confidence=0.5
        )
        mp_drawing = mp.solutions.drawing_utils
        print("[✅ SUCCESS] MediaPipe initialized successfully")
        return True
        
    except Exception as e:
        print(f"[❌ ERROR] Failed to initialize MediaPipe: {e}")
        return False

def start_prediction(pred_queue: queue.Queue):
    """Main prediction function"""
    
    # Initialize everything
    if not load_model_and_encoder():
        print("[❌ FATAL] Failed to load model/encoder")
        return
        
    if not initialize_mediapipe():
        print("[❌ FATAL] Failed to initialize MediaPipe")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[❌ ERROR] Could not open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    sequence = deque(maxlen=FRAMES_PER_SEQUENCE)
    frame_count = 0
    
    print("📷 Webcam started. Press 'q' to quit.")
    print("🤲 Show your hand gestures to control the browser!")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[⚠️ WARNING] Failed to read frame from camera")
                break

            frame_count += 1
            
            # Flip frame horizontally for mirror effect
            image = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            result = hands.process(rgb)
            
            # Extract hand landmarks
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                
                # Extract keypoints
                keypoints = np.array([
                    [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
                ]).flatten()
                
                sequence.append(keypoints)
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                
                # Add visual feedback
                cv2.putText(image, "Hand Detected", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # No hand detected, add zero padding
                sequence.append(np.zeros(FEATURES_PER_FRAME))
                cv2.putText(image, "No Hand Detected", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Default display text
            display_text = "Collecting frames..."
            color = (255, 255, 0)  # Yellow
            
            # Make prediction when we have enough frames
            if len(sequence) == FRAMES_PER_SEQUENCE:
                try:
                    # Prepare input for model
                    input_seq = np.expand_dims(sequence, axis=0)
                    
                    # Make prediction
                    prediction = model.predict(input_seq, verbose=0)[0]
                    pred_index = np.argmax(prediction)
                    confidence = prediction[pred_index]
                    
                    # Set confidence threshold
                    if confidence > 0.8:
                        
                        pred_label = label_encoder.inverse_transform([pred_index])[0]
                        
                        # Send prediction to queue
                        if not pred_queue.full():
                            pred_queue.put(pred_label)
                            
                        display_text = f"{pred_label} ({confidence:.2f})"
                        color = (0, 255, 0)  # Green
                        
                        print(f"[🎯 PREDICTION] {pred_label} (confidence: {confidence:.2f})")
                        
                    else:
                        display_text = f"Low confidence ({confidence:.2f})"
                        color = (0, 165, 255)  # Orange
                        
                except Exception as e:
                    print(f"[⚠️ PREDICTION ERROR] {e}")
                    display_text = "Prediction Error"
                    color = (0, 0, 255)  # Red

            # Display prediction text
            cv2.putText(image, display_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Display frame count
            cv2.putText(image, f"Frames: {len(sequence)}/{FRAMES_PER_SEQUENCE}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display FPS
            if frame_count % 30 == 0:
                fps = cap.get(cv2.CAP_PROP_FPS)
                cv2.putText(image, f"FPS: {fps:.1f}", (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show the frame
            cv2.imshow("🖐 Gesture Prediction", image)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[👋 QUIT] User pressed 'q' to quit")
                break
                
    except KeyboardInterrupt:
        print("\n[👋 INTERRUPT] Keyboard interrupt received")
        
    except Exception as e:
        print(f"[❌ ERROR] Unexpected error in prediction loop: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("[🧹 CLEANUP] Releasing resources...")
        cap.release()
        cv2.destroyAllWindows()
        print("[✅ SUCCESS] Camera and windows closed")

if __name__ == "__main__":
    print("[🧪 TEST MODE] Running gesture predictor in test mode")
    test_queue = queue.Queue()
    start_prediction(test_queue)
    
    # Print any predictions that were made
    predictions = []
    while not test_queue.empty():
        predictions.append(test_queue.get())
    
    if predictions:
        print(f"[📊 RESULTS] Predictions made: {predictions}")
    else:
        print("[📊 RESULTS] No predictions made")