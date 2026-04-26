import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

GESTURES = ["hello", "thanks", "yes", "no", "please", 
            "sorry", "help", "good", "bad", "iloveyou"]

DATA_DIR = "data"
SAMPLES_PER_GESTURE = 100
os.makedirs(DATA_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

for gesture in GESTURES:
    gesture_dir = os.path.join(DATA_DIR, gesture)
    os.makedirs(gesture_dir, exist_ok=True)
    
    print(f"\n📌 Get ready for gesture: {gesture.upper()}")
    print("Press SPACE to start collecting...")
    
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f"Gesture: {gesture} | Press SPACE to start", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Collect Data", frame)
        if cv2.waitKey(1) == ord(' '):
            break
    
    count = 0
    while count < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_hands.process(rgb)
        
        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, landmarks, HAND_CONNECTIONS)
            
            data_point = []
            for lm in landmarks.landmark:
                data_point.extend([lm.x, lm.y])
            
            np.save(os.path.join(gesture_dir, f"{count}.npy"), data_point)
            count += 1
            
            cv2.putText(frame, f"Collecting: {count}/{SAMPLES_PER_GESTURE}", 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        
        cv2.imshow("Collect Data", frame)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
print("✅ Data collection complete!")