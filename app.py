import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque

with open("model/gesture_model.pkl", "rb") as f:
    model, GESTURES = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)

cap = cv2.VideoCapture(0)

sentence = []
last_word = ""
prediction_buffer = deque(maxlen=15)

print("🚀 App Running! Press Q to quit, C to clear sentence")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    word = ""

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        data_point = []
        for lm in landmarks.landmark:
            data_point.extend([lm.x, lm.y])

        prediction = model.predict([data_point])[0]
        confidence = model.predict_proba([data_point])[0][prediction]
        word = GESTURES[prediction]

        prediction_buffer.append(word)

        if (prediction_buffer.count(word) >= 12 and 
            confidence > 0.85 and 
            word != last_word):
            sentence.append(word)
            last_word = word

        cv2.putText(frame, f"Gesture: {word} ({confidence*100:.0f}%)", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        last_word = ""

    full_sentence = " ".join(sentence[-6:])
    cv2.rectangle(frame, (0, 400), (640, 480), (50, 50, 50), -1)
    cv2.putText(frame, f"Sentence: {full_sentence}", 
                (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Q = Quit | C = Clear", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Sign Language Converter", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence.clear()
        last_word = ""

cap.release()
cv2.destroyAllWindows()