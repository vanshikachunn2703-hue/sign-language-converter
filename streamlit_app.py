import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import threading
from collections import deque

# Page config
st.set_page_config(
    page_title="Sign Language Converter",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 Sign Language to Text Converter")
st.markdown("### Real-time Hand Gesture Recognition")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    with open("model/gesture_model.pkl", "rb") as f:
        model, gestures = pickle.load(f)
    return model, gestures

model, GESTURES = load_model()

# ✅ Fix 1: Mediapipe inside cached function
@st.cache_resource
def load_mediapipe():
    return mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.8)

mp_hands = load_mediapipe()
mp_draw = mp.solutions.drawing_utils
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

# ✅ Fix 2: Non-blocking TTS
def speak(text):
    def _speak():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    threading.Thread(target=_speak, daemon=True).start()

# Sidebar
st.sidebar.title("📋 Supported Gestures")
gesture_info = {
    "hello": "🖐️ Open palm",
    "thanks": "🤲 Hand from chin",
    "yes": "✊ Fist nodding",
    "no": "☝️ Finger wagging",
    "please": "🤲 Hand on chest",
    "sorry": "✊ Fist on chest",
    "help": "👍 Fist on palm",
    "good": "👍 Thumbs up",
    "bad": "👎 Thumbs down",
    "iloveyou": "🤟 Pinky+index+thumb"
}
for gesture, description in gesture_info.items():
    st.sidebar.write(f"**{gesture}**: {description}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Made by Vanshika** 👩‍💻")
st.sidebar.markdown("BTech AI & DS, 2nd Year")

# ✅ Fix 3: Session state for camera control + sentence
if "running" not in st.session_state:
    st.session_state.running = False
if "sentence" not in st.session_state:
    st.session_state.sentence = []
if "last_word" not in st.session_state:
    st.session_state.last_word = ""

# Controls
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶️ Start Camera", use_container_width=True):
        st.session_state.running = True
with col2:
    if st.button("⏹️ Stop Camera", use_container_width=True):
        st.session_state.running = False
with col3:
    # ✅ Fix 4: Clear sentence button
    if st.button("🗑️ Clear Sentence", use_container_width=True):
        st.session_state.sentence = []
        st.session_state.last_word = ""

st.markdown("---")

FRAME_WINDOW = st.image([])
gesture_display = st.empty()
sentence_display = st.empty()

# ✅ Fix 5: TTS speak sentence button
if st.button("🔊 Speak Sentence"):
    full = " ".join(st.session_state.sentence)
    if full:
        speak(full)

prediction_buffer = deque(maxlen=15)

if st.session_state.running:
    cap = cv2.VideoCapture(0)

    # ✅ Fix 6: Camera open check
    if not cap.isOpened():
        st.error("❌ Camera not found!")
        st.session_state.running = False
    else:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:  # ✅ ret check
                st.warning("⚠️ Frame read failed!")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mp_hands.process(rgb)

            word = ""

            if result.multi_hand_landmarks:
                landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, landmarks, HAND_CONNECTIONS)

                data_point = []
                for lm in landmarks.landmark:
                    data_point.extend([lm.x, lm.y])

                # ✅ Landmark validation
                if len(data_point) == 42:
                    prediction = model.predict([data_point])[0]
                    confidence = model.predict_proba([data_point])[0][prediction]
                    word = GESTURES[prediction]

                    prediction_buffer.append(word)

                    if (prediction_buffer.count(word) >= 12 and
                        confidence > 0.70 and
                        word != st.session_state.last_word):
                        st.session_state.sentence.append(word)
                        st.session_state.last_word = word
                        speak(word)  # 🔊 Non-blocking TTS

                    gesture_display.markdown(f"### 👋 Detected: **{word}** ({confidence*100:.0f}%)")
            else:
                st.session_state.last_word = ""
                gesture_display.markdown("### 👋 Show your hand to the camera!")

            full_sentence = " ".join(st.session_state.sentence[-6:])
            sentence_display.markdown(f"## 💬 Sentence: **{full_sentence}**")

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()