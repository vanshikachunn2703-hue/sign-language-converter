import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

DATA_DIR = "data"
GESTURES = ["hello", "thanks", "yes", "no", "please", 
            "sorry", "help", "good", "bad", "iloveyou"]

X, y = [], []

for label, gesture in enumerate(GESTURES):
    gesture_dir = os.path.join(DATA_DIR, gesture)
    for file in os.listdir(gesture_dir):
        data = np.load(os.path.join(gesture_dir, file))
        X.append(data)
        y.append(label)

X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"✅ Model Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")

os.makedirs("model", exist_ok=True)
with open("model/gesture_model.pkl", "wb") as f:
    pickle.dump((model, GESTURES), f)

print("✅ Model saved!")