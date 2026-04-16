import os
import librosa
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings("ignore")

# 🔥 FEATURE EXTRACTION
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)

        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        spectral = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zero_cross = np.mean(librosa.feature.zero_crossing_rate(y))
        tempo = librosa.beat.tempo(y=y, sr=sr)[0]

        return np.hstack([mfcc, chroma, spectral, zero_cross, tempo])

    except Exception as e:
        print("Error:", file_path, e)
        return None


# 🔥 LOAD DATA
DATA_PATH = "./data"

X = []
y = []

for mood in os.listdir(DATA_PATH):
    mood_path = os.path.join(DATA_PATH, mood)

    for file in os.listdir(mood_path):
        file_path = os.path.join(mood_path, file)

        features = extract_features(file_path)

        if features is not None:
            X.append(features)
            y.append(mood.lower())  # ✅ normalize

print("Dataset ready:", len(X))


# 🔥 ENCODE LABELS (IMPORTANT)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


# 🔥 TRAIN MODEL
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5
)

model.fit(X, y_encoded)

print("Model trained ✅")


# 🔥 SAVE MODEL + ENCODER
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("Model + Encoder saved 🎯")