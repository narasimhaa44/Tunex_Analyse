from fastapi import FastAPI
import librosa
import numpy as np
import pickle
import requests
import io 
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

def extract_features_from_url(url):
    temp_file = "temp.mp3"

    response = requests.get(url)
    # with open(temp_file, "wb") as f:
    #     f.write(response.content)
    audio_bytes = io.BytesIO(response.content)

    y, sr = librosa.load(audio_bytes, sr=None, duration=30)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    spectral = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zero_cross = np.mean(librosa.feature.zero_crossing_rate(y))
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]

    features = np.hstack([mfcc, chroma, spectral, zero_cross, tempo])

    return features


@app.post("/predict")
def predict_mood(data: dict):
    url = data["audioUrl"]

    features = extract_features_from_url(url)

    pred = model.predict([features])[0]
    mood = encoder.inverse_transform([pred])[0]

    return {"mood": mood}