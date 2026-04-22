import os
import librosa
import numpy as np
import pickle
import requests
from pymongo import MongoClient

# 🔥 USE ATLAS (IMPORTANT)
client = MongoClient("mongodb+srv://narasimha_44:leela33@cluster0.o8cg67t.mongodb.net/tunexauth?retryWrites=true&w=majority")
db = client["tunexauth"]

songs_collection = db["songs"]
songMood_collection = db["songmoods"]

# 🔥 LOAD MODEL + ENCODER
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

TEMP_FILE = "temp.mp3"


# 🔥 FEATURE EXTRACTION
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=50)

        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        spectral = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zero_cross = np.mean(librosa.feature.zero_crossing_rate(y))
        tempo = librosa.beat.tempo(y=y, sr=sr)[0]

        return np.hstack([mfcc, chroma, spectral, zero_cross, tempo])

    except Exception as e:
        print("Error:", file_path, e)
        return None


# 🔥 TAGS
def get_tags(mood):
    if mood == "chill":
        return ["smooth", "peaceful", "soothing"]
    elif mood == "gym":
        return ["fast", "energetic", "beats"]
    elif mood == "love":
        return ["romantic", "soulful"]
    elif mood == "sad":
        return ["emotional", "moody", "nostalgia"]
    elif mood == "motivation":
        return ["energetic", "uplifting"]
    elif mood == "party":
        return ["dance", "fun", "beats"]
    else:
        return []


# 🔥 PROCESS SONGS
songs = list(songs_collection.find())

print("Total songs:", len(songs))  # ✅ DEBUG

for song in songs:
    url = song.get("audioUrl")

    if not url:
        print("No URL:", song.get("title"))
        continue

    try:
        print("Processing:", song["title"])

        # DOWNLOAD
        response = requests.get(url)

        with open(TEMP_FILE, "wb") as f:
            f.write(response.content)

        # FEATURES
        features = extract_features(TEMP_FILE)

        if features is None:
            continue

        # PREDICT
        pred = model.predict([features])[0]
        mood = encoder.inverse_transform([pred])[0]

        tags = get_tags(mood)

        # SAVE
        result = songMood_collection.update_one(
            {"songId": song["_id"]},
            {
                "$set": {
                    "songId": song["_id"],
                    "mood": mood,
                    "tags": tags,
                    "score": 0.5
                }
            },
            upsert=True
        )

        print("Inserted:", result.upserted_id or "updated")
        print("mood:",mood)

    except Exception as e:
        print("Failed:", song.get("title"), e)


# CLEANUP
if os.path.exists(TEMP_FILE):
    os.remove(TEMP_FILE)

print("🎯 DONE")