# ============================================================
# FILE 1 : generate_emotion_v3.py
# PURPOSE: Phase-5 GOLD CLEANER — Reduce Frustrated Noise
# STRATEGY: Sharper Frustrated rules, calmer default
# ============================================================

import pandas as pd

INPUT_PATH  = r"D:/model_bert_copy/data/gold/cx_phase4/val.csv"
OUTPUT_PATH = r"D:/model_bert_copy/data/gold/cx_phase5/val_with_emotion_v3.csv"

VERY_ANGRY_KW = ["fraud","cheated","scam","court","legal","harassment","refund fraud","money stolen"]
ANGRY_KW = ["angry","furious","complaint","bad service","poor service","rude","refund pending","lost","damaged","wrong delivery"]
FRUSTRATED_KW = ["waiting","delay","pending","no update","follow up","concern","confused","still waiting"]
SATISFIED_KW = ["good","great","excellent","happy","satisfied","thanks","resolved","well done"]


def assign_emotion(text, sentiment, aspect, aspect_sentiment):
    t = text.lower()

    if any(k in t for k in VERY_ANGRY_KW):
        return 4

    if sentiment == 0 and aspect in [2,7] and aspect_sentiment == 0:
        return 4

    if any(k in t for k in ANGRY_KW):
        return 3

    if sentiment == 0 and aspect_sentiment == 0:
        return 3

    if any(k in t for k in FRUSTRATED_KW) and sentiment != 2:
        return 2

    if any(k in t for k in SATISFIED_KW) or (sentiment == 2 and aspect_sentiment == 2):
        return 1

    return 0


df = pd.read_csv(INPUT_PATH)

emotions = []
for _, row in df.iterrows():
    emotions.append(assign_emotion(row["text"], int(row["sentiment"]), int(row["primary_aspect"]), int(row["aspect_sentiment"])))

df["emotion"] = emotions

df.to_csv(OUTPUT_PATH, index=False)

print("✅ Emotion v3 generated")
print(df["emotion"].value_counts())

