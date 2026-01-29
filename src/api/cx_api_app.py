# ============================================================
# FILE    : cx_api_app.py
# PURPOSE : Phase-7 REAL-TIME CX INFERENCE API (PRODUCTION)
# STACK   : FastAPI + PyTorch + Transformers
# MODE    : CPU/GPU auto-detect
# ============================================================

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# MODEL PATHS (LOCAL ARTIFACTS)
# ------------------------------------------------------------

BASE_PATH = "D:/model_bert_copy/artifacts"

PATHS = {
    "sentiment": f"{BASE_PATH}/sentiment_model",
    "intent": f"{BASE_PATH}/intent_model",
    "aspect": f"{BASE_PATH}/aspect_model",
    "aspect_sentiment": f"{BASE_PATH}/aspect_sentiment_model",
    "emotion": f"{BASE_PATH}/emotion_model",
}

# ------------------------------------------------------------
# LOAD TOKENIZER (SHARED)
# ------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(PATHS["sentiment"])

# ------------------------------------------------------------
# LOAD MODELS ONCE (IMPORTANT)
# ------------------------------------------------------------

models = {}
for name, path in PATHS.items():
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(device)
    model.eval()
    models[name] = model

# ------------------------------------------------------------
# PRIORITY ENGINE (LOCKED v1)
# ------------------------------------------------------------

def assign_priority(sentiment, aspect, aspect_sentiment, emotion):
    if emotion == 4:
        return 3
    if emotion == 3 and aspect in [2, 7]:
        return 3
    if sentiment == 0 and aspect in [2, 7] and aspect_sentiment == 0:
        return 3
    if emotion == 3:
        return 2
    if emotion == 2 and aspect in [0, 1, 5]:
        return 2
    if sentiment == 0 and aspect == 3:
        return 2
    if emotion == 2:
        return 1
    if sentiment == 1 and aspect in [0, 1, 5]:
        return 1
    if sentiment == 0 and aspect == 4:
        return 1
    return 0

# ------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------

app = FastAPI(title="CX Intelligence API", version="1.0")

class CXRequest(BaseModel):
    text: str

class CXResponse(BaseModel):
    sentiment: int
    customer_intent: int
    primary_aspect: int
    aspect_sentiment: int
    emotion: int
    priority: int

# ------------------------------------------------------------
# INFERENCE ENDPOINT
# ------------------------------------------------------------

@app.post("/predict", response_model=CXResponse)
def predict(req: CXRequest):

    inputs = tokenizer(
        req.text,
        truncation=True,
        padding="max_length",
        max_length=160,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        sentiment = torch.argmax(models["sentiment"](**inputs).logits, dim=1).item()
        intent = torch.argmax(models["intent"](**inputs).logits, dim=1).item()
        aspect = torch.argmax(models["aspect"](**inputs).logits, dim=1).item()
        asp_sent = torch.argmax(models["aspect_sentiment"](**inputs).logits, dim=1).item()
        emotion = torch.argmax(models["emotion"](**inputs).logits, dim=1).item()

    priority = assign_priority(sentiment, aspect, asp_sent, emotion)

    return CXResponse(
        sentiment=sentiment,
        customer_intent=intent,
        primary_aspect=aspect,
        aspect_sentiment=asp_sent,
        emotion=emotion,
        priority=priority
    )
