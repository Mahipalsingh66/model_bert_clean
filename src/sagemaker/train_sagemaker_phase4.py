# ============================================================
# FILE 1 : train_sagemaker_phase4.py
# PURPOSE: Phase-4 Training — Aspect-Sentiment Head (3-class)
# DESIGN : Frozen XLM-R backbone + new head
# TARGET : aspect_sentiment (0=Neg,1=Neu,2=Pos)
# INPUT  : text, primary_aspect
# ============================================================

import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# ---------------- CONFIG ----------------
MODEL_NAME = os.environ.get("MODEL_NAME", "xlm-roberta-base")
EPOCHS = int(os.environ.get("EPOCHS", 4))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
LR = float(os.environ.get("LR", 2e-5))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 160))

NUM_CLASSES = 3   # aspect_sentiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- SAGEMAKER PATHS ----------------
train_path = os.environ["SM_CHANNEL_TRAIN"]
val_path   = os.environ["SM_CHANNEL_VAL"]
model_dir  = os.environ["SM_MODEL_DIR"]
output_dir = os.environ["SM_OUTPUT_DATA_DIR"]

print("Train path:", train_path)
print("Val path  :", val_path)
print("Model dir :", model_dir)
print("Output dir:", output_dir)

# ---------------- LOAD DATA ----------------
train_df = pd.read_csv(os.path.join(train_path, "train.csv"))
val_df   = pd.read_csv(os.path.join(val_path, "val.csv"))

required = ["text", "primary_aspect", "aspect_sentiment"]
for c in required:
    if c not in train_df.columns:
        raise ValueError(f"Missing column: {c}")

# ---------------- TOKENIZER ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ---------------- DATASET ----------------
class AspectSentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts = df["text"].tolist()
        self.aspects = df["primary_aspect"].tolist()
        self.labels = df["aspect_sentiment"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Combine text + aspect hint
        combined = f"[ASPECT={self.aspects[idx]}] {self.texts[idx]}"

        enc = self.tokenizer(
            combined,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_ds = AspectSentimentDataset(train_df, tokenizer, MAX_LENGTH)
val_ds   = AspectSentimentDataset(val_df, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODEL ----------------
class AspectSentimentModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, NUM_CLASSES)

    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        pooled = out.last_hidden_state[:, 0]
        return self.classifier(pooled)

model = AspectSentimentModel(MODEL_NAME).to(device)

# 🔒 Freeze backbone
for p in model.encoder.parameters():
    p.requires_grad = False

# ---------------- LOSS (MILD WEIGHTS) ----------------
labels_all = torch.tensor(train_df["aspect_sentiment"].values)
counts = torch.bincount(labels_all, minlength=NUM_CLASSES).float()
weights = counts.sum() / counts
weights = (weights / weights.mean()).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR)

print("Class weights:", weights.cpu().numpy())

# ---------------- TRAIN ----------------
print("🚀 Phase-4 Training — Aspect Sentiment Head")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)

        logits = model(ids, mask)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} finished | Avg Loss: {total_loss/len(train_loader):.4f}")

# ---------------- EVALUATION ----------------
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)

        logits = model(ids, mask)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, digits=4)

# ---------------- SAVE METRICS ----------------
metrics = {"aspect_sentiment_accuracy": float(accuracy)}

with open(os.path.join(output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write(report)

np.savetxt(os.path.join(output_dir, "aspect_sentiment_confusion_matrix.txt"), cm, fmt="%d")

# Save model
model.encoder.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
torch.save(model.state_dict(), os.path.join(model_dir, "cx_backbone_phase4.pt"))

print("\n✅ PHASE-4 TRAINING COMPLETED")
print("Aspect-Sentiment Accuracy:", accuracy)


