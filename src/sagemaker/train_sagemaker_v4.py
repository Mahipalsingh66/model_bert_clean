# ============================================================
# FILE 1 : train_sagemaker_v4.py
# PURPOSE: Phase-3 CX Backbone — Train PRIMARY_ASPECT only
# INPUT  : text,sentiment,customer_intent,primary_aspect,aspect_flag
# DESIGN : Load Phase-2 backbone, freeze encoder + sentiment + intent
# OUTPUT : Aspect head trained, backbone preserved
# ============================================================

import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "xlm-roberta-base")
EPOCHS = int(os.environ.get("EPOCHS", 3))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
LR = float(os.environ.get("LR", 2e-5))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 160))

NUM_SENTIMENT = 3
NUM_INTENT = 3
NUM_ASPECT = 8   # Delay, Wrong, Damage, Behaviour, Support, Tracking, Pricing, Refund

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_path = os.environ["SM_CHANNEL_TRAIN"]
val_path   = os.environ["SM_CHANNEL_VAL"]
model_dir  = os.environ["SM_MODEL_DIR"]
output_dir = os.environ["SM_OUTPUT_DATA_DIR"]

# -------------------------
# LOAD DATA
# -------------------------
train_df = pd.read_csv(os.path.join(train_path, "train.csv"))
val_df   = pd.read_csv(os.path.join(val_path, "val.csv"))

required_cols = ["text", "sentiment", "customer_intent", "primary_aspect"]
for c in required_cols:
    if c not in train_df.columns:
        raise ValueError(f"Missing column in train.csv: {c}")
    if c not in val_df.columns:
        raise ValueError(f"Missing column in val.csv: {c}")

# -------------------------
# TOKENIZER
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# -------------------------
# DATASET
# -------------------------
class CXDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts = df["text"].tolist()
        self.aspects = df["primary_aspect"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels_aspect": torch.tensor(self.aspects[idx], dtype=torch.long)
        }

train_ds = CXDataset(train_df, tokenizer, MAX_LENGTH)
val_ds   = CXDataset(val_df, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# MODEL (LOAD BACKBONE)
# -------------------------
class CXAspectModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        # Only aspect head now
        self.aspect_head = nn.Linear(hidden, NUM_ASPECT)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        return self.aspect_head(pooled)

model = CXAspectModel(MODEL_NAME).to(device)

# 🔥 Freeze full encoder (protect sentiment + intent knowledge)
for param in model.encoder.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW(model.aspect_head.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# -------------------------
# TRAIN
# -------------------------
print("🚀 Training CX Backbone Phase-3 (PRIMARY ASPECT HEAD)")
print("🔒 Encoder frozen — training aspect head only")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels_aspect"].to(device)

        logits = model(ids, mask)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} finished | Avg Loss: {total_loss/len(train_loader):.4f}")

# -------------------------
# EVALUATION
# -------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels_aspect"].to(device)

        logits = model(ids, mask)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

# -------------------------
# SAVE METRICS
# -------------------------
metrics = {
    "aspect_accuracy": float(accuracy)
}

with open(os.path.join(output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# -------------------------
# SAVE MODEL
# -------------------------
model.encoder.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

torch.save(model.state_dict(), os.path.join(model_dir, "cx_backbone_phase3.pt"))

# Save confusion matrix
import numpy as np
np.savetxt(os.path.join(output_dir, "aspect_confusion_matrix.txt"), cm, fmt="%d")

print("✅ TRAINING COMPLETED")
print("Aspect Accuracy:", accuracy)

