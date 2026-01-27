# ============================================================
# FILE 1 : train_sagemaker_v3.py
# PURPOSE: Phase-2 CX Backbone — Sentiment + Customer Intent ONLY
# INPUT  : text, sentiment, customer_intent
# DESIGN : Freeze encoder first epoch, protect sentiment
# ============================================================

import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_path = os.environ["SM_CHANNEL_TRAIN"]
val_path   = os.environ["SM_CHANNEL_VAL"]
model_dir  = os.environ["SM_MODEL_DIR"]
output_dir = os.environ["SM_OUTPUT_DATA_DIR"]

# -------------------------
# LOAD DATA
# -------------------------
train_df = pd.read_csv(os.path.join(train_path, "train_cx_v1.csv"))
val_df   = pd.read_csv(os.path.join(val_path, "val_multi_v1.csv"))

# Safety check
for c in ["text", "sentiment", "customer_intent"]:
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
        self.sentiments = df["sentiment"].tolist()
        self.intents = df["customer_intent"].tolist()
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
            "labels_sentiment": torch.tensor(self.sentiments[idx], dtype=torch.long),
            "labels_intent": torch.tensor(self.intents[idx], dtype=torch.long),
        }

train_ds = CXDataset(train_df, tokenizer, MAX_LENGTH)
val_ds   = CXDataset(val_df, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# MODEL
# -------------------------
class CXMultiHeadModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.sentiment_head = nn.Linear(hidden, NUM_SENTIMENT)
        self.intent_head    = nn.Linear(hidden, NUM_INTENT)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]

        return (
            self.sentiment_head(pooled),
            self.intent_head(pooled)
        )

model = CXMultiHeadModel(MODEL_NAME).to(device)

# 🔥 Freeze encoder first (protect sentiment)
for param in model.encoder.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# -------------------------
# TRAIN
# -------------------------
print("🚀 Training CX Backbone Phase-2 (Sentiment + Intent)")
print("🔒 Encoder frozen for first epoch")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    # Unfreeze after first epoch
    if epoch == 1:
        for param in model.encoder.parameters():
            param.requires_grad = True
        print("🔓 Encoder unfrozen — fine-tuning full backbone")

    for batch in train_loader:
        optimizer.zero_grad()

        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        y_sent = batch["labels_sentiment"].to(device)
        y_int  = batch["labels_intent"].to(device)

        logits_sent, logits_int = model(ids, mask)

        loss_sent = loss_fn(logits_sent, y_sent)
        loss_int  = loss_fn(logits_int, y_int)

        # 🔥 Sentiment protected, intent secondary
        loss = 1.0 * loss_sent + 0.6 * loss_int
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
        y = batch["labels_sentiment"].to(device)

        logits_sent, _ = model(ids, mask)
        preds = torch.argmax(logits_sent, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)

# -------------------------
# SAVE METRICS
# -------------------------
metrics = {"sentiment_accuracy": float(accuracy)}

with open(os.path.join(output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# -------------------------
# SAVE MODEL
# -------------------------
model.encoder.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

torch.save(model.state_dict(), os.path.join(model_dir, "cx_backbone_phase2.pt"))

print("✅ TRAINING COMPLETED")
print("Sentiment Accuracy:", accuracy)
