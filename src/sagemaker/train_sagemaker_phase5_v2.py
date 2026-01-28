# ============================================================
# FILE 2 : train_sagemaker_phase5_v2.py
# PURPOSE: Optimized Emotion Head Training
# ============================================================

import os, json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

MODEL_NAME = os.environ.get("MODEL_NAME", "xlm-roberta-base")
EPOCHS = int(os.environ.get("EPOCHS", 5))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
LR = float(os.environ.get("LR", 2e-5))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 160))

NUM_ASPECTS = 8
NUM_ASPECT_SENT = 3
NUM_CLASSES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_path = os.environ["SM_CHANNEL_TRAIN"]
val_path   = os.environ["SM_CHANNEL_VAL"]
model_dir  = os.environ["SM_MODEL_DIR"]
output_dir = os.environ["SM_OUTPUT_DATA_DIR"]

train_df = pd.read_csv(os.path.join(train_path, "train.csv"))
val_df   = pd.read_csv(os.path.join(val_path, "val.csv"))

class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts = df["text"].tolist()
        self.aspects = df["primary_aspect"].tolist()
        self.aspect_sent = df["aspect_sentiment"].tolist()
        self.labels = df["emotion"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "aspect": torch.tensor(self.aspects[idx], dtype=torch.long),
            "aspect_sent": torch.tensor(self.aspect_sent[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class EmotionModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.aspect_embedding = nn.Embedding(NUM_ASPECTS, 64)
        self.aspect_sent_embedding = nn.Embedding(NUM_ASPECT_SENT, 32)
        self.classifier = nn.Linear(hidden + 96, NUM_CLASSES)

    def forward(self, ids, mask, asp, asp_s):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        cls = out.last_hidden_state[:,0]
        a = self.aspect_embedding(asp)
        s = self.aspect_sent_embedding(asp_s)
        x = torch.cat([cls,a,s], dim=1)
        return self.classifier(x)

model = EmotionModel(MODEL_NAME).to(device)

# 🔥 Unfreeze last 3 layers now
for name, param in model.encoder.named_parameters():
    if any(x in name for x in ["layer.9","layer.10","layer.11"]):
        param.requires_grad = True
    else:
        param.requires_grad = False

# Class weights (boost Frustrated class)
weights = torch.tensor([1.0, 1.0, 1.5, 1.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

train_loader = DataLoader(EmotionDataset(train_df, AutoTokenizer.from_pretrained(MODEL_NAME), MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(EmotionDataset(val_df, AutoTokenizer.from_pretrained(MODEL_NAME), MAX_LENGTH), batch_size=BATCH_SIZE)

print("\n🚀 Phase-5 Optimized Training v2\n")

for epoch in range(EPOCHS):
    model.train(); total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        asp  = batch["aspect"].to(device)
        asp_s= batch["aspect_sent"].to(device)
        y    = batch["labels"].to(device)
        logits = model(ids, mask, asp, asp_s)
        loss = criterion(logits, y)
        loss.backward(); optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Avg Loss {total_loss/len(train_loader):.4f}")

model.eval(); all_preds=[]; all_labels=[]
with torch.no_grad():
    for batch in val_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        asp  = batch["aspect"].to(device)
        asp_s= batch["aspect_sent"].to(device)
        y    = batch["labels"].to(device)
        preds = torch.argmax(model(ids, mask, asp, asp_s), dim=1)
        all_preds.extend(preds.cpu().numpy()); all_labels.extend(y.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
rep = classification_report(all_labels, all_preds, digits=4)

with open(os.path.join(output_dir, "metrics.json"), "w") as f: json.dump({"emotion_accuracy": float(acc)}, f, indent=4)
with open(os.path.join(output_dir, "classification_report.txt"), "w") as f: f.write(rep)
np.savetxt(os.path.join(output_dir, "emotion_confusion_matrix.txt"), cm, fmt="%d")

model.encoder.save_pretrained(model_dir)
torch.save(model.state_dict(), os.path.join(model_dir, "cx_backbone_phase5_v2.pt"))

print("\n✅ PHASE-5 OPTIMIZED TRAINING COMPLETED")
print("Emotion Accuracy:", acc)


