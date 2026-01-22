# run_sentiment_pipeline.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# CONFIG (CHANGE ONLY HERE)
# =========================
MODEL_NAME = "xlm-roberta-base"
TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/val.csv"

MAX_LENGTH = 128
BATCH_SIZE = 8          # Windows-safe
LR = 2e-5
EPOCHS = 3
GAMMA = 2.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# DATASET
# =========================
class SentimentDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length):
        df = pd.read_csv(csv_path)
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# =========================
# FOCAL LOSS
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        loss = self.alpha[targets] * loss
        return loss.mean()


# =========================
# TRAINING
# =========================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================
# EVALUATION
# =========================
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = torch.argmax(outputs.logits, dim=1)
            preds.extend(predictions.cpu().numpy())
            labels.extend(batch["labels"].numpy())

    print("\n📊 Classification Report")
    print(classification_report(labels, preds, digits=3))

    print("📉 Confusion Matrix")
    print(confusion_matrix(labels, preds))


# =========================
# MAIN PIPELINE
# =========================
def main():
    print("🚀 Starting Sentiment Training Pipeline")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = SentimentDataset(TRAIN_CSV, tokenizer, MAX_LENGTH)
    val_ds = SentimentDataset(VAL_CSV, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    ).to(DEVICE)

    # -------- CLASS WEIGHTS --------
    label_counts = np.bincount(train_ds.labels, minlength=3)
    weights = 1.0 / label_counts
    weights = weights / weights.sum() * 3
    class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

    criterion = FocalLoss(alpha=class_weights, gamma=GAMMA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # -------- FREEZE ENCODER (EPOCH 1) --------
    for param in model.base_model.parameters():
        param.requires_grad = False

    print("\n🔒 Epoch 1: Training classifier head only")
    loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    print(f"Epoch 1 Loss: {loss:.4f}")

    # -------- UNFREEZE EVERYTHING --------
    for param in model.base_model.parameters():
        param.requires_grad = True

    print("\n🔓 Fine-tuning full model")
    for epoch in range(2, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        print(f"Epoch {epoch} Loss: {loss:.4f}")

    # -------- EVALUATION --------
    evaluate(model, val_loader, DEVICE)

    print("\n✅ Pipeline complete")


if __name__ == "__main__":
    main()
