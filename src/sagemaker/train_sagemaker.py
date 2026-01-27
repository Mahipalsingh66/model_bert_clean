# # src/sagemaker/train_sagemaker.py

# import os
# import torch
# import pandas as pd
# from torch.utils.data import DataLoader
# from transformers import AutoModelForSequenceClassification
# from src.features.tokenizer import load_tokenizer
# from src.data.loader import SentimentDataset

# MODEL_NAME = "xlm-roberta-base"

# def main():
#     # SageMaker paths
#     train_path = os.environ.get("SM_CHANNEL_TRAIN")
#     val_path   = os.environ.get("SM_CHANNEL_VAL")
#     model_dir  = os.environ.get("SM_MODEL_DIR")

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print("Using device:", device)

#     tokenizer = load_tokenizer()

#     train_dataset = SentimentDataset(
#         csv_path=os.path.join(train_path, "train.csv"),
#         tokenizer=tokenizer,
#         max_length=128
#     )

#     val_dataset = SentimentDataset(
#         csv_path=os.path.join(val_path, "val.csv"),
#         tokenizer=tokenizer,
#         max_length=128
#     )

#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#     val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

#     model = AutoModelForSequenceClassification.from_pretrained(
#         MODEL_NAME,
#         num_labels=3
#     ).to(device)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
#     criterion = torch.nn.CrossEntropyLoss()

#     # -------- Training loop --------
#     EPOCHS = 3

#     for epoch in range(EPOCHS):
#         model.train()
#         total_loss = 0

#         for step, batch in enumerate(train_loader):
#             optimizer.zero_grad()

#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)

#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask
#             )

#             loss = criterion(outputs.logits, labels)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#             if step % 100 == 0:
#                 print(f"Epoch {epoch+1} | Step {step} | Loss {loss.item():.4f}")

#         print(f"\nEpoch {epoch+1} completed | Avg loss {total_loss/len(train_loader):.4f}\n")

#     # -------- Save model --------
#     print("Saving model to:", model_dir)
#     model.save_pretrained(model_dir)
#     tokenizer.save_pretrained(model_dir)

#     print("✅ Training finished and model saved.")

# if __name__ == "__main__":
#     main()
# import os
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     get_linear_schedule_with_warmup
# )

# # -------------------------
# # CONFIG (same as your local)
# # -------------------------
# MODEL_NAME = "xlm-roberta-base"
# EPOCHS = 1          # FIRST RUN = 1 epoch only (cost safe)
# BATCH_SIZE = 8
# LR = 2e-5
# NUM_LABELS = 3

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # -------------------------
# # SageMaker paths
# # -------------------------
# train_path = os.environ["SM_CHANNEL_TRAIN"]
# val_path   = os.environ["SM_CHANNEL_VAL"]
# model_dir  = os.environ["SM_MODEL_DIR"]

# print("Train path:", train_path)
# print("Val path:", val_path)
# print("Model dir:", model_dir)
# print("Device:", device)

# # -------------------------
# # Load CSV
# # -------------------------
# train_df = pd.read_csv(os.path.join(train_path, "train.csv"))
# val_df   = pd.read_csv(os.path.join(val_path, "val.csv"))

# print("Train samples:", len(train_df))
# print("Val samples:", len(val_df))

# assert "text" in train_df.columns, "train.csv must have 'text' column"
# assert "label" in train_df.columns, "train.csv must have 'label' column"

# # -------------------------
# # Tokenizer (replaces load_tokenizer)
# # -------------------------
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# # -------------------------
# # Custom Dataset (replaces SentimentDataset)
# # -------------------------
# class SentimentDataset(Dataset):
#     def __init__(self, df, tokenizer, max_length=128):
#         self.texts = df["text"].tolist()
#         self.labels = df["label"].tolist()
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         encoding = self.tokenizer(
#             self.texts[idx],
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"
#         )

#         return {
#             "input_ids": encoding["input_ids"].squeeze(0),
#             "attention_mask": encoding["attention_mask"].squeeze(0),
#             "labels": torch.tensor(self.labels[idx], dtype=torch.long)
#         }

# train_dataset = SentimentDataset(train_df, tokenizer)
# train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# # -------------------------
# # Model (same as local)
# # -------------------------
# model = AutoModelForSequenceClassification.from_pretrained(
#     MODEL_NAME,
#     num_labels=NUM_LABELS
# ).to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# # -------------------------
# # Class weights (same logic as your local)
# # -------------------------
# labels_all = torch.tensor(train_dataset.labels)
# class_counts = torch.bincount(labels_all, minlength=NUM_LABELS).float()
# class_weights = class_counts.sum() / class_counts
# class_weights = class_weights.to(device)

# criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# # -------------------------
# # Scheduler (same as local)
# # -------------------------
# total_steps = len(train_loader) * EPOCHS
# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=int(0.1 * total_steps),
#     num_training_steps=total_steps
# )

# # -------------------------
# # Training loop
# # -------------------------
# model.train()
# print("Starting training...")

# for epoch in range(EPOCHS):
#     epoch_loss = 0

#     for step, batch in enumerate(train_loader):
#         optimizer.zero_grad()

#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)

#         outputs = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )

#         loss = criterion(outputs.logits, labels)
#         loss.backward()

#         optimizer.step()
#         scheduler.step()

#         epoch_loss += loss.item()

#         if step % 20 == 0:
#             print(f"Epoch {epoch+1} | Step {step} | Loss {loss.item():.4f}")

#     print(f"Epoch {epoch+1} finished | Avg Loss: {epoch_loss/len(train_loader):.4f}")

# # -------------------------
# # Save model (SageMaker way)
# # -------------------------
# model.save_pretrained(model_dir)
# tokenizer.save_pretrained(model_dir)

# print("TRAINING COMPLETED SUCCESSFULLY")
# print("MODEL SAVED TO:", model_dir)
import os
import json
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = "xlm-roberta-base"
EPOCHS = 2                 # You can increase later
BATCH_SIZE = 8
LR = 2e-5
NUM_LABELS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# SageMaker paths
# -------------------------
train_path = os.environ["SM_CHANNEL_TRAIN"]
val_path   = os.environ["SM_CHANNEL_VAL"]
model_dir  = os.environ["SM_MODEL_DIR"]
output_dir = os.environ["SM_OUTPUT_DATA_DIR"]   # For metrics

print("Train path:", train_path)
print("Val path:", val_path)
print("Model dir:", model_dir)
print("Output dir:", output_dir)
print("Device:", device)

# -------------------------
# Load CSV
# -------------------------
train_df = pd.read_csv(os.path.join(train_path, "train.csv"))
val_df   = pd.read_csv(os.path.join(val_path, "val.csv"))

print("Train samples:", len(train_df))
print("Val samples:", len(val_df))

assert "text" in train_df.columns, "train.csv must have 'text' column"
assert "label" in train_df.columns, "train.csv must have 'label' column"

# -------------------------
# Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# -------------------------
# Dataset
# -------------------------
class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = SentimentDataset(train_df, tokenizer)
val_dataset   = SentimentDataset(val_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# Model
# -------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# -------------------------
# Class weights
# -------------------------
labels_all = torch.tensor(train_dataset.labels)
class_counts = torch.bincount(labels_all, minlength=NUM_LABELS).float()
class_weights = class_counts.sum() / class_counts
class_weights = class_weights.to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# -------------------------
# Scheduler
# -------------------------
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# -------------------------
# TRAIN + EVALUATE
# -------------------------
print("Starting training...")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for step, batch in enumerate(train_loader):
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
        scheduler.step()

        epoch_loss += loss.item()

        if step % 50 == 0:
            print(f"Epoch {epoch+1} | Step {step} | Loss {loss.item():.4f}")

    print(f"Epoch {epoch+1} finished | Avg Loss: {epoch_loss/len(train_loader):.4f}")

# -------------------------
# 🔥 EVALUATION
# -------------------------
print("Running evaluation on validation set...")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, digits=4)
cm = confusion_matrix(all_labels, all_preds)

print("\nVALIDATION ACCURACY:", accuracy)
print("\nCLASSIFICATION REPORT:\n", report)
print("\nCONFUSION MATRIX:\n", cm)

# -------------------------
# 🔥 SAVE METRICS TO S3 (AUTOMATIC)
# -------------------------
metrics = {
    "accuracy": float(accuracy)
}

with open(os.path.join(output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write(report)

np.savetxt(os.path.join(output_dir, "confusion_matrix.txt"), cm, fmt="%d")

print("\nMETRICS SAVED TO:", output_dir)

# -------------------------
# Save model
# -------------------------
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

print("\nTRAINING COMPLETED SUCCESSFULLY")
print("MODEL SAVED TO:", model_dir)
