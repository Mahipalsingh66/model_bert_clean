# # ============================================================
# # ENTERPRISE TRAINING SCRIPT — XLM-R SENTIMENT (GOLD_v2.3 READY)
# # Compatible with SageMaker Manual Tuning + Automatic HPO
# # Owner   : Mahipal Singh
# # Purpose : Production-grade training with full hyperparameter control
# # ============================================================

# import os
# import json
# import argparse
# import pandas as pd
# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     get_linear_schedule_with_warmup
# )
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score

# # ------------------------------------------------------------
# # ARGPARSE — ALL TUNABLE HYPERPARAMETERS
# # ------------------------------------------------------------

# def parse_args():
#     parser = argparse.ArgumentParser()

#     # Core
#     parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
#     parser.add_argument("--epochs", type=int, default=3)
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--learning_rate", type=float, default=2e-5)
#     parser.add_argument("--weight_decay", type=float, default=0.01)

#     # Stability
#     parser.add_argument("--warmup_ratio", type=float, default=0.1)
#     parser.add_argument("--max_grad_norm", type=float, default=1.0)
#     parser.add_argument("--seed", type=int, default=42)

#     # Class weights
#     parser.add_argument("--use_class_weights", type=int, default=0)
#     parser.add_argument("--neg_weight", type=float, default=1.0)
#     parser.add_argument("--neu_weight", type=float, default=1.0)
#     parser.add_argument("--pos_weight", type=float, default=1.0)

#     # Misc
#     parser.add_argument("--max_length", type=int, default=128)

#     return parser.parse_args()

# args = parse_args()

# # ------------------------------------------------------------
# # REPRODUCIBILITY
# # ------------------------------------------------------------

# torch.manual_seed(args.seed)
# np.random.seed(args.seed)

# # ------------------------------------------------------------
# # SAGEMAKER PATHS
# # ------------------------------------------------------------

# train_path = os.environ["SM_CHANNEL_TRAIN"]
# val_path   = os.environ["SM_CHANNEL_VAL"]
# model_dir  = os.environ["SM_MODEL_DIR"]
# output_dir = os.environ["SM_OUTPUT_DATA_DIR"]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print("Train path:", train_path)
# print("Val path:", val_path)
# print("Model dir:", model_dir)
# print("Output dir:", output_dir)
# print("Device:", device)

# # ------------------------------------------------------------
# # LOAD DATA
# # ------------------------------------------------------------

# train_df = pd.read_csv(os.path.join(train_path, "train.csv"))
# val_df   = pd.read_csv(os.path.join(val_path, "val.csv"))

# assert "text" in train_df.columns
# assert "label" in train_df.columns

# print("Train samples:", len(train_df))
# print("Val samples  :", len(val_df))

# # ------------------------------------------------------------
# # TOKENIZER
# # ------------------------------------------------------------

# tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# # ------------------------------------------------------------
# # DATASET
# # ------------------------------------------------------------

# class SentimentDataset(Dataset):
#     def __init__(self, df, tokenizer, max_length):
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

# train_dataset = SentimentDataset(train_df, tokenizer, args.max_length)
# val_dataset   = SentimentDataset(val_df, tokenizer, args.max_length)

# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
# val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# # ------------------------------------------------------------
# # MODEL
# # ------------------------------------------------------------

# model = AutoModelForSequenceClassification.from_pretrained(
#     args.model_name,
#     num_labels=3
# ).to(device)

# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=args.learning_rate,
#     weight_decay=args.weight_decay
# )

# # ------------------------------------------------------------
# # LOSS FUNCTION — ENTERPRISE CLASS WEIGHTS
# # ------------------------------------------------------------

# if args.use_class_weights == 1:
#     weights = torch.tensor([
#         args.neg_weight,
#         args.neu_weight,
#         args.pos_weight
#     ], dtype=torch.float).to(device)

#     print("Using manual class weights:", weights.tolist())
#     criterion = torch.nn.CrossEntropyLoss(weight=weights)
# else:
#     print("Using standard CrossEntropyLoss")
#     criterion = torch.nn.CrossEntropyLoss()

# # ------------------------------------------------------------
# # SCHEDULER
# # ------------------------------------------------------------

# total_steps = len(train_loader) * args.epochs
# warmup_steps = int(args.warmup_ratio * total_steps)

# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=warmup_steps,
#     num_training_steps=total_steps
# )

# # ------------------------------------------------------------
# # TRAINING LOOP
# # ------------------------------------------------------------

# print("Starting training...")

# for epoch in range(args.epochs):
#     model.train()
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

#         torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

#         optimizer.step()
#         scheduler.step()

#         epoch_loss += loss.item()

#         if step % 50 == 0:
#             print(f"Epoch {epoch+1} | Step {step} | Loss {loss.item():.4f}")

#     print(f"Epoch {epoch+1} finished | Avg Loss: {epoch_loss/len(train_loader):.4f}")

# # ------------------------------------------------------------
# # EVALUATION (FULL METRICS FOR HPO)
# # ------------------------------------------------------------

# print("Running evaluation on validation set...")

# model.eval()
# all_preds = []
# all_labels = []

# with torch.no_grad():
#     for batch in val_loader:
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)

#         outputs = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )

#         preds = torch.argmax(outputs.logits, dim=1)

#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())

# accuracy = accuracy_score(all_labels, all_preds)
# macro_f1 = f1_score(all_labels, all_preds, average="macro")

# # Per-class recall (0=neg, 1=neu, 2=pos)
# recalls = recall_score(all_labels, all_preds, average=None, labels=[0, 1, 2])
# neg_recall, neu_recall, pos_recall = recalls.tolist()

# report = classification_report(all_labels, all_preds, digits=4)
# cm = confusion_matrix(all_labels, all_preds)

# print("\nVALIDATION ACCURACY:", accuracy)
# print("VALIDATION MACRO F1:", macro_f1)
# print("NEG RECALL:", neg_recall)
# print("NEU RECALL:", neu_recall)
# print("POS RECALL:", pos_recall)

# # IMPORTANT FOR SAGEMAKER HPO LOG PARSING
# print(f"validation:macro_f1={macro_f1}")

# # ------------------------------------------------------------
# # SAVE METRICS
# # ------------------------------------------------------------

# metrics = {
#     "accuracy": float(accuracy),
#     "macro_f1": float(macro_f1),
#     "neg_recall": float(neg_recall),
#     "neu_recall": float(neu_recall),
#     "pos_recall": float(pos_recall)
# }

# with open(os.path.join(output_dir, "metrics.json"), "w") as f:
#     json.dump(metrics, f, indent=4)

# with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
#     f.write(report)

# np.savetxt(os.path.join(output_dir, "confusion_matrix.txt"), cm, fmt="%d")

# print("\nMETRICS SAVED TO:", output_dir)

# # ------------------------------------------------------------
# # SAVE MODEL
# # ------------------------------------------------------------

# model.save_pretrained(model_dir)
# tokenizer.save_pretrained(model_dir)

# print("\nTRAINING COMPLETED SUCCESSFULLY")
# print("MODEL SAVED TO:", model_dir)
# ============================================================
# FILE 1 : train_sagemaker_v2.py
# PURPOSE: Multi-head training (sentiment + intent + aspect)
# INPUT  : text, sentiment, customer_intent, primary_aspect
# ============================================================

import os
import json
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score

# -------------------------
# CONFIG (SAGEMAKER)
# -------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "xlm-roberta-base")
EPOCHS = int(os.environ.get("EPOCHS", 3))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
LR = float(os.environ.get("LR", 2e-5))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 160))

NUM_SENTIMENT = 3
NUM_INTENT = 3
NUM_ASPECT = 5

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
            "labels_sentiment": torch.tensor(self.sentiments[idx]),
            "labels_intent": torch.tensor(self.intents[idx]),
            "labels_aspect": torch.tensor(self.aspects[idx])
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
        self.aspect_head    = nn.Linear(hidden, NUM_ASPECT)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]

        return (
            self.sentiment_head(pooled),
            self.intent_head(pooled),
            self.aspect_head(pooled)
        )

model = CXMultiHeadModel(MODEL_NAME).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# -------------------------
# TRAIN
# -------------------------
print("🚀 Training CX Backbone v1 ...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        y_sent = batch["labels_sentiment"].to(device)
        y_int  = batch["labels_intent"].to(device)
        y_asp  = batch["labels_aspect"].to(device)

        logits_sent, logits_int, logits_asp = model(ids, mask)

        loss_sent = loss_fn(logits_sent, y_sent)
        loss_int  = loss_fn(logits_int, y_int)
        loss_asp  = loss_fn(logits_asp, y_asp)

        loss = 1.0 * loss_sent + 0.7 * loss_int + 0.7 * loss_asp
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} finished | Avg Loss: {total_loss/len(train_loader):.4f}")

# -------------------------
# EVALUATION (ONLY SENTIMENT METRIC FOR NOW)
# -------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels_sentiment"].to(device)

        logits_sent, _, _ = model(ids, mask)
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

torch.save(model.state_dict(), os.path.join(model_dir, "cx_backbone_v1.pt"))

print("✅ TRAINING COMPLETED")
print("Sentiment Accuracy:", accuracy)

# ============================================================
# FILE 2 : run_training_v2.py
# PURPOSE: Launch ONE cost-safe training job
# ============================================================

"""
import sagemaker
from sagemaker.pytorch import PyTorch

role = "arn:aws:iam::419154172513:role/SageMakerExecutionRole-BERT"
bucket = "brt-ml-bucket-419154172513"

session = sagemaker.Session()

estimator = PyTorch(
    entry_point="train_sagemaker_v2.py",
    source_dir="D:/model_bert_copy/src/sagemaker",
    role=role,
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    framework_version="2.0.1",
    py_version="py310",
    output_path=f"s3://{bucket}/models/",
    disable_profiler=True,
    debugger_hook_config=False,
    hyperparameters={
        "EPOCHS": 3,
        "BATCH_SIZE": 16,
        "LR": 2e-5,
        "MAX_LENGTH": 160,
        "MODEL_NAME": "xlm-roberta-base"
    },
    sagemaker_session=session
)

# FINAL CX DATA
estimator.fit({
    "train": f"s3://{bucket}/gold/v2.3_multitask/train_cx_v1.csv",
    "val":   f"s3://{bucket}/gold/v2.3_multitask/val_multi_v1.csv"
})
"""
