# # ============================================================
# # MULTI-DOMAIN MULTI-TASK TRAINING (PRODUCTION FINAL)
# # File: train_sagemaker_multitask_v3.py
# # ============================================================

# import os
# import json
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer, AutoModel
# from sklearn.metrics import accuracy_score

# # -------------------------
# # SAGEMAKER ENV
# # -------------------------
# train_path = os.environ["SM_CHANNEL_TRAIN"]
# val_path   = os.environ["SM_CHANNEL_VAL"]
# model_dir  = os.environ["SM_MODEL_DIR"]
# output_dir = os.environ["SM_OUTPUT_DATA_DIR"]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # -------------------------
# # CONFIG
# # -------------------------
# MODEL_NAME = "xlm-roberta-base"
# MAX_LEN = 160
# EPOCHS = int(os.environ.get("EPOCHS", 3))
# BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
# LR = float(os.environ.get("LR", 2e-5))

# SHARED_COLS = ["sentiment", "aspect_sentiment", "emotion"]

# DOMAIN_HEADS = {
#     "logistics": {
#         "primary_aspect": 8,
#         "customer_intent": 7,
#     },
#     "banking": {
#         "primary_aspect": 14,
#         "customer_intent": 7,
#     }
# }

# # -------------------------
# # LOAD DATA
# # -------------------------
# train_df = pd.read_csv(os.path.join(train_path, "train.csv"))
# val_df   = pd.read_csv(os.path.join(val_path, "val.csv"))

# # -------------------------
# # TOKENIZER
# # -------------------------
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# # -------------------------
# # DATASET
# # -------------------------
# class MultiTaskDataset(Dataset):
#     def __init__(self, df):
#         self.texts = [f"{i} : {t}" for i, t in zip(df["industry"], df["text"])]
#         self.industry = df["industry"].tolist()
#         self.labels = df.to_dict("records")

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         enc = tokenizer(
#             self.texts[idx],
#             truncation=True,
#             padding="max_length",
#             max_length=MAX_LEN,
#             return_tensors="pt"
#         )

#         item = {
#             "input_ids": enc["input_ids"].squeeze(0),
#             "attention_mask": enc["attention_mask"].squeeze(0),
#             "industry": self.industry[idx],
#         }

#         for col in SHARED_COLS + ["primary_aspect", "customer_intent"]:
#             item[col] = int(self.labels[idx][col])

#         return item

# train_loader = DataLoader(MultiTaskDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
# val_loader   = DataLoader(MultiTaskDataset(val_df), batch_size=BATCH_SIZE)

# # -------------------------
# # MODEL
# # -------------------------
# class CXMultiDomainModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = AutoModel.from_pretrained(MODEL_NAME)
#         hidden = self.encoder.config.hidden_size

#         self.shared_heads = nn.ModuleDict({
#             col: nn.Linear(hidden, train_df[col].nunique())
#             for col in SHARED_COLS
#         })

#         self.domain_heads = nn.ModuleDict()
#         for domain, heads in DOMAIN_HEADS.items():
#             self.domain_heads[domain] = nn.ModuleDict({
#                 name: nn.Linear(hidden, size)
#                 for name, size in heads.items()
#             })

#     def forward(self, input_ids, attention_mask, industry):
#         out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         cls = out.last_hidden_state[:, 0]

#         outputs = {k: head(cls) for k, head in self.shared_heads.items()}
#         domain_out = []

#         for i, dom in enumerate(industry):
#             heads = self.domain_heads[dom]
#             domain_out.append({
#                 "primary_aspect": heads["primary_aspect"](cls[i:i+1]),
#                 "customer_intent": heads["customer_intent"](cls[i:i+1])
#             })

#         return outputs, domain_out

# model = CXMultiDomainModel().to(device)

# # -------------------------
# # TRAINING
# # -------------------------
# optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
# criterion = nn.CrossEntropyLoss()

# for epoch in range(EPOCHS):
#     model.train()
#     total_loss = 0

#     for batch in train_loader:
#         optimizer.zero_grad()

#         outputs, domain_out = model(
#             batch["input_ids"].to(device),
#             batch["attention_mask"].to(device),
#             batch["industry"]
#         )

#         loss = 0
#         for col in SHARED_COLS:
#             loss += criterion(outputs[col], torch.tensor(batch[col]).to(device))

#         for i, dom in enumerate(batch["industry"]):
#             loss += criterion(
#                 domain_out[i]["primary_aspect"],
#                 torch.tensor([batch["primary_aspect"][i]]).to(device)
#             )
#             loss += criterion(
#                 domain_out[i]["customer_intent"],
#                 torch.tensor([batch["customer_intent"][i]]).to(device)
#             )

#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     print(f"Epoch {epoch+1} | Loss {total_loss/len(train_loader):.4f}")

# # -------------------------
# # SAVE MODEL
# # -------------------------
# model.encoder.save_pretrained(model_dir)
# tokenizer.save_pretrained(model_dir)

# torch.save(
#     model.state_dict(),
#     os.path.join(model_dir, "cx_multidomain_multitask_v3.pt")
# )

# print("✅ MULTI-DOMAIN MODEL SAVED SUCCESSFULLY")

# ============================================================
# MULTI-DOMAIN MULTI-TASK TRAINING SCRIPT (FINAL v3)
# File: train_sagemaker_multitask_v3.py
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
# SAGEMAKER ENV
# -------------------------
train_path = os.environ["SM_CHANNEL_TRAIN"]
val_path   = os.environ["SM_CHANNEL_VAL"]
model_dir  = os.environ["SM_MODEL_DIR"]
output_dir = os.environ["SM_OUTPUT_DATA_DIR"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 160

EPOCHS = int(os.environ.get("EPOCHS", 3))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
LR = float(os.environ.get("LR", 2e-5))

LABEL_COLS = [
    "sentiment",
    "customer_intent",
    "primary_aspect",
    "aspect_sentiment",
    "emotion"
]

# -------------------------
# LOAD DATA
# -------------------------
train_df = pd.read_csv(os.path.join(train_path, "train.csv"))
val_df   = pd.read_csv(os.path.join(val_path, "val.csv"))

required_cols = ["text", "industry"] + LABEL_COLS
for c in required_cols:
    if c not in train_df.columns:
        raise ValueError(f"Missing column in train.csv: {c}")
    if c not in val_df.columns:
        raise ValueError(f"Missing column in val.csv: {c}")

# 🔒 IMPORTANT: head sizes from TRAIN only
HEAD_SIZES = {
    col: int(train_df[col].max()) + 1
    for col in LABEL_COLS
}

# -------------------------
# TOKENIZER
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# -------------------------
# DATASET
# -------------------------
class MultiTaskDataset(Dataset):
    def __init__(self, df):
        # 🔥 Industry-conditioned input
        self.texts = [
            f"{ind} : {txt}"
            for ind, txt in zip(df["industry"], df["text"])
        ]
        self.labels = df[LABEL_COLS].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

        for i, col in enumerate(LABEL_COLS):
            item[col] = torch.tensor(int(self.labels[idx][i]), dtype=torch.long)

        return item


train_loader = DataLoader(
    MultiTaskDataset(train_df),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    MultiTaskDataset(val_df),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -------------------------
# MODEL
# -------------------------
class CXMultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size

        self.heads = nn.ModuleDict({
            col: nn.Linear(hidden, HEAD_SIZES[col])
            for col in LABEL_COLS
        })

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = out.last_hidden_state[:, 0]
        return {k: head(cls) for k, head in self.heads.items()}


model = CXMultiTaskModel().to(device)

# -------------------------
# OPTIMIZER & LOSS
# -------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# -------------------------
# TRAINING
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        outputs = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device)
        )

        loss = sum(
            criterion(outputs[col], batch[col].to(device))
            for col in LABEL_COLS
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

# -------------------------
# VALIDATION (RESTORED ✅)
# -------------------------
model.eval()
metrics = {}

with torch.no_grad():
    for col in LABEL_COLS:
        preds, labels = [], []

        for batch in val_loader:
            outputs = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device)
            )

            pred = torch.argmax(outputs[col], dim=1).cpu().numpy()
            preds.extend(pred)
            labels.extend(batch[col].numpy())

        acc = accuracy_score(labels, preds)
        metrics[f"{col}_accuracy"] = float(acc)
        print(f"{col} accuracy: {acc:.4f}")

# -------------------------
# SAVE METRICS
# -------------------------
with open(os.path.join(output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# -------------------------
# SAVE MODEL (PRODUCTION SAFE)
# -------------------------
model.encoder.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

torch.save(
    model.state_dict(),
    os.path.join(model_dir, "cx_multitask_industry_v3.pt")
)

print("✅ MULTI-DOMAIN MODEL SAVED SUCCESSFULLY")