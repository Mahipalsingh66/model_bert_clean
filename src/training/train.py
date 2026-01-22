# # train.py
# import torch
# from torch.utils.data import DataLoader
# from transformers import AutoModelForSequenceClassification
# from src.features.tokenizer import load_tokenizer
# from src.data.loader import SentimentDataset
# import torch.nn.functional as F
# from pathlib import Path
# MODEL_NAME = "xlm-roberta-base"
# DEVICE = "cpu"  # move to GPU later

# class FocalLoss(torch.nn.Module):
#     def __init__(self, alpha=None, gamma=2.0):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, logits, targets):
#         ce_loss = F.cross_entropy(logits, targets, reduction="none")
#         pt = torch.exp(-ce_loss)
#         focal_loss = (1 - pt) ** self.gamma * ce_loss

#         if self.alpha is not None:
#             alpha_t = self.alpha[targets]
#             focal_loss = alpha_t * focal_loss

#         return focal_loss.mean()

# def main():
#     tokenizer = load_tokenizer()

#     train_dataset = SentimentDataset(
#         csv_path=r"D:/model_bert_copy/data/processed/train.csv",
#         tokenizer=tokenizer,
#         max_length=128
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=4,
#         shuffle=True
#     )

#     model = AutoModelForSequenceClassification.from_pretrained(
#         MODEL_NAME,
#         num_labels=3
#     ).to(DEVICE)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

#     # ---- class weights ----
#     labels_all = train_dataset.labels
#     label_tensor = torch.tensor(labels_all)

#     class_counts = torch.bincount(label_tensor, minlength=3).float()
#     class_weights = 1.0 / class_counts
#     class_weights = class_weights / class_weights.sum() * 3
#     class_weights = class_weights.to(DEVICE)

#     criterion = FocalLoss(alpha=class_weights, gamma=2.0)

#     model.train()
#     total_loss = 0

#     for step, batch in enumerate(train_loader):
#         optimizer.zero_grad()

#         input_ids = batch["input_ids"].to(DEVICE)
#         attention_mask = batch["attention_mask"].to(DEVICE)
#         labels = batch["labels"].to(DEVICE)

#         outputs = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )

#         logits = outputs.logits
#         loss = criterion(logits, labels)

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#         if step % 50 == 0:
#             print(f"Step {step} | Loss: {loss.item():.4f}")

#         if step == 200:  # sanity run
#             break

#     avg_loss = total_loss / (step + 1)
#     print("\n✅ Training sanity check complete")
#     print(f"Average Loss: {avg_loss:.4f}")

# if __name__ == "__main__":
#     main()
# train.py v--2 
# import torch
# from torch.utils.data import DataLoader
# from transformers import (
#     AutoModelForSequenceClassification,
#     get_linear_schedule_with_warmup
# )
# from src.features.tokenizer import load_tokenizer
# from src.data.loader import SentimentDataset

# MODEL_NAME = "xlm-roberta-base"
# DEVICE = "cpu"  # GPU later

# def main():
#     tokenizer = load_tokenizer()

#     train_dataset = SentimentDataset(
#         csv_path=r"D:/model_bert_copy/data/processed/train.csv",
#         tokenizer=tokenizer,
#         max_length=128
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=8,        # ↑ stability
#         shuffle=True
#     )

#     model = AutoModelForSequenceClassification.from_pretrained(
#         MODEL_NAME,
#         num_labels=3
#     ).to(DEVICE)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

#     # ---- class weights (mild, not aggressive) ----
#     labels_all = torch.tensor(train_dataset.labels)
#     class_counts = torch.bincount(labels_all, minlength=3).float()
#     class_weights = class_counts.sum() / class_counts
#     class_weights = class_weights.to(DEVICE)

#     criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

#     total_steps = len(train_loader) * 3  # 3 epochs
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=int(0.1 * total_steps),
#         num_training_steps=total_steps
#     )

#     model.train()

#     for epoch in range(3):
#         epoch_loss = 0

#         for step, batch in enumerate(train_loader):
#             optimizer.zero_grad()

#             input_ids = batch["input_ids"].to(DEVICE)
#             attention_mask = batch["attention_mask"].to(DEVICE)
#             labels = batch["labels"].to(DEVICE)

#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=labels  # IMPORTANT
#             )

#             logits = outputs.logits
#             loss = criterion(logits, labels)

#             loss.backward()
#             optimizer.step()
#             scheduler.step()

#             epoch_loss += loss.item()

#             if step % 50 == 0:
#                 print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")

#         print(f"\nEpoch {epoch+1} completed | Avg Loss: {epoch_loss/len(train_loader):.4f}\n")

#     print("✅ Training complete")

# if __name__ == "__main__":
#     main()

# train.py
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from pathlib import Path

from src.features.tokenizer import load_tokenizer
from src.data.loader import SentimentDataset

MODEL_NAME = "xlm-roberta-base"
DEVICE = "cpu"  # switch to cuda later

SAVE_DIR = Path("D:/model_bert_copy/models/xlmroberta_gold_v1")


def main():
    # --------------------
    # Tokenizer
    # --------------------
    tokenizer = load_tokenizer()

    # --------------------
    # Dataset
    # --------------------
    train_dataset = SentimentDataset(
        csv_path=r"D:/model_bert_copy/data/processed/train.csv",
        tokenizer=tokenizer,
        max_length=128
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )

    # --------------------
    # Model
    # --------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    ).to(DEVICE)

    # --------------------
    # Optimizer
    # --------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.01
    )

    # --------------------
    # Class Weights (safe)
    # --------------------
    labels_all = torch.tensor(train_dataset.labels)
    class_counts = torch.bincount(labels_all, minlength=3).float()

    # avoid divide-by-zero
    class_counts = torch.clamp(class_counts, min=1.0)

    class_weights = class_counts.sum() / class_counts
    class_weights = class_weights.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # --------------------
    # Scheduler
    # --------------------
    epochs = 3
    total_steps = len(train_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # --------------------
    # Training
    # --------------------
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if step % 50 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Step {step}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        print(
            f"\nEpoch {epoch+1} completed | "
            f"Avg Loss: {epoch_loss / len(train_loader):.4f}\n"
        )

    # --------------------
    # Save Model
    # --------------------
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print(f"✅ Training complete. Model saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
