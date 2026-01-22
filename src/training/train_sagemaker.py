# import os
# import torch
# from torch.utils.data import DataLoader
# from transformers import AutoModelForSequenceClassification
# from src.features.tokenizer import load_tokenizer
# from src.data.loader import SentimentDataset

# MODEL_NAME = "xlm-roberta-base"

# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print("Using device:", device)

#     # SageMaker paths
#     train_path = os.environ["SM_CHANNEL_TRAIN"]
#     model_dir = os.environ["SM_MODEL_DIR"]

#     train_csv = os.path.join(train_path, "train_small.csv")

#     tokenizer = load_tokenizer()

#     train_dataset = SentimentDataset(
#         csv_path=train_csv,
#         tokenizer=tokenizer,
#         max_length=128
#     )

#     train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

#     model = AutoModelForSequenceClassification.from_pretrained(
#         MODEL_NAME,
#         num_labels=3
#     ).to(device)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
#     criterion = torch.nn.CrossEntropyLoss()

#     model.train()

#     for epoch in range(1):   # only 1 epoch for test
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

#             if step % 20 == 0:
#                 print(f"Step {step} | Loss: {loss.item():.4f}")

#     # Save model to SageMaker output
#     model.save_pretrained(model_dir)
#     tokenizer.save_pretrained(model_dir)

#     print("✅ Model saved to", model_dir)

# if __name__ == "__main__":
#     main()
# src/training/train_sagemaker.py

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.data.loader import SentimentDataset

MODEL_NAME = "xlm-roberta-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # SageMaker provides these paths automatically
    train_path = os.environ.get("SM_CHANNEL_TRAIN")
    val_path = os.environ.get("SM_CHANNEL_VAL")
    model_dir = os.environ.get("SM_MODEL_DIR")

    print("Train path:", train_path)
    print("Val path  :", val_path)
    print("Model dir :", model_dir)

    train_csv = os.path.join(train_path, "train.csv")
    val_csv = os.path.join(val_path, "val.csv")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = SentimentDataset(
        csv_path=train_csv,
        tokenizer=tokenizer,
        max_length=128
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # ---- class weights ----
    labels_all = torch.tensor(train_dataset.labels)
    class_counts = torch.bincount(labels_all, minlength=3).float()
    class_weights = class_counts.sum() / class_counts
    class_weights = class_weights.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    model.train()

    EPOCHS = 2   # SAFE FIRST RUN

    for epoch in range(EPOCHS):
        epoch_loss = 0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if step % 100 == 0:
                print(f"Epoch {epoch+1} | Step {step} | Loss {loss.item():.4f}")

        print(f"\nEpoch {epoch+1} finished | Avg loss {epoch_loss/len(train_loader):.4f}\n")

    # Save model + tokenizer
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print("✅ Model saved to:", model_dir)

if __name__ == "__main__":
    main()
