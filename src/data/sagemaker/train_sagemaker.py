# src/sagemaker/train_sagemaker.py

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from src.features.tokenizer import load_tokenizer
from src.data.loader import SentimentDataset

MODEL_NAME = "xlm-roberta-base"

def main():
    # SageMaker paths
    train_path = os.environ.get("SM_CHANNEL_TRAIN")
    val_path   = os.environ.get("SM_CHANNEL_VAL")
    model_dir  = os.environ.get("SM_MODEL_DIR")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    tokenizer = load_tokenizer()

    train_dataset = SentimentDataset(
        csv_path=os.path.join(train_path, "train.csv"),
        tokenizer=tokenizer,
        max_length=128
    )

    val_dataset = SentimentDataset(
        csv_path=os.path.join(val_path, "val.csv"),
        tokenizer=tokenizer,
        max_length=128
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # -------- Training loop --------
    EPOCHS = 3

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

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

            total_loss += loss.item()

            if step % 100 == 0:
                print(f"Epoch {epoch+1} | Step {step} | Loss {loss.item():.4f}")

        print(f"\nEpoch {epoch+1} completed | Avg loss {total_loss/len(train_loader):.4f}\n")

    # -------- Save model --------
    print("Saving model to:", model_dir)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print("✅ Training finished and model saved.")

if __name__ == "__main__":
    main()
