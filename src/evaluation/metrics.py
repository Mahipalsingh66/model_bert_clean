# # metrics.py
# import torch
# import numpy as np
# from torch.utils.data import DataLoader
# from transformers import AutoModelForSequenceClassification
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# from src.features.tokenizer import load_tokenizer
# from src.data.loader import SentimentDataset

# MODEL_NAME = "xlm-roberta-base"
# DEVICE = "cpu"   # force CPU for validation

# def main():
#     tokenizer = load_tokenizer()

#     val_dataset = SentimentDataset(
#         csv_path=r"D:/model_bert_copy/data/processed/val.csv",
#         tokenizer=tokenizer,
#         max_length=128
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=8,
#         shuffle=False
#     )

#     model = AutoModelForSequenceClassification.from_pretrained(
#         MODEL_NAME,
#         num_labels=3
#     ).to(DEVICE)

#     model.eval()

#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for batch in val_loader:
#             input_ids = batch["input_ids"].to(DEVICE)
#             attention_mask = batch["attention_mask"].to(DEVICE)
#             labels = batch["labels"].to(DEVICE)

#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask
#             )

#             preds = torch.argmax(outputs.logits, dim=1)

#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     # Metrics
#     accuracy = accuracy_score(all_labels, all_preds)
#     macro_f1 = f1_score(all_labels, all_preds, average="macro")
#     cm = confusion_matrix(all_labels, all_preds)

#     print("\n✅ VALIDATION RESULTS")
#     print(f"Accuracy   : {accuracy:.4f}")
#     print(f"Macro F1   : {macro_f1:.4f}")

#     print("\n📊 Confusion Matrix (rows=true, cols=pred)")
#     print(cm)

#     print("\n📋 Classification Report")
#     print(classification_report(
#         all_labels,
#         all_preds,
#         target_names=["Negative", "Neutral", "Positive"]
#     ))

# if __name__ == "__main__":
#     main()
# metrics.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from src.features.tokenizer import load_tokenizer
from src.data.loader import SentimentDataset

MODEL_DIR = "D:/model_bert_copy/models/xlmroberta_gold_v1"
DEVICE = "cpu"

def main():
    tokenizer = load_tokenizer()

    val_dataset = SentimentDataset(
        csv_path=r"D:/model_bert_copy/data/processed/val.csv",
        tokenizer=tokenizer,
        max_length=128
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,   # larger batch OK for eval
        shuffle=False
    )

    # ✅ LOAD TRAINED MODEL
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    ).to(DEVICE)

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ---- Metrics ----
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    print("\n✅ VALIDATION RESULTS")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Macro F1   : {macro_f1:.4f}")

    print("\n📊 Confusion Matrix (rows=true, cols=pred)")
    print(cm)

    print("\n📋 Classification Report")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=["Negative", "Neutral", "Positive"],
        digits=4,
        zero_division=0
    ))

if __name__ == "__main__":
    main()
