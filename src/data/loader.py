# # loader.py
# import pandas as pd
# import torch
# from torch.utils.data import Dataset

# class SentimentDataset(Dataset):
#     def __init__(self, csv_path, tokenizer, max_length=128):
#         self.df = pd.read_csv(csv_path)
#         self.texts = self.df["text"].tolist()
#         self.labels = self.df["label"].astype(int).tolist()

#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = str(self.texts[idx])
#         label = int(self.labels[idx])

#         encoding = self.tokenizer(
#             text,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"
#         )

#         return {
#             "input_ids": encoding["input_ids"].squeeze(0),
#             "attention_mask": encoding["attention_mask"].squeeze(0),
#             "labels": torch.tensor(label, dtype=torch.long)
#         }
import pandas as pd
import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=128):
        self.df = pd.read_csv(csv_path)

        # IMPORTANT: use final_label if exists, else fallback to label
        if "final_label" in self.df.columns:
            self.texts = self.df["text"].astype(str).tolist()
            self.labels = self.df["final_label"].astype(int).tolist()
        else:
            self.texts = self.df["text"].astype(str).tolist()
            self.labels = self.df["label"].astype(int).tolist()

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }
