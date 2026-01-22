# import pandas as pd
# from sklearn.model_selection import train_test_split

# # INPUT_PATH = "D:/model_bert_copy/data/processed/cleaned.csv"
# INPUT_PATH = r"D:\model_bert_copy\data\raw\clean_updated.csv"
# OUTPUT_DIR = "data/processed"

# def main():
#     df = pd.read_csv(INPUT_PATH)

#     train, temp = train_test_split(
#         df,
#         test_size=0.30,
#         stratify=df["label"],
#         random_state=42
#     )

#     val, test = train_test_split(
#         temp,
#         test_size=0.50,
#         stratify=temp["label"],
#         random_state=42
#     )

#     train.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
#     val.to_csv(f"{OUTPUT_DIR}/val.csv", index=False)
#     test.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

#     print("✅ Dataset split completed")
#     print("Train:", len(train))
#     print("Val:", len(val))
#     print("Test:", len(test))

# if __name__ == "__main__":
#     main()
# splitter.py
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

INPUT_PATH = r"D:/model_bert_copy/data/raw/clean_updated.csv"
OUTPUT_DIR = Path("D:/model_bert_copy/data/processed")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- FIX: handle encoding ----
    try:
        df = pd.read_csv(INPUT_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        print("⚠️ UTF-8 failed, falling back to latin1 encoding")
        df = pd.read_csv(INPUT_PATH, encoding="latin1")

    # Basic sanity
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    # ---- Stratified split (VERY IMPORTANT) ----
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["label"]
    )

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False, encoding="utf-8")

    print("✅ Dataset split complete")
    print(f"Train size: {len(train_df)}")
    print(f"Val size  : {len(val_df)}")
    print(f"Test size : {len(test_df)}")

if __name__ == "__main__":
    main()
