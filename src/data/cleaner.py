import pandas as pd

INPUT_PATH = r"D:/model_bert_copy/data_create/Mix_data.csv"
OUTPUT_PATH = r"D:/model_bert_copy/data/processed/cleaned.csv"

def main():
    df = pd.read_csv(INPUT_PATH)

    # Rename column safely
    if "Label" in df.columns:
        df = df.rename(columns={"Label": "label"})

    df = df[["text", "label"]]

    # Drop rows where text or label is missing
    df = df.dropna(subset=["text", "label"])

    # Clean text
    df["text"] = df["text"].astype(str).str.strip()

    # Ensure label is int
    df["label"] = df["label"].astype(int)

    # Remove duplicate texts
    df = df.drop_duplicates(subset=["text"])

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("✅ Cleaned file created:", OUTPUT_PATH)
    print("Label distribution:")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()
