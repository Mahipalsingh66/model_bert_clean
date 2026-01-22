import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = "data/processed/cleaned.csv"
OUTPUT_DIR = "data/processed"

def main():
    df = pd.read_csv(INPUT_PATH)

    train, temp = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=42
    )

    val, test = train_test_split(
        temp,
        test_size=0.50,
        stratify=temp["label"],
        random_state=42
    )

    train.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    val.to_csv(f"{OUTPUT_DIR}/val.csv", index=False)
    test.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

    print("✅ Dataset split completed")
    print("Train:", len(train))
    print("Val:", len(val))
    print("Test:", len(test))

if __name__ == "__main__":
    main()
