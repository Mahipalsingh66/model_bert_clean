import pandas as pd
from pathlib import Path

INPUT_PATH = "data/gold/v2/train.csv"
OUTPUT_PATH = "data/gold/v2_model/train.csv"

def main():
    df = pd.read_csv(INPUT_PATH)

    # Only keep what model needs
    df_model = df[["text", "final_label"]].rename(
        columns={"final_label": "label"}
    )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df_model.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("✅ Gold_v2 model-ready dataset created")
    print(df_model["label"].value_counts())

if __name__ == "__main__":
    main()
