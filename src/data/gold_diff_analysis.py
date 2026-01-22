import pandas as pd

def main():
    v1 = pd.read_csv("data/gold/v1/train.csv")
    v2 = pd.read_csv("data/gold/v2/train.csv")

    changed = v2[v2["original_label"] != v2["final_label"]]

    print("🔁 Total changed samples:", len(changed))
    print("\n📌 Top rules applied:")
    print(changed["rule_applied"].value_counts())

    print("\n📌 Label transitions:")
    print(
        changed.groupby(["original_label", "final_label"])
        .size()
        .sort_values(ascending=False)
    )

    print("\n📌 Sample changes:")
    print(
        changed[["text", "original_label", "final_label", "rule_applied"]]
        .head(5)
        .to_string(index=False)
    )

if __name__ == "__main__":
    main()
