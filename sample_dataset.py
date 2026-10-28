
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Original dataset
input_path = Path("data/raw/fraud_oracle.csv")

# New 1% sample dataset
output_path = Path("data/raw/fraud_oracle_1pct.csv")

# Check that the original dataset exists
if not input_path.exists():
    raise FileNotFoundError(
        f"Dataset not found: {input_path}. "
        "Check the filename and location."
    )

# Load original data
df = pd.read_csv(input_path)

print("Original dataset rows:", len(df))

# Select 1% while preserving the fraud/genuine ratio
sample, _ = train_test_split(
    df,
    train_size=0.01,
    stratify=df["FraudFound_P"],
    random_state=42
)

# Save the sample
sample.to_csv(output_path, index=False)

print("Sample dataset rows:", len(sample))
print("\nFraud distribution:")
print(sample["FraudFound_P"].value_counts())

print("\nSaved successfully to:", output_path)
