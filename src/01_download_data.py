from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


dataset = fetch_ucirepo(id=296)

# UCI separates identifiers, input features and target
ids = dataset.data.ids
X = dataset.data.features
y = dataset.data.targets

# Combine all columns into one DataFrame
df = pd.concat(
    [
        ids.reset_index(drop=True),
        X.reset_index(drop=True),
        y.reset_index(drop=True),
    ],
    axis=1,
)

project_root = Path(__file__).resolve().parent.parent
output_file = project_root / "data" / "diabetic_data.csv"

df.to_csv(output_file, index=False)

print("Dataset downloaded successfully")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Saved to:", output_file)

print("\nIdentifier columns:")
print(ids.columns.tolist())

print("\nTarget distribution:")
print(df["readmitted"].value_counts())