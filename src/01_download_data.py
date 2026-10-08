from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


# Download dataset 296 from the UCI Machine Learning Repository
dataset = fetch_ucirepo(id=296)

# Separate input columns and target column supplied by UCI
X = dataset.data.features
y = dataset.data.targets

# Combine them into one DataFrame
df = pd.concat([X, y], axis=1)

# Build a reliable path to the project's data folder
project_root = Path(__file__).resolve().parent.parent
output_file = project_root / "data" / "diabetic_data.csv"

# Save the dataset locally
df.to_csv(output_file, index=False)

print("Dataset downloaded successfully")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Saved to:", output_file)
print("\nTarget distribution:")
print(df["readmitted"].value_counts())