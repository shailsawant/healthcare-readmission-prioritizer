from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "model_data.csv"
train_file = project_root / "data" / "train_data.csv"
test_file = project_root / "data" / "test_data.csv"

df = pd.read_csv(input_file, low_memory=False)

# patient_nbr identifies which encounters belong to the same patient
groups = df["patient_nbr"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_indexes, test_indexes = next(
    splitter.split(df, groups=groups)
)

train_df = df.iloc[train_indexes].copy()
test_df = df.iloc[test_indexes].copy()

# Confirm that no patient occurs in both datasets
train_patients = set(train_df["patient_nbr"])
test_patients = set(test_df["patient_nbr"])
shared_patients = train_patients.intersection(test_patients)

train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

print("PATIENT-BASED SPLIT COMPLETED")

print("\nTraining encounters:", len(train_df))
print("Testing encounters:", len(test_df))

print("\nTraining patients:", train_df["patient_nbr"].nunique())
print("Testing patients:", test_df["patient_nbr"].nunique())
print("Patients present in both sets:", len(shared_patients))

print("\nTraining target distribution:")
print(train_df["readmitted_30_days"].value_counts())
print(
    train_df["readmitted_30_days"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nTesting target distribution:")
print(test_df["readmitted_30_days"].value_counts())
print(
    test_df["readmitted_30_days"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nSaved training data to:", train_file)
print("Saved testing data to:", test_file)