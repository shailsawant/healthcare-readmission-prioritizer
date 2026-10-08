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

# First split: 70% training and 30% temporary data
first_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42,
)

train_indexes, temporary_indexes = next(
    first_split.split(df, groups=groups)
)

train_df = df.iloc[train_indexes].copy()
temporary_df = df.iloc[temporary_indexes].copy()

# Second split: divide temporary data equally
temporary_groups = temporary_df["patient_nbr"]

second_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42,
)

validation_indexes, test_indexes = next(
    second_split.split(
        temporary_df,
        groups=temporary_groups,
    )
)

validation_df = temporary_df.iloc[validation_indexes].copy()
test_df = temporary_df.iloc[test_indexes].copy()

train_file = project_root / "data" / "train_data.csv"
validation_file = project_root / "data" / "validation_data.csv"
test_file = project_root / "data" / "test_data.csv"

train_df.to_csv(train_file, index=False)
validation_df.to_csv(validation_file, index=False)
test_df.to_csv(test_file, index=False)

train_patients = set(train_df["patient_nbr"])
validation_patients = set(validation_df["patient_nbr"])
test_patients = set(test_df["patient_nbr"])

print("TRAINING")
print("Encounters:", len(train_df))
print("Patients:", len(train_patients))
print("Positive rate:", round(
    train_df["readmitted_30_days"].mean() * 100, 2
))

print("\nVALIDATION")
print("Encounters:", len(validation_df))
print("Patients:", len(validation_patients))
print("Positive rate:", round(
    validation_df["readmitted_30_days"].mean() * 100, 2
))

print("\nFINAL TEST")
print("Encounters:", len(test_df))
print("Patients:", len(test_patients))
print("Positive rate:", round(
    test_df["readmitted_30_days"].mean() * 100, 2
))

print("\nPATIENT OVERLAP")
print(
    "Train and validation:",
    len(train_patients & validation_patients),
)
print(
    "Train and test:",
    len(train_patients & test_patients),
)
print(
    "Validation and test:",
    len(validation_patients & test_patients),
)