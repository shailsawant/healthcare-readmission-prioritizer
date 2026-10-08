from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parent.parent
data_file = project_root / "data" / "diabetic_data.csv"

# low_memory=False prevents mixed-type guessing in separate chunks
df = pd.read_csv(data_file, low_memory=False)

print("DATASET SIZE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nCOLUMN NAMES")
for number, column in enumerate(df.columns, start=1):
    print(number, column)

print("\nDATA TYPES")
print(df.dtypes)

print("\nSTANDARD MISSING VALUES")
missing_values = df.isna().sum()
print(missing_values[missing_values > 0].sort_values(ascending=False))

print("\nQUESTION-MARK VALUES")
question_marks = (df.astype(str) == "?").sum()
print(question_marks[question_marks > 0].sort_values(ascending=False))

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nUNIQUE ENCOUNTERS")
print(df["encounter_id"].nunique())

print("\nUNIQUE PATIENTS")
print(df["patient_nbr"].nunique())

print("\nENCOUNTERS PER PATIENT")
print(df["patient_nbr"].value_counts().describe())

print("\nTARGET DISTRIBUTION")
print(df["readmitted"].value_counts())

print("\nTARGET PERCENTAGE")
print((df["readmitted"].value_counts(normalize=True) * 100).round(2))