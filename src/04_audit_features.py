from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parent.parent
data_file = project_root / "data" / "prepared_data.csv"

df = pd.read_csv(data_file, low_memory=False)

identifier_columns = ["encounter_id", "patient_nbr"]
target_column = "readmitted_30_days"

feature_df = df.drop(
    columns=identifier_columns + [target_column]
)

numeric_columns = feature_df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = feature_df.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("TOTAL MODEL FEATURES:", feature_df.shape[1])
print("NUMERIC FEATURES:", len(numeric_columns))
print("CATEGORICAL FEATURES:", len(categorical_columns))

print("\nNUMERIC COLUMNS")
for column in numeric_columns:
    print(column)

print("\nCATEGORICAL COLUMNS AND UNIQUE VALUES")
for column in categorical_columns:
    print(f"{column}: {feature_df[column].nunique()}")

print("\nCONSTANT COLUMNS")
constant_columns = [
    column
    for column in feature_df.columns
    if feature_df[column].nunique() <= 1
]

print(constant_columns)

print("\nHIGH-CARDINALITY CATEGORICAL COLUMNS")
for column in categorical_columns:
    unique_count = feature_df[column].nunique()

    if unique_count > 50:
        print(f"{column}: {unique_count}")