from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "diabetic_data.csv"
output_file = project_root / "data" / "prepared_data.csv"

df = pd.read_csv(input_file, low_memory=False)

print("Original rows:", len(df))
print("Original columns:", len(df.columns))


# --------------------------------------------------
# 1. Remove encounters where follow-up/readmission
# is not a meaningful outcome
# --------------------------------------------------

expired_or_hospice_codes = [11, 13, 14, 19, 20, 21]

df = df[
    ~df["discharge_disposition_id"].isin(expired_or_hospice_codes)
].copy()

print("Rows after removing expired/hospice cases:", len(df))


# --------------------------------------------------
# 2. Create a binary target
# --------------------------------------------------

df["readmitted_30_days"] = (
    df["readmitted"] == "<30"
).astype(int)

df = df.drop(columns=["readmitted"])


# --------------------------------------------------
# 3. Drop columns with excessive missing data
# or weak availability in a deployed system
# --------------------------------------------------

columns_to_drop = [
    "weight",
    "payer_code",
    "medical_specialty",
    "max_glu_serum",
    "A1Cresult",
]

df = df.drop(columns=columns_to_drop)


# --------------------------------------------------
# 4. Replace missing categorical values
# --------------------------------------------------

categorical_columns = df.select_dtypes(
    include=["object", "string"]
).columns

df[categorical_columns] = df[categorical_columns].fillna("Unknown")


# --------------------------------------------------
# 5. Save prepared data
# --------------------------------------------------

df.to_csv(output_file, index=False)

print("\nPrepared rows:", df.shape[0])
print("Prepared columns:", df.shape[1])
print("Saved to:", output_file)

print("\nRemaining missing values:")
print(df.isna().sum()[df.isna().sum() > 0])

print("\nBinary target distribution:")
print(df["readmitted_30_days"].value_counts())

print("\nBinary target percentage:")
print(
    (
        df["readmitted_30_days"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )
)