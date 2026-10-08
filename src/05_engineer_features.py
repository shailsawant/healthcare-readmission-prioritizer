from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "prepared_data.csv"
output_file = project_root / "data" / "model_data.csv"

df = pd.read_csv(input_file, low_memory=False)


def group_diagnosis(value):
    """
    Convert detailed ICD-9 diagnosis codes into broader categories.
    """

    if pd.isna(value) or str(value).strip() == "Unknown":
        return "Unknown"

    code = str(value).strip()

    # E codes generally represent external causes of injury
    if code.startswith("E"):
        return "Injury"

    # V codes represent supplementary health-status factors
    if code.startswith("V"):
        return "Other"

    try:
        code_number = float(code)
    except ValueError:
        return "Other"

    if 250 <= code_number < 251:
        return "Diabetes"

    if 390 <= code_number <= 459 or code_number == 785:
        return "Circulatory"

    if 460 <= code_number <= 519 or code_number == 786:
        return "Respiratory"

    if 520 <= code_number <= 579 or code_number == 787:
        return "Digestive"

    if 580 <= code_number <= 629 or code_number == 788:
        return "Genitourinary"

    if 710 <= code_number <= 739:
        return "Musculoskeletal"

    if 140 <= code_number <= 239:
        return "Neoplasm"

    if 800 <= code_number <= 999:
        return "Injury"

    return "Other"


# Group the three detailed diagnosis columns
for column in ["diag_1", "diag_2", "diag_3"]:
    new_column = f"{column}_group"
    df[new_column] = df[column].apply(group_diagnosis)

# Remove the original high-cardinality diagnosis codes
df = df.drop(columns=["diag_1", "diag_2", "diag_3"])

# Remove columns that contain only one value
df = df.drop(columns=["examide", "citoglipton"])

# These are category codes, not numerical measurements
categorical_id_columns = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

for column in categorical_id_columns:
    df[column] = df[column].astype(str)

df.to_csv(output_file, index=False)

print("Feature engineering completed")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Saved to:", output_file)

print("\nDiagnosis group distributions:")

for column in ["diag_1_group", "diag_2_group", "diag_3_group"]:
    print(f"\n{column}")
    print(df[column].value_counts())

print("\nRemaining missing values:", df.isna().sum().sum())