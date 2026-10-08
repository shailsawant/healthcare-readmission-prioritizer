from pathlib import Path
import json

import joblib
import pandas as pd


project_root = Path(__file__).resolve().parent.parent

train_file = project_root / "data" / "train_data.csv"
validation_file = (
    project_root / "data" / "validation_data.csv"
)

selected_model_file = (
    project_root / "models" / "random_forest.joblib"
)

deployment_model_file = (
    project_root / "models" / "readmission_model.joblib"
)

config_file = (
    project_root / "models" / "model_config.json"
)

train_df = pd.read_csv(train_file, low_memory=False)
validation_df = pd.read_csv(
    validation_file,
    low_memory=False,
)

development_df = pd.concat(
    [train_df, validation_df],
    ignore_index=True,
)

target_column = "readmitted_30_days"
identifier_columns = ["encounter_id", "patient_nbr"]

X_development = development_df.drop(
    columns=identifier_columns + [target_column]
)

y_development = development_df[target_column]

categorical_id_columns = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

for column in categorical_id_columns:
    X_development[column] = (
        X_development[column].astype(str)
    )

# Load the selected Random Forest pipeline
pipeline = joblib.load(selected_model_file)

print("Training final deployment model...")
pipeline.fit(X_development, y_development)

joblib.dump(pipeline, deployment_model_file)

model_config = {
    "model_version": "1.0.0",
    "model_type": "Random Forest",
    "medium_threshold": 0.45,
    "high_threshold": 0.55,
    "target": "30-day hospital readmission",
    "output_name": "risk_score",
    "clinical_use": False,
    "disclaimer": (
        "Educational prototype only. "
        "Not intended for clinical use."
    ),
}

with open(config_file, "w", encoding="utf-8") as file:
    json.dump(model_config, file, indent=4)

model_size_mb = (
    deployment_model_file.stat().st_size
    / (1024 * 1024)
)

print("Training encounters:", len(development_df))
print(
    "Training patients:",
    development_df["patient_nbr"].nunique(),
)
print(
    "Positive rate:",
    round(y_development.mean() * 100, 2),
)

print("Model saved to:", deployment_model_file)
print("Model size:", round(model_size_mb, 2), "MB")
print("Configuration saved to:", config_file)