from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


project_root = Path(__file__).resolve().parent.parent
validation_file = (
    project_root / "data" / "validation_data.csv"
)
model_file = (
    project_root / "models" / "random_forest.joblib"
)

validation_df = pd.read_csv(
    validation_file,
    low_memory=False,
)

target_column = "readmitted_30_days"
identifier_columns = ["encounter_id", "patient_nbr"]

X_validation = validation_df.drop(
    columns=identifier_columns + [target_column]
)
y_validation = validation_df[target_column]

categorical_id_columns = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

for column in categorical_id_columns:
    X_validation[column] = (
        X_validation[column].astype(str)
    )

pipeline = joblib.load(model_file)

probabilities = pipeline.predict_proba(
    X_validation
)[:, 1]

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]

results = []

for threshold in thresholds:
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        predictions,
    ).ravel()

    results.append(
        {
            "Threshold": threshold,
            "Precision": precision_score(
                y_validation,
                predictions,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_validation,
                predictions,
                zero_division=0,
            ),
            "F1": f1_score(
                y_validation,
                predictions,
                zero_division=0,
            ),
            "False_Positives": fp,
            "False_Negatives": fn,
            "True_Positives": tp,
        }
    )

results_df = pd.DataFrame(results)

print("RANDOM FOREST THRESHOLD COMPARISON")
print(results_df.round(4).to_string(index=False))