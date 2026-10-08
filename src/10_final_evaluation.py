from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


project_root = Path(__file__).resolve().parent.parent
test_file = project_root / "data" / "test_data.csv"
model_file = (
    project_root / "models" / "random_forest.joblib"
)
predictions_file = (
    project_root / "data" / "final_test_predictions.csv"
)

test_df = pd.read_csv(test_file, low_memory=False)

target_column = "readmitted_30_days"
identifier_columns = ["encounter_id", "patient_nbr"]

X_test = test_df.drop(
    columns=identifier_columns + [target_column]
)
y_test = test_df[target_column]

categorical_id_columns = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

for column in categorical_id_columns:
    X_test[column] = X_test[column].astype(str)

pipeline = joblib.load(model_file)

risk_scores = pipeline.predict_proba(X_test)[:, 1]

followup_threshold = 0.45

predictions = (
    risk_scores >= followup_threshold
).astype(int)

priorities = np.select(
    [
        risk_scores >= 0.55,
        risk_scores >= 0.45,
    ],
    [
        "HIGH",
        "MEDIUM",
    ],
    default="LOW",
)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test,
    predictions,
    zero_division=0,
)
recall = recall_score(
    y_test,
    predictions,
    zero_division=0,
)
f1 = f1_score(
    y_test,
    predictions,
    zero_division=0,
)
roc_auc = roc_auc_score(y_test, risk_scores)

print("FINAL TEST RESULTS")
print("Threshold:", followup_threshold)
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 score:", round(f1, 4))
print("ROC AUC:", round(roc_auc, 4))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, predictions))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, predictions))

priority_counts = pd.Series(
    priorities
).value_counts()

print("\nPRIORITY DISTRIBUTION")
print(priority_counts)

print("\nPRIORITY PERCENTAGE")
print(
    (
        priority_counts / len(priorities) * 100
    ).round(2)
)

results_df = pd.DataFrame(
    {
        "encounter_id": test_df["encounter_id"],
        "actual_readmission": y_test,
        "risk_score": risk_scores.round(4),
        "priority": priorities,
        "followup_required": predictions,
    }
)

results_df.to_csv(predictions_file, index=False)

print("\nPredictions saved to:", predictions_file)