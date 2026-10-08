from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


project_root = Path(__file__).resolve().parent.parent
train_file = project_root / "data" / "train_data.csv"
test_file = project_root / "data" / "test_data.csv"
model_file = project_root / "models" / "logistic_regression.joblib"

train_df = pd.read_csv(train_file, low_memory=False)
test_df = pd.read_csv(test_file, low_memory=False)

target_column = "readmitted_30_days"
identifier_columns = ["encounter_id", "patient_nbr"]

X_train = train_df.drop(
    columns=identifier_columns + [target_column]
)
y_train = train_df[target_column]

X_test = test_df.drop(
    columns=identifier_columns + [target_column]
)
y_test = test_df[target_column]


# These columns contain category codes, not measured quantities.
categorical_id_columns = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

for column in categorical_id_columns:
    X_train[column] = X_train[column].astype(str)
    X_test[column] = X_test[column].astype(str)


numeric_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("Numeric features:", len(numeric_columns))
print("Categorical features:", len(categorical_columns))


numeric_pipeline = Pipeline(
    steps=[
        ("missing_values", SimpleImputer(strategy="median")),
        ("scaling", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "missing_values",
            SimpleImputer(
                strategy="constant",
                fill_value="Unknown",
            ),
        ),
        (
            "one_hot_encoding",
            OneHotEncoder(
                handle_unknown="ignore",
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_columns),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ),
    ]
)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

print("\nTraining Logistic Regression...")
pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
probabilities = pipeline.predict_proba(X_test)[:, 1]

print("\nMODEL RESULTS")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions), 4))
print("Recall:", round(recall_score(y_test, predictions), 4))
print("F1 score:", round(f1_score(y_test, predictions), 4))
print("ROC AUC:", round(roc_auc_score(y_test, probabilities), 4))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, predictions))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, predictions))

joblib.dump(pipeline, model_file)
print("Model saved to:", model_file)