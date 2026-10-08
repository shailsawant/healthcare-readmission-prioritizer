from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


project_root = Path(__file__).resolve().parent.parent
train_file = project_root / "data" / "train_data.csv"
# test_file = project_root / "data" / "test_data.csv"
validation_file = project_root / "data" / "validation_data.csv"
models_folder = project_root / "models"

train_df = pd.read_csv(train_file, low_memory=False)
validation_df = pd.read_csv(validation_file, low_memory=False)

target_column = "readmitted_30_days"
identifier_columns = ["encounter_id", "patient_nbr"]

X_train = train_df.drop(
    columns=identifier_columns + [target_column]
)
y_train = train_df[target_column]

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
    X_train[column] = X_train[column].astype(str)
    X_validation[column] = X_validation[column].astype(str)

numeric_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

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
            OneHotEncoder(handle_unknown="ignore"),
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

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),
    "Linear SVM": LinearSVC(
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    ),
}

results = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_validation)

    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(X_validation)[:, 1]
    else:
        scores = pipeline.decision_function(X_validation)

    accuracy = accuracy_score(y_validation, predictions)
    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )
    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )
    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0,
    )
    roc_auc = roc_auc_score(y_validation, scores)

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc,
        }
    )

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
    )

    joblib.dump(
        pipeline,
        models_folder / f"{safe_name}.joblib",
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_validation, predictions))

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["Recall", "F1"],
    ascending=False,
)

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

print(
    "\nHighest recall model:",
    results_df.iloc[0]["Model"],
)