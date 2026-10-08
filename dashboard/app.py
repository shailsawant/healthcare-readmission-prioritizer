from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


project_root = Path(__file__).resolve().parent.parent
model_file = project_root / "models" / "readmission_model.joblib"
config_file = project_root / "models" / "model_config.json"


@st.cache_resource
def load_model():
    return joblib.load(model_file)


@st.cache_data
def load_config():
    with open(config_file, "r", encoding="utf-8") as file:
        return json.load(file)


model = load_model()
config = load_config()

st.set_page_config(
    page_title="Hospital Readmission Prioritizer",
    page_icon="🏥",
    layout="wide",
)

st.title("Hospital Readmission Prioritizer")

st.warning(
    "Educational prototype only. Not intended for clinical use. "
    "Do not enter names, contact details or identifiable patient information."
)

st.write(
    "Estimate a patient's relative 30-day readmission risk and "
    "prioritize post-discharge follow-up."
)

age_options = [
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
]

diagnosis_options = [
    "Circulatory",
    "Diabetes",
    "Respiratory",
    "Digestive",
    "Genitourinary",
    "Musculoskeletal",
    "Neoplasm",
    "Injury",
    "Other",
    "Unknown",
]

medication_statuses = [
    "No",
    "Steady",
    "Up",
    "Down",
]

with st.form("assessment_form"):
    st.subheader("Patient and Encounter Information")

    column_1, column_2, column_3 = st.columns(3)

    with column_1:
        race = st.selectbox(
            "Race",
            [
                "Caucasian",
                "AfricanAmerican",
                "Asian",
                "Hispanic",
                "Other",
                "Unknown",
            ],
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Unknown/Invalid"],
        )

        age = st.selectbox(
            "Age range",
            age_options,
            index=6,
        )

        admission_type_id = st.selectbox(
            "Admission type",
            options=["1", "2", "3", "5", "6", "7", "8"],
            format_func=lambda value: {
                "1": "Emergency",
                "2": "Urgent",
                "3": "Elective",
                "5": "Not available",
                "6": "NULL",
                "7": "Trauma center",
                "8": "Not mapped",
            }[value],
        )

    with column_2:
        discharge_disposition_id = st.selectbox(
            "Discharge destination",
            options=["1", "2", "3", "4", "6", "7", "8"],
            format_func=lambda value: {
                "1": "Home",
                "2": "Short-term hospital",
                "3": "Skilled nursing facility",
                "4": "Intermediate care facility",
                "6": "Home health service",
                "7": "Left against medical advice",
                "8": "Home with home IV service",
            }[value],
        )

        admission_source_id = st.selectbox(
            "Admission source",
            options=["1", "2", "3", "4", "5", "6", "7", "9"],
            format_func=lambda value: {
                "1": "Physician referral",
                "2": "Clinic referral",
                "3": "HMO referral",
                "4": "Hospital transfer",
                "5": "Skilled nursing transfer",
                "6": "Other facility transfer",
                "7": "Emergency room",
                "9": "Information unavailable",
            }[value],
        )

        time_in_hospital = st.number_input(
            "Days in hospital",
            min_value=1,
            max_value=14,
            value=4,
        )

        num_medications = st.number_input(
            "Number of medications",
            min_value=0,
            max_value=100,
            value=15,
        )

    with column_3:
        num_lab_procedures = st.number_input(
            "Number of laboratory procedures",
            min_value=0,
            max_value=150,
            value=40,
        )

        num_procedures = st.number_input(
            "Number of procedures",
            min_value=0,
            max_value=10,
            value=1,
        )

        number_diagnoses = st.number_input(
            "Number of diagnoses",
            min_value=1,
            max_value=20,
            value=7,
        )

        diabetesMed = st.selectbox(
            "Diabetes medication prescribed",
            ["Yes", "No"],
        )

    st.subheader("Previous Healthcare Usage")

    usage_1, usage_2, usage_3 = st.columns(3)

    with usage_1:
        number_outpatient = st.number_input(
            "Outpatient visits in previous year",
            min_value=0,
            max_value=50,
            value=0,
        )

    with usage_2:
        number_emergency = st.number_input(
            "Emergency visits in previous year",
            min_value=0,
            max_value=50,
            value=0,
        )

    with usage_3:
        number_inpatient = st.number_input(
            "Inpatient visits in previous year",
            min_value=0,
            max_value=50,
            value=0,
        )

    st.subheader("Diagnosis Groups")

    diagnosis_1, diagnosis_2, diagnosis_3 = st.columns(3)

    with diagnosis_1:
        diag_1_group = st.selectbox(
            "Primary diagnosis group",
            diagnosis_options,
        )

    with diagnosis_2:
        diag_2_group = st.selectbox(
            "Secondary diagnosis group",
            diagnosis_options,
            index=1,
        )

    with diagnosis_3:
        diag_3_group = st.selectbox(
            "Additional diagnosis group",
            diagnosis_options,
            index=8,
        )

    with st.expander("Medication details"):
        st.caption(
            "Select whether each medicine was unchanged, increased "
            "or decreased during the encounter."
        )

        medication_names = [
            "metformin",
            "repaglinide",
            "nateglinide",
            "chlorpropamide",
            "glimepiride",
            "acetohexamide",
            "glipizide",
            "glyburide",
            "tolbutamide",
            "pioglitazone",
            "rosiglitazone",
            "acarbose",
            "miglitol",
            "troglitazone",
            "tolazamide",
            "insulin",
            "glyburide-metformin",
            "glipizide-metformin",
            "glimepiride-pioglitazone",
            "metformin-rosiglitazone",
            "metformin-pioglitazone",
        ]

        medication_values = {}

        medication_columns = st.columns(3)

        for index, medication in enumerate(medication_names):
            with medication_columns[index % 3]:
                default_index = 1 if medication == "insulin" else 0

                medication_values[medication] = st.selectbox(
                    medication.replace("-", " ").title(),
                    medication_statuses,
                    index=default_index,
                    key=medication,
                )

        change = st.selectbox(
            "Was any diabetes medication changed?",
            ["No", "Ch"],
        )

    submitted = st.form_submit_button(
        "Calculate Follow-up Priority",
        type="primary",
    )


if submitted:
    input_data = {
        "race": race,
        "gender": gender,
        "age": age,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "change": change,
        "diabetesMed": diabetesMed,
        "diag_1_group": diag_1_group,
        "diag_2_group": diag_2_group,
        "diag_3_group": diag_3_group,
    }

    input_data.update(medication_values)

    input_df = pd.DataFrame([input_data])

    risk_score = float(
        model.predict_proba(input_df)[0][1]
    )

    medium_threshold = config["medium_threshold"]
    high_threshold = config["high_threshold"]

    if risk_score >= high_threshold:
        priority = "HIGH"
        action = "Human follow-up recommended."
        st.error(f"Priority: {priority}")

    elif risk_score >= medium_threshold:
        priority = "MEDIUM"
        action = "Automated guidance or secondary screening recommended."
        st.warning(f"Priority: {priority}")

    else:
        priority = "LOW"
        action = "Routine discharge pathway."
        st.success(f"Priority: {priority}")

    st.metric(
        "Model risk score",
        f"{risk_score:.3f}",
    )

    st.write(action)

    st.caption(
        f"Model version: {config['model_version']}. "
        "The score is a prioritization signal, not a diagnosis "
        "or calibrated medical probability."
    )