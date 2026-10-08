# Healthcare Readmission Prioritizer

A machine-learning prototype that estimates relative 30-day hospital
readmission risk and prioritizes post-discharge follow-up.

**Live application:**  
https://healthcare-readmission-prioritizer.streamlit.app/

> Educational prototype only. Not clinically validated and not intended
> for diagnosis, treatment or direct patient-care decisions.

## Problem

Hospitals and caregiver-support teams have limited capacity for
post-discharge follow-up. This project ranks encounters into three
operational priority levels:

- **HIGH:** Human follow-up recommended
- **MEDIUM:** Automated guidance or secondary screening
- **LOW:** Routine discharge pathway

The system supports workload prioritization. It does not diagnose a
condition or prescribe treatment.

## Dataset

The project uses the UCI Diabetes 130-US Hospitals dataset:

- 101,766 hospital encounters
- 71,518 patients
- 130 US hospitals and integrated delivery networks
- Data collected between 1999 and 2008
- Demographics, diagnoses, hospital utilization and medication data

Dataset source:  
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

The dataset is downloaded programmatically and is not committed to this
repository.

## Machine Learning Workflow

```mermaid
flowchart TD
    A[UCI hospital data] --> B[Data audit]
    B --> C[Missing-data handling]
    C --> D[Diagnosis grouping]
    D --> E[Patient-based split]
    E --> F[Model comparison]
    F --> G[Threshold selection]
    G --> H[Final test evaluation]
    H --> I[Streamlit application]