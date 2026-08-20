"""
generate_sample_data.py

Generates a synthetic patient/admission dataset modeled on the structure
of the UCI "Diabetes 130-US Hospitals" dataset (not real patient data).
Produces a target readmission rate of roughly 11%, matching real-world
30-day readmission benchmarks, with feature relationships strong enough
for a model to actually learn from.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N = 12000  # patients


def generate_healthcare_data(n: int = N) -> pd.DataFrame:
    age_group = RNG.choice(
        ["[0-30)", "[31-60)", "[61-90)", "[90+)"], size=n, p=[0.08, 0.37, 0.47, 0.08]
    )
    gender = RNG.choice(["Male", "Female"], size=n, p=[0.47, 0.53])
    race = RNG.choice(
        ["Caucasian", "African American", "Hispanic", "Asian", "Other"],
        size=n, p=[0.62, 0.19, 0.09, 0.05, 0.05],
    )

    admission_type = RNG.choice(
        ["Emergency", "Elective", "Urgent"], size=n, p=[0.55, 0.20, 0.25]
    )
    discharge_disposition = RNG.choice(
        ["Home", "Home Health", "Rehab", "SNF", "AMA"],
        size=n, p=[0.58, 0.18, 0.10, 0.11, 0.03],
    )

    time_in_hospital = RNG.integers(1, 15, size=n)
    num_prior_inpatient = RNG.poisson(0.6, size=n)
    num_diagnoses = RNG.integers(1, 16, size=n)
    num_medications = RNG.integers(1, 30, size=n)
    num_lab_procedures = RNG.integers(1, 100, size=n)

    insulin = RNG.choice(["No", "Steady", "Up", "Down"], size=n, p=[0.45, 0.35, 0.10, 0.10])
    hba1c_result = RNG.choice(["None", "Normal", ">7", ">8"], size=n, p=[0.5, 0.2, 0.15, 0.15])
    diabetes_med = RNG.choice(["Yes", "No"], size=n, p=[0.55, 0.45])

    emergency_admission = (admission_type == "Emergency").astype(int)

    # ---- Build a real (noisy) risk signal so the model has something to learn ----
    risk = (
        0.35 * (num_prior_inpatient / num_prior_inpatient.max())
        + 0.30 * (time_in_hospital / time_in_hospital.max())
        + 0.25 * (num_diagnoses / num_diagnoses.max())
        + 0.15 * (num_medications / num_medications.max())
        + 0.10 * emergency_admission
        + 0.08 * (discharge_disposition == "AMA")
        + 0.06 * (discharge_disposition == "Home Health")
        + 0.05 * (hba1c_result == ">8")
        + 0.04 * (age_group == "[90+)")
    )
    risk = (risk - risk.min()) / (risk.max() - risk.min())
    noise = RNG.normal(0, 0.15, size=n)
    risk_noisy = np.clip(risk + noise, 0, 1)

    # Calibrate threshold to hit ~11% positive rate
    threshold = np.quantile(risk_noisy, 0.89)
    readmitted_30d = (risk_noisy >= threshold).astype(int)

    df = pd.DataFrame({
        "patient_id": np.arange(1, n + 1),
        "age_group": age_group,
        "gender": gender,
        "race": race,
        "admission_type": admission_type,
        "discharge_disposition": discharge_disposition,
        "time_in_hospital": time_in_hospital,
        "num_prior_inpatient": num_prior_inpatient,
        "num_diagnoses": num_diagnoses,
        "num_medications": num_medications,
        "num_lab_procedures": num_lab_procedures,
        "insulin": insulin,
        "hba1c_result": hba1c_result,
        "diabetesMed": diabetes_med,
        "readmitted_30d": readmitted_30d,
    })
    return df


def main():
    df = generate_healthcare_data()
    Path("data").mkdir(exist_ok=True)
    out_path = Path("data") / "sample_readmission_data.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Created {out_path} ({len(df):,} rows, {df['readmitted_30d'].mean():.1%} readmit rate)")


if __name__ == "__main__":
    main()
