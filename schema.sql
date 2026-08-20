-- BridgeCare Analytics — 30-Day Readmission Risk
-- Database schema (PostgreSQL)
-- Matches the columns actually produced by generate_sample_data.py
-- and consumed by train_model.py.

CREATE TABLE patients (
    patient_id INT PRIMARY KEY,
    age_group VARCHAR(10),         -- '[0-30)', '[31-60)', '[61-90)', '[90+)'
    gender VARCHAR(10),
    race VARCHAR(30)
);

CREATE TABLE admissions (
    admission_id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(patient_id),
    admission_type VARCHAR(20),           -- Emergency / Elective / Urgent
    discharge_disposition VARCHAR(20),    -- Home / Home Health / Rehab / SNF / AMA
    time_in_hospital INT,                 -- length of stay, days
    num_prior_inpatient INT,
    num_diagnoses INT,
    num_medications INT,
    num_lab_procedures INT,
    insulin VARCHAR(10),                  -- No / Steady / Up / Down
    hba1c_result VARCHAR(10),             -- None / Normal / >7 / >8
    diabetes_med VARCHAR(3),              -- Yes / No
    readmitted_30d INT                    -- target: 1 = readmitted within 30 days
);

CREATE TABLE risk_predictions (
    prediction_id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(patient_id),
    admission_id INT REFERENCES admissions(admission_id),
    risk_score FLOAT,                     -- model's predicted probability
    risk_tier VARCHAR(10),                -- Low / Moderate / High
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_risk_predictions_patient ON risk_predictions(patient_id);
CREATE INDEX idx_admissions_patient ON admissions(patient_id);
