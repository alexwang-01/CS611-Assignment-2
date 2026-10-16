import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIRECTORY = ROOT / 'datamart' / 'gold'
MODEL_DIRECTORY = Path(os.environ.get('MODEL_STORE', '/opt/airflow/model_store'))

config = {}
config["model_version"] = "credit_model_v1"
config["evaluation_type"] = "retrospective_time_split"
config["train_start_date"] = "2023-01-01"
config["validation_start_date"] = "2024-01-01"
config["test_start_date"] = "2024-07-01"
config["test_end_date"] = "2025-01-01"
config["random_state"] = 88

MODEL_FILE = MODEL_DIRECTORY / (config["model_version"] + ".pkl")

feature_cols = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Num_Credit_Inquiries", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Credit_History_Age_Months", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance",
    "debt_to_income_ratio", "emi_to_income_ratio", "monthly_balance_ratio",
    "delayed_payment_flag",
    "fe_1", "fe_2", "fe_3", "fe_4", "fe_5",
    "fe_6", "fe_7", "fe_8", "fe_9", "fe_10",
    "fe_11", "fe_12", "fe_13", "fe_14", "fe_15",
    "fe_16", "fe_17", "fe_18", "fe_19", "fe_20",
]


def prepare_features(df, model_artefact):
    X = df[model_artefact['feature_columns']]
    transformers = model_artefact['preprocessing_transformers']
    if model_artefact['model_name'] == 'Logistic Regression':
        X = transformers['imputer'].transform(X)
        X = transformers['stdscaler'].transform(X)
    return X
