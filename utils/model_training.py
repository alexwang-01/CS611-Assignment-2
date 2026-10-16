import pickle
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import numpy as np

from config import GOLD_DIRECTORY, MODEL_DIRECTORY, MODEL_FILE, config, feature_cols


def main():
    model_dataset_filepath = GOLD_DIRECTORY / 'model_dataset' / 'gold_model_dataset.parquet'
    data_pdf = pd.read_parquet(model_dataset_filepath)
    data_pdf['feature_snapshot_date'] = pd.to_datetime(data_pdf['feature_snapshot_date'])
    data_pdf = data_pdf.sort_values(['feature_snapshot_date', 'loan_id']).reset_index(drop=True)

    # Split by application date
    train_pdf = data_pdf[
        (data_pdf["feature_snapshot_date"] >= config["train_start_date"])
        & (data_pdf["feature_snapshot_date"] < config["validation_start_date"])
    ]
    validation_pdf = data_pdf[
        (data_pdf["feature_snapshot_date"] >= config["validation_start_date"])
        & (data_pdf["feature_snapshot_date"] < config["test_start_date"])
    ]
    test_pdf = data_pdf[
        (data_pdf["feature_snapshot_date"] >= config["test_start_date"])
        & (data_pdf["feature_snapshot_date"] <= config["test_end_date"])
    ]

    X_train = train_pdf[feature_cols]
    y_train = train_pdf["label"]
    X_validation = validation_pdf[feature_cols]
    y_validation = validation_pdf["label"]
    X_test = test_pdf[feature_cols]
    y_test = test_pdf["label"]

    print("Train:", X_train.shape[0], "label rate:", round(y_train.mean(), 3))
    print("Validation:", X_validation.shape[0], "label rate:", round(y_validation.mean(), 3))
    print("Test / OOT:", X_test.shape[0], "label rate:", round(y_test.mean(), 3))

    # Logistic Regression: fit imputation and scaling on Train
    imputer = SimpleImputer(strategy="median")
    transformer_imputer = imputer.fit(X_train)

    X_train_imputed = transformer_imputer.transform(X_train)
    X_validation_imputed = transformer_imputer.transform(X_validation)
    X_test_imputed = transformer_imputer.transform(X_test)

    scaler = StandardScaler()
    transformer_stdscaler = scaler.fit(X_train_imputed)

    X_train_processed = transformer_stdscaler.transform(X_train_imputed)
    X_validation_processed = transformer_stdscaler.transform(X_validation_imputed)
    X_test_processed = transformer_stdscaler.transform(X_test_imputed)

    # XGBoost uses the original numeric features, including missing values

    lr_clf = LogisticRegression(max_iter=2000, random_state=config["random_state"])
    lr_clf.fit(X_train_processed, y_train)

    xgb_clf = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=config["random_state"],
        n_jobs=2,
    )
    xgb_clf.fit(X_train, y_train)

    lr_train_scores = lr_clf.predict_proba(X_train_processed)[:, 1]
    lr_validation_scores = lr_clf.predict_proba(X_validation_processed)[:, 1]
    xgb_train_scores = xgb_clf.predict_proba(X_train)[:, 1]
    xgb_validation_scores = xgb_clf.predict_proba(X_validation)[:, 1]

    comparison_pdf = pd.DataFrame([
        {
            "model": "Logistic Regression",
            "train_auc": roc_auc_score(y_train, lr_train_scores),
            "validation_auc": roc_auc_score(y_validation, lr_validation_scores),
        },
        {
            "model": "XGBoost",
            "train_auc": roc_auc_score(y_train, xgb_train_scores),
            "validation_auc": roc_auc_score(y_validation, xgb_validation_scores),
        },
    ])
    print(comparison_pdf.to_string(index=False))

    # Select using validation AUC only
    best_index = comparison_pdf["validation_auc"].idxmax()
    best_model_name = comparison_pdf.loc[best_index, "model"]

    if best_model_name == "Logistic Regression":
        best_model = lr_clf
        validation_scores = lr_validation_scores
        test_scores = best_model.predict_proba(X_test_processed)[:, 1]
    else:
        best_model = xgb_clf
        validation_scores = xgb_validation_scores
        test_scores = best_model.predict_proba(X_test)[:, 1]

    train_auc_score = float(comparison_pdf.loc[best_index, "train_auc"])
    validation_auc_score = float(comparison_pdf.loc[best_index, "validation_auc"])
    test_auc_score = roc_auc_score(y_test, test_scores)

    print("Selected model:", best_model_name)
    print("Train AUC:", round(train_auc_score, 4))
    print("Validation AUC:", round(validation_auc_score, 4))
    print("Test / OOT AUC:", round(test_auc_score, 4))

    model_artefact = {}
    model_artefact["model"] = best_model
    model_artefact["model_name"] = best_model_name
    model_artefact["model_version"] = config["model_version"]
    model_artefact["feature_columns"] = feature_cols
    model_artefact["preprocessing_transformers"] = {}

    if best_model_name == "Logistic Regression":
        model_artefact["preprocessing_transformers"]["imputer"] = transformer_imputer
        model_artefact["preprocessing_transformers"]["stdscaler"] = transformer_stdscaler

    model_artefact["data_dates"] = config
    model_artefact["data_stats"] = {
        "X_train": X_train.shape[0],
        "X_validation": X_validation.shape[0],
        "X_test": X_test.shape[0],
        "y_train": float(y_train.mean()),
        "y_validation": float(y_validation.mean()),
        "y_test": float(y_test.mean()),
    }
    model_artefact["results"] = {
        "auc_train": train_auc_score,
        "auc_validation": validation_auc_score,
        "auc_test": float(test_auc_score),
    }
    model_artefact["model_comparison"] = comparison_pdf.to_dict(orient="records")
    model_artefact["hp_params"] = best_model.get_params()
    # Store a validation reference for monthly stability monitoring
    score_bins = np.linspace(0, 1, 11)
    reference_counts, _ = np.histogram(validation_scores, bins=score_bins)
    model_artefact['score_bins'] = score_bins.tolist()
    model_artefact['reference_score_counts'] = reference_counts.tolist()
    model_artefact['reference_missing_fraction'] = float(X_validation.isna().mean().mean())

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, 'wb') as file:
        pickle.dump(model_artefact, file)
    print('Model saved to:', MODEL_FILE)


if __name__ == '__main__':
    main()
