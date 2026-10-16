import argparse
import pickle

import pandas as pd

from config import GOLD_DIRECTORY, MODEL_FILE, config, prepare_features


def main(snapshotdate):
    snapshot_date = pd.Timestamp(snapshotdate)
    if snapshot_date not in pd.date_range(config['test_start_date'], config['test_end_date'], freq='MS'):
        raise ValueError('Prediction month must be within the test period.')

    # Load model artefact
    with open(MODEL_FILE, 'rb') as file:
        model_artefact = pickle.load(file)

    # Read application-time features without labels
    partition_name = 'gold_feature_store_' + snapshotdate.replace('-', '_') + '.parquet'
    features_pdf = pd.read_parquet(GOLD_DIRECTORY / 'feature_store' / partition_name)
    features_pdf = features_pdf.sort_values('loan_id').reset_index(drop=True)
    X_inference = prepare_features(features_pdf, model_artefact)
    scores = model_artefact['model'].predict_proba(X_inference)[:, 1]

    predictions_pdf = features_pdf[['loan_id', 'Customer_ID', 'feature_snapshot_date']].copy()
    predictions_pdf['model_version'] = model_artefact['model_version']
    predictions_pdf['model_name'] = model_artefact['model_name']
    predictions_pdf['model_prediction'] = scores
    predictions_pdf['evaluation_type'] = config['evaluation_type']

    output_directory = GOLD_DIRECTORY / 'model_predictions'
    output_directory.mkdir(parents=True, exist_ok=True)
    filepath = output_directory / ('predictions_' + snapshotdate.replace('-', '_') + '.parquet')
    predictions_pdf.to_parquet(filepath, index=False)
    print('Saved:', filepath, '| rows:', len(predictions_pdf))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshotdate', required=True)
    args = parser.parse_args()
    main(args.snapshotdate)
