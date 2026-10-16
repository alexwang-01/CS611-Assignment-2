import argparse
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import GOLD_DIRECTORY, MODEL_FILE, config


def plot_monitoring(output_directory):
    files = sorted(output_directory.glob('monitoring_*.parquet'))
    df = pd.concat([pd.read_parquet(filepath) for filepath in files], ignore_index=True)
    df = df[df['model_version'] == config['model_version']].sort_values('snapshot_date')
    months = pd.to_datetime(df['snapshot_date']).dt.strftime('%b\n%Y')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.suptitle('Loan model monitoring | Historical test cohorts', fontsize=16, fontweight='bold')

    axes[0].plot(months, df['auc'], marker='o', color='#2563eb', label='Monthly AUC')
    axes[0].axhline(df['reference_validation_auc'].iloc[0], linestyle='--', color='#64748b', label='Validation AUC')
    axes[0].set(title='Model performance', ylabel='ROC AUC', ylim=(0, 1))
    axes[0].legend(frameon=False, loc='lower left')

    axes[1].plot(months, df['score_psi'], marker='o', color='#2563eb', label='Monthly PSI')
    axes[1].axhline(0.1, linestyle=':', color='#64748b', label='0.10: review')
    axes[1].axhline(0.25, linestyle='--', color='#64748b', label='0.25: investigate')
    axes[1].set(title='Prediction stability vs validation', ylabel='Score PSI', ylim=(0, None))
    axes[1].legend(frameon=False, loc='center left')

    axes[2].plot(months, df['missing_fraction'] * 100, marker='o', color='#2563eb', label='Missing values (% of feature values)')
    axes[2].plot(months, df['clickstream_missing_fraction'] * 100, marker='s', color='#d97706', label='No clickstream (% of loans)')
    axes[2].axhline(df['reference_missing_fraction'].iloc[0] * 100, linestyle='--', color='#64748b', label='Validation missing values (%)')
    axes[2].set(title='Input completeness', ylabel='Percent', ylim=(0, 105))
    axes[2].legend(frameon=False, fontsize=9, loc='lower left', bbox_to_anchor=(0, 0.08))

    for ax in axes:
        ax.grid(axis='y', alpha=0.2)
        ax.set_xlabel('Application month')
    fig.text(0.03, 0.015, 'Source: Gold model_monitoring. Labels: DPD >= 30 at MOB 6. PSI: 10 fixed score bins; validation reference.', fontsize=9, color='#475569')
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(output_directory / 'model_monitoring.png', dpi=160)
    plt.close(fig)


def main(snapshotdate):
    # Read predictions and the stored validation reference
    with open(MODEL_FILE, 'rb') as file:
        model_artefact = pickle.load(file)
    suffix = snapshotdate.replace('-', '_') + '.parquet'
    predictions_pdf = pd.read_parquet(GOLD_DIRECTORY / 'model_predictions' / ('predictions_' + suffix))
    feature_pdf = pd.read_parquet(GOLD_DIRECTORY / 'feature_store' / ('gold_feature_store_' + suffix))

    # This retrospective evaluation uses the outcomes observed six months later
    label_date = pd.Timestamp(snapshotdate) + pd.DateOffset(months=6)
    label_partition = 'gold_label_store_' + label_date.strftime('%Y_%m_%d') + '.parquet'
    label_pdf = pd.read_parquet(GOLD_DIRECTORY / 'label_store' / label_partition)
    label_pdf = label_pdf[['loan_id', 'label', 'snapshot_date']].rename(columns={'snapshot_date': 'label_snapshot_date'})
    df = predictions_pdf.merge(label_pdf, on='loan_id', how='left', validate='one_to_one')
    labelled = df.dropna(subset=['label'])
    auc = roc_auc_score(labelled['label'], labelled['model_prediction']) if labelled['label'].nunique() == 2 else np.nan

    # Add half a count per bin before normalising to avoid log(0)
    counts, _ = np.histogram(df['model_prediction'], bins=model_artefact['score_bins'])
    reference_counts = np.asarray(model_artefact['reference_score_counts'])
    actual = (counts + 0.5) / (counts.sum() + 0.5 * len(counts))
    expected = (reference_counts + 0.5) / (reference_counts.sum() + 0.5 * len(reference_counts))
    psi = np.sum((actual - expected) * np.log(actual / expected))
    clickstream_columns = [
        'fe_1', 'fe_2', 'fe_3', 'fe_4', 'fe_5', 'fe_6', 'fe_7', 'fe_8', 'fe_9', 'fe_10',
        'fe_11', 'fe_12', 'fe_13', 'fe_14', 'fe_15', 'fe_16', 'fe_17', 'fe_18', 'fe_19', 'fe_20',
    ]
    result = {
        'snapshot_date': pd.Timestamp(snapshotdate),
        'label_snapshot_date': label_date,
        'model_version': model_artefact['model_version'],
        'model_name': model_artefact['model_name'],
        'evaluation_type': config['evaluation_type'],
        'prediction_rows': len(df),
        'labelled_rows': len(labelled),
        'label_coverage': len(labelled) / len(df),
        'auc': auc,
        'score_psi': float(psi),
        'missing_fraction': feature_pdf[model_artefact['feature_columns']].isna().mean().mean(),
        'clickstream_missing_fraction': feature_pdf[clickstream_columns].isna().all(axis=1).mean(),
        'reference_validation_auc': model_artefact['results']['auc_validation'],
        'reference_missing_fraction': model_artefact['reference_missing_fraction'],
    }
    output_directory = GOLD_DIRECTORY / 'model_monitoring'
    output_directory.mkdir(parents=True, exist_ok=True)
    filepath = output_directory / ('monitoring_' + suffix)
    pd.DataFrame([result]).to_parquet(filepath, index=False)
    plot_monitoring(output_directory)
    print('Saved:', filepath, '| AUC:', round(auc, 4), '| PSI:', round(psi, 4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshotdate', required=True)
    args = parser.parse_args()
    main(args.snapshotdate)
