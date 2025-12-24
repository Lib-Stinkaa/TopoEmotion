#!/usr/bin/env python3
"""
Persistence Landscape Features LOSO Validation

Feature-level fusion of 8-signal topological features:
- LOSO validation strategy
- 10-classifier ensemble
- Adaptive weighting based on F1 scores

Goal: Primary topological feature approach
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from joblib import Parallel, delayed
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

from config import (SIGNAL_TYPES, CLASSIFIERS, RANDOM_STATE,
                    N_PARALLEL_JOBS, VALENCE_THRESHOLD, AROUSAL_THRESHOLD)

# Configuration
FEATURES_FILE = 'sup_exp/topological_features/all_features.csv'
OUTPUT_DIR = 'sup_exp/loso_results_landscape'

META_COLS = ['window_id', 'subject_id', 'signal_type', 'video_id',
             'start_time', 'end_time', 'start_time_ms', 'end_time_ms',
             'valence', 'arousal', 'original_length', 'downsampled_length',
             'downsample_factor']


def align_multimodal_features(df):
    """Align multimodal features - feature-level fusion"""
    feature_cols = [col for col in df.columns if col not in META_COLS]
    n_features_per_modal = len(feature_cols)

    if 'valence' in feature_cols or 'arousal' in feature_cols:
        raise ValueError("Features contain labels - potential data leakage")

    df['sample_id'] = df.apply(
        lambda row: f"{row['subject_id']}_{row['video_id']}_{row['window_id']}", axis=1
    )

    modality_samples = {}
    for signal_type in SIGNAL_TYPES:
        signal_data = df[df['signal_type'] == signal_type]
        modality_samples[signal_type] = set(signal_data['sample_id'].unique())

    common_samples = set.intersection(*modality_samples.values())

    if not common_samples:
        raise ValueError("No samples with all 8 modalities")

    aligned_data = []
    for sample_id in sorted(common_samples):
        subject_id, video_id, window_id = map(int, sample_id.split('_'))
        sample_features = []
        sample_metadata = None

        for signal_type in SIGNAL_TYPES:
            signal_data = df[
                (df['signal_type'] == signal_type) &
                (df['subject_id'] == subject_id) &
                (df['video_id'] == video_id) &
                (df['window_id'] == window_id)
            ]

            if len(signal_data) == 0:
                continue

            features = signal_data[feature_cols].values[0]
            sample_features.extend(features)

            if sample_metadata is None:
                sample_metadata = {
                    'subject_id': subject_id,
                    'video_id': video_id,
                    'window_id': window_id,
                    'valence': signal_data['valence'].iloc[0],
                    'arousal': signal_data['arousal'].iloc[0]
                }

        if len(sample_features) == n_features_per_modal * len(SIGNAL_TYPES):
            aligned_data.append({**sample_metadata, 'features': sample_features})

    return pd.DataFrame(aligned_data)


def calculate_adaptive_weights(classifiers, X_val, y_val):
    """Calculate adaptive weights based on F1 scores"""
    weights = {}
    for name, clf in classifiers.items():
        try:
            y_pred = clf.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            weights[name] = max(f1, 0.001)
        except:
            weights[name] = 0.001

    total = sum(weights.values())
    if total > 0:
        weights = {name: w/total for name, w in weights.items()}
    else:
        weights = {name: 1.0/len(classifiers) for name in classifiers.keys()}

    return weights


def weighted_ensemble_predict(classifiers, weights, X):
    """Weighted ensemble prediction"""
    weighted_probs = []

    for name, clf in classifiers.items():
        try:
            if hasattr(clf, 'predict_proba'):
                prob = clf.predict_proba(X)
                prob_positive = prob[:, 1] if prob.shape[1] == 2 else prob[:, 0]
            else:
                decision = clf.decision_function(X)
                prob_positive = (decision - decision.min()) / (decision.max() - decision.min() + 1e-10)
            weighted_probs.append(weights[name] * prob_positive)
        except:
            continue

    if weighted_probs:
        ensemble_prob = np.sum(weighted_probs, axis=0)
        ensemble_pred = (ensemble_prob > 0.5).astype(int)
    else:
        ensemble_pred = np.zeros(len(X), dtype=int)

    return ensemble_pred


def train_ensemble_on_fused_features(X_train, y_train, X_test):
    """Train 10-classifier ensemble on fused features"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    val_size = min(len(X_train_scaled) // 5, 500)
    X_train_only = X_train_scaled[:-val_size] if len(X_train_scaled) > 10 else X_train_scaled
    y_train_only = y_train[:-val_size] if len(y_train) > 10 else y_train
    X_val = X_train_scaled[-val_size:] if len(X_train_scaled) > 10 else X_train_only
    y_val = y_train[-val_size:] if len(y_train) > 10 else y_train_only

    trained_classifiers = {}
    for name, clf_template in CLASSIFIERS.items():
        try:
            from sklearn.base import clone
            clf = clone(clf_template)
            clf.fit(X_train_only, y_train_only)
            trained_classifiers[name] = clf
        except:
            continue

    if not trained_classifiers:
        return None, None, None

    weights = calculate_adaptive_weights(trained_classifiers, X_val, y_val)
    y_pred = weighted_ensemble_predict(trained_classifiers, weights, X_test_scaled)

    return y_pred, trained_classifiers, weights


def process_single_fold(test_subject, fold_idx, aligned_df, target_col):
    """Process a single LOSO fold"""
    train_df = aligned_df[aligned_df['subject_id'] != test_subject]
    test_df = aligned_df[aligned_df['subject_id'] == test_subject]

    if len(train_df) == 0 or len(test_df) == 0:
        return None

    X_train = np.array(train_df['features'].tolist())
    y_train = train_df[target_col].values
    X_test = np.array(test_df['features'].tolist())
    y_test = test_df[target_col].values

    y_pred, clfs, weights = train_ensemble_on_fused_features(X_train, y_train, X_test)

    if y_pred is None:
        return None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    fold_result = {
        'fold': fold_idx + 1,
        'test_subject': int(test_subject),
        'n_train': len(train_df),
        'n_test': len(test_df),
        'accuracy': float(acc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }

    return fold_result


def loso_cross_validation_parallel(aligned_df, target='valence', n_jobs=15):
    """Parallel LOSO cross-validation"""
    subjects = sorted(aligned_df['subject_id'].unique())
    target_col = f'{target}_binary'

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_fold)(test_subject, fold_idx, aligned_df, target_col)
        for fold_idx, test_subject in enumerate(subjects)
    )

    results = [r for r in results if r is not None]
    if not results:
        return None

    accs = [r['accuracy'] for r in results]
    f1s = [r['f1'] for r in results]
    precs = [r['precision'] for r in results]
    recs = [r['recall'] for r in results]

    summary = {
        'target': target,
        'n_folds': len(subjects),
        'mean_accuracy': float(np.mean(accs)),
        'std_accuracy': float(np.std(accs)),
        'mean_f1': float(np.mean(f1s)),
        'std_f1': float(np.std(f1s)),
        'mean_precision': float(np.mean(precs)),
        'std_precision': float(np.std(precs)),
        'mean_recall': float(np.mean(recs)),
        'std_recall': float(np.std(recs))
    }

    return summary


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(FEATURES_FILE)
    aligned_df = align_multimodal_features(df)
    aligned_df['valence_binary'] = (aligned_df['valence'] > VALENCE_THRESHOLD).astype(int)
    aligned_df['arousal_binary'] = (aligned_df['arousal'] > AROUSAL_THRESHOLD).astype(int)

    valence_summary = loso_cross_validation_parallel(aligned_df, 'valence', n_jobs=N_PARALLEL_JOBS)
    arousal_summary = loso_cross_validation_parallel(aligned_df, 'arousal', n_jobs=N_PARALLEL_JOBS)

    if valence_summary:
        with open(os.path.join(OUTPUT_DIR, 'valence_results.json'), 'w') as f:
            json.dump(valence_summary, f, indent=2)

    if arousal_summary:
        with open(os.path.join(OUTPUT_DIR, 'arousal_results.json'), 'w') as f:
            json.dump(arousal_summary, f, indent=2)

    if valence_summary and arousal_summary:
        final_summary = {
            'valence': valence_summary,
            'arousal': arousal_summary
        }
        with open(os.path.join(OUTPUT_DIR, 'final_summary.json'), 'w') as f:
            json.dump(final_summary, f, indent=2)


if __name__ == "__main__":
    main()
