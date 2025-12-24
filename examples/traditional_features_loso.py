#!/usr/bin/env python3
"""
Traditional Time-Frequency Features LOSO Validation

Uses the same framework as topological features:
- LOSO validation strategy
- 10-classifier ensemble
- Adaptive weighting based on F1 scores

Goal: Compare traditional features vs topological features
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
FEATURES_FILE = 'sup_exp/traditional_features/all_traditional_features.csv'
OUTPUT_DIR = 'sup_exp/loso_results_traditional'


def align_multimodal_features(df):
    """Align multimodal features - feature-level fusion"""
    signal_feature_map = {}
    for signal_type in SIGNAL_TYPES:
        prefix = signal_type.replace('_clean', '')
        signal_feature_map[signal_type] = [col for col in df.columns
                                           if col.startswith(prefix + '_')]

    first_signal = SIGNAL_TYPES[0]
    prefix = first_signal.replace('_clean', '')
    feature_cols = signal_feature_map[first_signal]

    base_df = df[df['signal_type'] == first_signal][
        ['subject_id', 'video_id', 'window_id', 'valence', 'arousal'] + feature_cols
    ].copy()
    base_df.columns = ['subject_id', 'video_id', 'window_id', 'valence', 'arousal'] + \
                      [f'{prefix}_{i}' for i in range(len(feature_cols))]

    for signal_type in SIGNAL_TYPES[1:]:
        prefix = signal_type.replace('_clean', '')
        feature_cols = signal_feature_map[signal_type]
        signal_df = df[df['signal_type'] == signal_type][
            ['subject_id', 'video_id', 'window_id'] + feature_cols
        ].copy()
        signal_df.columns = ['subject_id', 'video_id', 'window_id'] + \
                           [f'{prefix}_{i}' for i in range(len(feature_cols))]
        base_df = base_df.merge(signal_df, on=['subject_id', 'video_id', 'window_id'], how='inner')

    meta_columns = ['subject_id', 'video_id', 'window_id', 'valence', 'arousal']
    feature_columns = [col for col in base_df.columns if col not in meta_columns]
    base_df['features'] = base_df[feature_columns].values.tolist()

    return base_df[meta_columns + ['features']].copy()


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

    trained_classifiers = {}
    for name, clf_template in CLASSIFIERS.items():
        try:
            from sklearn.base import clone
            clf = clone(clf_template)
            clf.fit(X_train_scaled, y_train)
            trained_classifiers[name] = clf
        except:
            continue

    val_size = min(len(X_train_scaled) // 5, 500)
    X_val = X_train_scaled[-val_size:]
    y_val = y_train[-val_size:]

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

    return {
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
        with open(os.path.join(OUTPUT_DIR, 'final_summary.json'), 'w') as f:
            json.dump({'valence': valence_summary, 'arousal': arousal_summary}, f, indent=2)


if __name__ == "__main__":
    main()
