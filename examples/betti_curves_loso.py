#!/usr/bin/env python3
"""
Betti Curves Features LOSO Validation

Alternative topological feature representation using Betti curves:
- LOSO validation strategy
- 10-classifier ensemble
- Adaptive weighting based on F1 scores

Goal: Compare with persistence landscape features
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from joblib import Parallel, delayed
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from config import (SIGNAL_TYPES, CLASSIFIERS, RANDOM_STATE,
                    N_PARALLEL_JOBS, VALENCE_THRESHOLD, AROUSAL_THRESHOLD)

# Configuration
DATA_DIR = 'sup_exp/topological_features'
OUTPUT_DIR = 'sup_exp/loso_results_betti'
N_BINS = 100


def compute_betti_curve(persistence_pairs, n_bins=100):
    """Compute Betti curve from persistence pairs"""
    if len(persistence_pairs) == 0:
        return np.zeros(n_bins)

    pairs = np.array(persistence_pairs)
    births = pairs[:, 0]
    deaths = pairs[:, 1]

    finite_mask = np.isfinite(deaths)
    if np.any(~finite_mask):
        deaths = deaths.copy()
        deaths[~finite_mask] = np.max(deaths[finite_mask]) * 1.5 if np.any(finite_mask) else 1.0

    max_epsilon = np.max(deaths)
    epsilons = np.linspace(0, max_epsilon, n_bins)
    betti_curve = np.zeros(n_bins)

    for i, eps in enumerate(epsilons):
        betti_curve[i] = np.sum((births <= eps) & (deaths > eps))

    return betti_curve


def extract_betti_features_from_dgms(dgms, n_bins=100):
    """Extract Betti curve features from persistence diagrams"""
    features = []

    h0_pairs = dgms.get('H0', np.array([]))
    h0_curve = compute_betti_curve(h0_pairs, n_bins=n_bins) if len(h0_pairs) > 0 else np.zeros(n_bins)
    features.extend(h0_curve)

    h1_pairs = dgms.get('H1', np.array([]))
    h1_curve = compute_betti_curve(h1_pairs, n_bins=n_bins) if len(h1_pairs) > 0 else np.zeros(n_bins)
    features.extend(h1_curve)

    return np.array(features)


def load_and_extract_betti_features():
    """Load persistence diagrams and extract Betti curve features"""
    all_features = []

    for signal_type in SIGNAL_TYPES:
        pkl_file = os.path.join(DATA_DIR, f'{signal_type}_persistence_diagrams.pkl')
        if not os.path.exists(pkl_file):
            continue

        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        for item in data:
            dgms = item['persistence_diagrams']
            betti_features = extract_betti_features_from_dgms(dgms, n_bins=N_BINS)

            feature_row = {
                'subject_id': item['subject_id'],
                'video_id': item['video_id'],
                'window_id': item['window_id'],
                'signal_type': signal_type,
                'valence': item['valence'],
                'arousal': item['arousal']
            }

            for i, val in enumerate(betti_features):
                feature_row[f'betti_{i}'] = val

            all_features.append(feature_row)

    return pd.DataFrame(all_features)


def align_multimodal_features(df):
    """Align multimodal Betti curve features"""
    feature_cols = [col for col in df.columns if col.startswith('betti_')]
    n_features_per_signal = len(feature_cols)

    aligned_dfs = []
    for signal_type in SIGNAL_TYPES:
        signal_df = df[df['signal_type'] == signal_type].copy()
        prefix = signal_type.replace('_clean', '')
        rename_dict = {f'betti_{i}': f'{prefix}_betti_{i}' for i in range(n_features_per_signal)}
        signal_df = signal_df.rename(columns=rename_dict)
        keep_cols = ['subject_id', 'video_id', 'window_id', 'valence', 'arousal'] + list(rename_dict.values())
        signal_df = signal_df[keep_cols]
        aligned_dfs.append(signal_df)

    result_df = aligned_dfs[0]
    for i in range(1, len(aligned_dfs)):
        result_df = result_df.merge(
            aligned_dfs[i],
            on=['subject_id', 'video_id', 'window_id', 'valence', 'arousal'],
            how='inner'
        )

    feature_columns = [col for col in result_df.columns
                      if col not in ['subject_id', 'video_id', 'window_id', 'valence', 'arousal']]
    result_df['features'] = result_df[feature_columns].values.tolist()

    return result_df[['subject_id', 'video_id', 'window_id', 'valence', 'arousal', 'features']]


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
    X_val = X_train_scaled[-val_size:]
    y_val = y_train[-val_size:]

    trained_classifiers = {}
    for name, clf_template in CLASSIFIERS.items():
        try:
            from sklearn.base import clone
            clf = clone(clf_template)
            clf.fit(X_train_scaled, y_train)
            trained_classifiers[name] = clf
        except:
            continue

    weights = calculate_adaptive_weights(trained_classifiers, X_val, y_val)
    y_pred = weighted_ensemble_predict(trained_classifiers, weights, X_test_scaled)

    return y_pred, trained_classifiers, weights


def evaluate_single_fold(fold_data):
    """Evaluate single fold"""
    test_subject, aligned_df, task = fold_data

    train_df = aligned_df[aligned_df['subject_id'] != test_subject].copy()
    test_df = aligned_df[aligned_df['subject_id'] == test_subject].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        return None

    X_train = np.array(train_df['features'].tolist())
    X_test = np.array(test_df['features'].tolist())
    y_train = train_df[f'{task}_binary'].values
    y_test = test_df[f'{task}_binary'].values

    y_pred, classifiers, weights = train_ensemble_on_fused_features(X_train, y_train, X_test)

    metrics = {
        'subject_id': test_subject,
        'n_train': len(train_df),
        'n_test': len(test_df),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    return metrics


def run_loso_evaluation(aligned_df, task='valence'):
    """Run LOSO cross-validation"""
    subjects = sorted(aligned_df['subject_id'].unique())
    fold_tasks = [(subject, aligned_df, task) for subject in subjects]

    results = Parallel(n_jobs=N_PARALLEL_JOBS)(
        delayed(evaluate_single_fold)(fold_data)
        for fold_data in fold_tasks
    )

    results = [r for r in results if r is not None]
    if not results:
        return None

    results_df = pd.DataFrame(results)

    summary = {
        'task': task,
        'n_subjects': len(results),
        'accuracy_mean': results_df['accuracy'].mean(),
        'accuracy_std': results_df['accuracy'].std(),
        'f1_mean': results_df['f1'].mean(),
        'f1_std': results_df['f1'].std(),
        'precision_mean': results_df['precision'].mean(),
        'precision_std': results_df['precision'].std(),
        'recall_mean': results_df['recall'].mean(),
        'recall_std': results_df['recall'].std()
    }

    return summary, results_df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_and_extract_betti_features()
    aligned_df = align_multimodal_features(df)

    aligned_df['valence_binary'] = (aligned_df['valence'] > VALENCE_THRESHOLD).astype(int)
    aligned_df['arousal_binary'] = (aligned_df['arousal'] > AROUSAL_THRESHOLD).astype(int)

    results = {}
    detailed_results = {}

    for task in ['valence', 'arousal']:
        summary, details = run_loso_evaluation(aligned_df, task)
        if summary:
            results[task] = summary
            detailed_results[task] = details

    summary_file = os.path.join(OUTPUT_DIR, 'summary_results.json')
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    for task, details in detailed_results.items():
        detail_file = os.path.join(OUTPUT_DIR, f'{task}_detailed_results.csv')
        details.to_csv(detail_file, index=False)


if __name__ == "__main__":
    main()
