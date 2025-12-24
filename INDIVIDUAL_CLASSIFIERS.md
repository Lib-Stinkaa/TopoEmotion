# Individual Classifier Performance

LOSO cross-validation on 8-signal fusion, 30 subjects, using persistence landscape features.

**Code**: `examples/landscape_features_loso.py` (uses adaptive ensemble of all 10 classifiers)

**Individual Performance Code**: `sup_exp/loso_emotion_prediction/run_single_classifier.py`

## Valence Classification

| Classifier | Accuracy | F1 Score |
|------------|----------|----------|
| ExtraTrees | 0.6362 ± 0.0921 | 0.6221 ± 0.1143 |
| RandomForest | 0.6332 ± 0.0849 | 0.6206 ± 0.1032 |
| GradientBoosting | 0.6251 ± 0.0808 | 0.6158 ± 0.0930 |
| NaiveBayes | 0.6052 ± 0.1182 | 0.5554 ± 0.1303 |
| SVM | 0.6027 ± 0.0862 | 0.5882 ± 0.1023 |
| LogisticRegression | 0.5841 ± 0.0722 | 0.5788 ± 0.0771 |
| DecisionTree | 0.5820 ± 0.0485 | 0.5818 ± 0.0601 |
| MLP | 0.5798 ± 0.0844 | 0.5738 ± 0.0901 |
| Ridge | 0.5779 ± 0.0694 | 0.5724 ± 0.0739 |
| KNN | 0.5558 ± 0.0557 | 0.5613 ± 0.0555 |

## Arousal Classification

| Classifier | Accuracy | F1 Score |
|------------|----------|----------|
| RandomForest | 0.6296 ± 0.0757 | 0.6269 ± 0.0801 |
| ExtraTrees | 0.6257 ± 0.0808 | 0.6227 ± 0.0862 |
| GradientBoosting | 0.6097 ± 0.0837 | 0.6014 ± 0.0970 |
| SVM | 0.6044 ± 0.0818 | 0.5935 ± 0.0991 |
| LogisticRegression | 0.5704 ± 0.0649 | 0.5675 ± 0.0715 |
| MLP | 0.5645 ± 0.0733 | 0.5502 ± 0.0971 |
| DecisionTree | 0.5611 ± 0.0593 | 0.5624 ± 0.0583 |
| Ridge | 0.5603 ± 0.0632 | 0.5587 ± 0.0677 |
| KNN | 0.5538 ± 0.0500 | 0.5532 ± 0.0550 |
| NaiveBayes | 0.5381 ± 0.0976 | 0.4897 ± 0.1155 |

**Note**: The adaptive ensemble (Ours) in ENSEMBLE_COMPARISON.md combines these 10 classifiers with F1-weighted voting, achieving 76.53±8.39% (valence) and 70.10±7.59% (arousal), significantly outperforming any individual classifier.
