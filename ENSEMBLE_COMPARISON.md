# Ensemble Learning vs Deep Learning vs Traditional ML

LOSO cross-validation on 8-signal fusion, 30 subjects.

**Ours (Ensemble) Code**: `examples/landscape_features_loso.py`
**Traditional ML Code**: `sup_exp/loso_emotion_prediction/03_loso_validator.py`
**Deep Learning Code**: `examples/deep_learning_mlp.py`, `examples/deep_learning_cnn.py`, `examples/deep_learning_lstm.py`

## Valence Classification

| Model | Approach | Accuracy | F1 Score |
|-------|----------|----------|----------|
| **Ours** | **Ensemble (10 Classifiers)** | **0.7653 ± 0.0839** | **0.8594 ± 0.0752** |
| ExtraTrees | Traditional ML | 0.6362 ± 0.0921 | 0.6221 ± 0.1143 |
| RandomForest | Traditional ML | 0.6332 ± 0.0849 | 0.6206 ± 0.1032 |
| GradientBoosting | Traditional ML | 0.6251 ± 0.0808 | 0.6158 ± 0.0930 |
| CNN | Deep Learning | 0.5975 ± 0.1009 | 0.5830 ± 0.1092 |
| MLP | Deep Learning | 0.5897 ± 0.1030 | 0.5760 ± 0.1144 |
| LSTM | Deep Learning | 0.5700 ± 0.0868 | 0.5670 ± 0.0895 |

## Arousal Classification

| Model | Approach | Accuracy | F1 Score |
|-------|----------|----------|----------|
| **Ours** | **Ensemble (10 Classifiers)** | **0.7010 ± 0.0759** | **0.8089 ± 0.0827** |
| RandomForest | Traditional ML | 0.6296 ± 0.0757 | 0.6269 ± 0.0801 |
| ExtraTrees | Traditional ML | 0.6257 ± 0.0808 | 0.6227 ± 0.0862 |
| GradientBoosting | Traditional ML | 0.6097 ± 0.0837 | 0.6014 ± 0.0970 |
| MLP | Deep Learning | 0.5910 ± 0.0764 | 0.5753 ± 0.0963 |
| LSTM | Deep Learning | 0.5766 ± 0.0666 | 0.5707 ± 0.0686 |
| CNN | Deep Learning | 0.5865 ± 0.0833 | 0.5667 ± 0.1081 |

**Key Insight**: Adaptive weighted ensemble learning significantly outperforms both individual traditional ML classifiers and deep learning models, demonstrating the advantage of combining multiple classifiers.
