# Single Modality vs Multimodal Fusion

RandomForest with LOSO cross-validation, 30 subjects.

**Single Modality Code**: `sup_exp/loso_emotion_prediction/run_single_modality.py`
**Multimodal Fusion Code**: `examples/landscape_features_loso.py`

## Valence Classification

| Approach | Modality | Accuracy | F1 Score |
|----------|----------|----------|----------|
| **Multimodal Fusion (Ours)** | **8-Signal Fusion** | **0.7653 ± 0.0839** | **0.8594 ± 0.0752** |
| Single | GSR | 0.6956 ± 0.1007 | 0.7924 ± 0.0936 |
| Single | EMG_ZYGO | 0.6590 ± 0.0884 | 0.7842 ± 0.0688 |
| Single | RSP | 0.6614 ± 0.1002 | 0.7809 ± 0.0765 |
| Single | ECG | 0.6673 ± 0.0967 | 0.7807 ± 0.0775 |
| Single | BVP | 0.6525 ± 0.0952 | 0.7794 ± 0.0727 |
| Single | EMG_CORU | 0.6531 ± 0.0938 | 0.7741 ± 0.0738 |
| Single | SKT | 0.6225 ± 0.1086 | 0.7450 ± 0.1121 |
| Single | EMG_TRAP | 0.6155 ± 0.0975 | 0.7417 ± 0.0871 |

## Arousal Classification

| Approach | Modality | Accuracy | F1 Score |
|----------|----------|----------|----------|
| **Multimodal Fusion (Ours)** | **8-Signal Fusion** | **0.7010 ± 0.0759** | **0.8089 ± 0.0827** |
| Single | EMG_CORU | 0.6128 ± 0.1117 | 0.7418 ± 0.0949 |
| Single | ECG | 0.6214 ± 0.1128 | 0.7417 ± 0.0995 |
| Single | EMG_ZYGO | 0.6291 ± 0.0958 | 0.7380 ± 0.0936 |
| Single | BVP | 0.6127 ± 0.1082 | 0.7334 ± 0.0991 |
| Single | RSP | 0.6064 ± 0.1030 | 0.7292 ± 0.0979 |
| Single | EMG_TRAP | 0.5946 ± 0.1167 | 0.7149 ± 0.1096 |
| Single | GSR | 0.5886 ± 0.0882 | 0.7050 ± 0.0989 |
| Single | SKT | 0.5662 ± 0.0904 | 0.6999 ± 0.0792 |

**Key Insight**: Multimodal fusion significantly outperforms all single modalities, demonstrating the advantage of complementary information integration.
