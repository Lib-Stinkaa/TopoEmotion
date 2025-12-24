# Topological Features Comparison

LOSO cross-validation on 8-signal fusion, 30 subjects.

**Persistence Landscape Code**: `examples/landscape_features_loso.py`
**Traditional Features Code**: `examples/traditional_features_loso.py`
**Betti Curves Code**: `examples/betti_curves_loso.py`

## Valence Classification

| Feature Type | Accuracy | F1 Score |
|--------------|----------|----------|
| **Persistence Landscapes (Ours)** | **0.7653 ± 0.0839** | **0.8594 ± 0.0752** |
| Traditional Time-Frequency | 0.5953 ± 0.0642 | 0.5869 ± 0.0808 |
| Betti Curves | TBD | TBD |

## Arousal Classification

| Feature Type | Accuracy | F1 Score |
|--------------|----------|----------|
| **Persistence Landscapes (Ours)** | **0.7010 ± 0.0759** | **0.8089 ± 0.0827** |
| Traditional Time-Frequency | 0.5753 ± 0.0495 | 0.5740 ± 0.0574 |
| Betti Curves | TBD | TBD |

**Key Insight**: Persistence landscape features significantly outperform traditional time-frequency features, demonstrating the advantage of topological data analysis for emotion recognition. Betti curves provide an alternative topological representation for comparison.
