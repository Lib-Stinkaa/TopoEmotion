# TopoEmotion: A Persistent Homology Framework for Robust Multi-Modal Physiological Emotion Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243?logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.13.1-8CAAE6?logo=scipy)](https://scipy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![NeuroKit2](https://img.shields.io/badge/NeuroKit2-0.2.7-9B59B6)](https://neuropsychology.github.io/NeuroKit/)
[![Ripser](https://img.shields.io/badge/Ripser-0.6.12-2ECC71)](https://ripser.scikit-tda.org/)

## Overview

**TopoEmotion** is a computational framework for emotion recognition from multimodal physiological signals using persistent homology theory. 

## Technical Characteristics

### Signal Modalities
- CASE dataset
- Eight-channel multimodal physiological signals: ECG, BVP, GSR, RSP, SKT, EMG_ZYGO, EMG_CORU, EMG_TRAP
- 1000Hz, 30 participants, 8 movie clips

### Computational Pipeline
- **Preprocessing Module**: Domain-specific bandpass filtering with NeuroKit2 framework
- **Embedding Optimization**: Data-driven parameter estimation via Average Mutual Information and False Nearest Neighbors algorithms
- **Topological Extraction**: Persistent homology computation with Vietoris-Rips filtration
- **Feature Representation**: Multi-resolution persistence landscape embeddings

### Methodological Validation
- Leave-One-Subject-Out cross-validation 
## Installation

```bash
git clone https://github.com/Lib-Stinkaa/TopoEmotion.git
cd TopoEmotion
pip install -r requirements.txt
```

## Quick Start

```python
from topoemotion.preprocessing import preprocess_signal, downsample_signal
from topoemotion.embedding import compute_delay, compute_dimension
from topoemotion.features import extract_features

# Signal preprocessing
signal_clean = preprocess_signal(raw_signal, 'ecg', sampling_rate=1000)
signal_ds = downsample_signal(signal_clean, 1000, 100)

# Embedding parameter optimization
delay = compute_delay(signal_ds)
dimension = compute_dimension(signal_ds, delay)

# Topological feature extraction
features = extract_features(signal_ds, dimension, delay)
```

## Processing Pipeline

```
Raw Signal → Bandpass Filtering → Anti-aliasing Downsampling →
Takens Embedding → Persistent Homology → Landscape Features → Classification
```

## Methodology

**Parameter Estimation**:
- **Delay τ**: First local minimum of Average Mutual Information
- **Dimension d**: False Nearest Neighbors ratio < 10% threshold


## Project Structure

```
TopoEmotion/
├── topoemotion/              # Core computational modules
│   ├── preprocessing/        # Signal conditioning
│   ├── embedding/            # Phase space reconstruction
│   ├── features/             # Topological feature extraction
│   ├── utils/                # Auxiliary functions
│   └── config.py             # System configuration
├── visualization/            # Scientific plotting utilities
├── examples/                 # Usage demonstrations
│   ├── basic_usage.py        # Pipeline walkthrough
│   ├── deep_learning_mlp.py  # MLP model
│   ├── deep_learning_cnn.py  # CNN model
│   └── deep_learning_lstm.py # LSTM model
├── results/
│   └── figures/              # Academic-style visualizations
├── CLASSIFIER_RESULTS.md     # Individual classifier metrics
└── MODEL_COMPARISON.md       # Performance comparison analysis
```


## Visualization Examples

The framework generates publication-ready figures:
- **fig1**: Takens embedding parameters
- **fig2-fig4**: Signal processing stages (raw → preprocessed → downsampled)
- **fig5**: Time-delay embedding 2D projections
- **fig6**: False Nearest Neighbors dimension estimation
- **fig7**: Persistence landscape features (8-signal panel)

Generate all figures:
```bash
python generate_final_figures.py
```

## Citation

```bibtex
@software{topoemotion2025,
  title={TopoEmotion: A Persistent Homology Framework for Robust Multi-Modal Physiological Emotion Recognition},
  author={Your Name},
  year={2025},
  url={https://github.com/Lib-Stinkaa/TopoEmotion}
}
```

## License

MIT License
