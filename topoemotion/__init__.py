"""
TopoEmotion: Topological Data Analysis for Emotion Recognition
===============================================================

A Python package for emotion recognition from physiological signals
using Topological Data Analysis (TDA).

Modules:
    - preprocessing: Signal preprocessing and filtering
    - embedding: Time-delay embedding parameter estimation
    - features: Topological feature extraction
    - classification: Emotion classification with LOSO validation
    - utils: Utility functions and visualization
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .preprocessing.preprocess import preprocess_signal, preprocess_subject
from .preprocessing.downsample import downsample_signal
from .embedding.embedding import compute_delay, compute_dimension, takens_embedding
from .features.tda import extract_features

__all__ = [
    "preprocess_signal",
    "preprocess_subject",
    "downsample_signal",
    "compute_delay",
    "compute_dimension",
    "takens_embedding",
    "extract_features",
]
