"""Configuration for TopoEmotion pipeline."""

import os

# ==================== Paths ====================
DATA_ROOT = "data/interpolated"
OUTPUT_ROOT = "outputs"

# ==================== Signal Settings ====================
SIGNAL_TYPES = [
    'ecg_clean', 'bvp_clean', 'gsr_clean', 'rsp_clean',
    'skt_clean', 'emg_zygo_clean', 'emg_coru_clean', 'emg_trap_clean'
]

# Sampling rates (after downsampling)
SAMPLING_RATE = 100  # Hz (unified for all signals)

# ==================== Window Settings ====================
WINDOW_SECONDS = 8.0  # Window size in seconds
SLIDE_SECONDS = 4.0   # Slide step in seconds (50% overlap)

# ==================== Embedding Parameters ====================
# Calculated from data using AMI and FNN methods
DELAY_SAMPLES = {
    'ecg_clean': 5,
    'bvp_clean': 12,
    'gsr_clean': 31,
    'rsp_clean': 39,
    'skt_clean': 34,
    'emg_zygo_clean': 5,
    'emg_coru_clean': 5,
    'emg_trap_clean': 6
}

EMBEDDING_DIM = {
    'ecg_clean': 3,
    'bvp_clean': 3,
    'gsr_clean': 3,
    'rsp_clean': 4,
    'skt_clean': 4,
    'emg_zygo_clean': 3,
    'emg_coru_clean': 3,
    'emg_trap_clean': 3
}

# ==================== TDA Parameters ====================
MAX_DIM = 1  # Compute H0 and H1
N_LANDSCAPES = 5  # Number of persistence landscape layers
N_BINS = 50  # Resolution of persistence landscapes

# ==================== Classification Settings ====================
VALENCE_THRESHOLD = 5.0
AROUSAL_THRESHOLD = 5.0
RANDOM_STATE = 42

# ==================== Visualization Settings ====================
# Unified color palette (colorblind-friendly)
COLORS = {
    'primary': '#0173B2',    # Blue
    'secondary': '#DE8F05',  # Orange
    'success': '#029E73',    # Green
    'danger': '#CC78BC',     # Purple
    'warning': '#ECE133',    # Yellow
    'info': '#56B4E9',       # Light blue
    'dark': '#949494',       # Gray
    'red': '#CA0020',        # Red
}

# Signal type colors
SIGNAL_COLORS = {
    'ecg_clean': COLORS['primary'],
    'bvp_clean': COLORS['info'],
    'gsr_clean': COLORS['success'],
    'rsp_clean': COLORS['secondary'],
    'skt_clean': COLORS['warning'],
    'emg_zygo_clean': COLORS['danger'],
    'emg_coru_clean': COLORS['dark'],
    'emg_trap_clean': COLORS['red']
}

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
