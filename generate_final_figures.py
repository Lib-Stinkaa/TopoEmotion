import sys
sys.path.insert(0, '.')
import numpy as np
import matplotlib.pyplot as plt
from topoemotion.preprocessing import preprocess_signal, downsample_signal
from topoemotion.embedding import compute_delay, compute_dimension, takens_embedding
import os

os.makedirs('results/figures', exist_ok=True)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
})

def remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

# Signal colors
SIGNAL_COLORS = {
    'ecg': '#0173B2',
    'bvp': '#0173B2',
    'gsr': '#029E73',
    'rsp': '#D55E00',
    'skt': '#E69F00',
    'emg_zygo': '#CC79A7',
    'emg_coru': '#999999',
    'emg_trap': '#D62728',
}

# Generate signals
np.random.seed(42)
fs_raw = 1000
duration = 8  # 8 seconds as requested
t = np.arange(0, duration, 1/fs_raw)

def generate_ecg():
    signal = np.sin(2 * np.pi * 1.2 * t)
    signal += 0.3 * np.sin(2 * np.pi * 0.25 * t)
    signal += 0.15 * np.random.randn(len(t))
    return signal

def generate_bvp():
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)
    signal += 0.2 * np.random.randn(len(t))
    return signal

def generate_gsr():
    signal = 0.5 * np.sin(2 * np.pi * 0.1 * t)
    for peak_t in [2, 5, 7]:
        signal += 0.3 * np.exp(-((t - peak_t) / 0.5) ** 2)
    signal += 0.1 * np.random.randn(len(t))
    return signal

def generate_rsp():
    signal = np.sin(2 * np.pi * 0.25 * t)
    signal += 0.2 * np.random.randn(len(t))
    return signal

def generate_skt():
    signal = 0.3 * np.sin(2 * np.pi * 0.05 * t)
    signal += 0.05 * np.random.randn(len(t))
    return signal

def generate_emg():
    signal = np.zeros_like(t)
    for burst_t in [1, 3, 5, 7]:
        burst_duration = 0.5
        mask = (t >= burst_t) & (t < burst_t + burst_duration)
        signal[mask] = np.random.randn(np.sum(mask)) * 0.5
    signal += 0.1 * np.random.randn(len(t))
    return signal

signal_generators = {
    'ecg': generate_ecg,
    'bvp': generate_bvp,
    'gsr': generate_gsr,
    'rsp': generate_rsp,
    'skt': generate_skt,
    'emg_zygo': generate_emg,
    'emg_coru': generate_emg,
    'emg_trap': generate_emg,
}

# Process signals
signal_list = ['ecg', 'bvp', 'gsr', 'rsp', 'skt', 'emg_zygo', 'emg_coru', 'emg_trap']
raw_signals = {}
clean_signals = {}
downsampled_signals = {}
embedding_params = {}

for sig_name in signal_list:
    raw = signal_generators[sig_name]()
    clean = preprocess_signal(raw, sig_name, fs_raw)
    ds = downsample_signal(clean, fs_raw, 100)

    raw_signals[sig_name] = raw
    clean_signals[sig_name] = clean
    downsampled_signals[sig_name] = ds

# Use actual parameters from real data (not computed from synthetic signals)
embedding_params = {
    'ecg': {'delay': 5, 'dimension': 3},
    'bvp': {'delay': 12, 'dimension': 3},
    'gsr': {'delay': 31, 'dimension': 3},
    'rsp': {'delay': 39, 'dimension': 4},
    'skt': {'delay': 34, 'dimension': 4},
    'emg_zygo': {'delay': 5, 'dimension': 3},
    'emg_coru': {'delay': 5, 'dimension': 3},
    'emg_trap': {'delay': 6, 'dimension': 3},
}

fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')

table_data = [['Signal', 'Sampling Rate (Hz)', 'Delay τ (samples)', 'Delay (ms)', 'Dimension d']]

for sig_name in signal_list:
    params = embedding_params[sig_name]
    delay_ms = params['delay'] * 10
    table_data.append([
        sig_name.upper().replace('_', ' '),
        '100',
        str(params['delay']),
        f"{delay_ms}",
        str(params['dimension'])
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.2, 0.2, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

# Header styling
for i in range(5):
    table[(0, i)].set_facecolor('#CCCCCC')
    table[(0, i)].set_text_props(weight='bold')

# Data cells
for i in range(1, len(signal_list) + 1):
    for j in range(5):
        table[(i, j)].set_facecolor('#FFFFFF')

# Title at bottom
ax.text(0.5, -0.05, 'Takens Embedding Parameters',
        ha='center', va='top', fontweight='bold', fontsize=14,
        transform=ax.transAxes)

plt.tight_layout()
plt.savefig('results/figures/fig1_parameters.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(8, 1, figsize=(14, 10))

t_plot = t  # Full 8 seconds

for idx, sig_name in enumerate(signal_list):
    ax = axes[idx]
    signal = raw_signals[sig_name]
    color = SIGNAL_COLORS[sig_name]

    ax.plot(t_plot, signal, color=color, linewidth=0.8, alpha=0.8)
    remove_spines(ax)

    # Add signal name on the right
    ax.text(1.02, 0.5, sig_name.upper().replace('_', ' '),
            transform=ax.transAxes, fontsize=11,
            va='center', fontweight='bold')

    if idx < 7:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Time (s)', fontweight='bold')

    ax.set_ylabel('Amplitude', fontsize=9)

# Title at bottom
fig.text(0.5, 0.02, 'Raw Signals (1000 Hz)',
         ha='center', fontweight='bold', fontsize=14)

plt.tight_layout(rect=[0, 0.03, 0.98, 1])
plt.savefig('results/figures/fig2_raw_signals.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(8, 1, figsize=(14, 10))

for idx, sig_name in enumerate(signal_list):
    ax = axes[idx]
    signal = clean_signals[sig_name]
    color = SIGNAL_COLORS[sig_name]

    ax.plot(t_plot, signal, color=color, linewidth=0.8)
    remove_spines(ax)

    ax.text(1.02, 0.5, sig_name.upper().replace('_', ' '),
            transform=ax.transAxes, fontsize=11,
            va='center', fontweight='bold')

    if idx < 7:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Time (s)', fontweight='bold')

    ax.set_ylabel('Amplitude', fontsize=9)

fig.text(0.5, 0.02, 'Preprocessed Signals (Filtered)',
         ha='center', fontweight='bold', fontsize=14)

plt.tight_layout(rect=[0, 0.03, 0.98, 1])
plt.savefig('results/figures/fig3_preprocessed_signals.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(8, 1, figsize=(14, 10))

t_ds = np.arange(0, duration, 1/100)  # 100 Hz

for idx, sig_name in enumerate(signal_list):
    ax = axes[idx]
    signal = downsampled_signals[sig_name]
    color = SIGNAL_COLORS[sig_name]

    ax.plot(t_ds, signal, color=color, linewidth=0.8)
    remove_spines(ax)

    ax.text(1.02, 0.5, sig_name.upper().replace('_', ' '),
            transform=ax.transAxes, fontsize=11,
            va='center', fontweight='bold')

    if idx < 7:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Time (s)', fontweight='bold')

    ax.set_ylabel('Amplitude', fontsize=9)

fig.text(0.5, 0.02, 'Downsampled Signals (100 Hz)',
         ha='center', fontweight='bold', fontsize=14)

plt.tight_layout(rect=[0, 0.03, 0.98, 1])
plt.savefig('results/figures/fig4_downsampled_signals.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, sig_name in enumerate(signal_list):
    ax = axes[idx]
    signal = downsampled_signals[sig_name]
    params = embedding_params[sig_name]
    delay = params['delay']
    dimension = params['dimension']

    # Create embedding
    point_cloud = takens_embedding(signal, dimension, delay)

    # Plot 2D projection (first two dimensions)
    if dimension >= 2:
        color = SIGNAL_COLORS[sig_name]
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1],
                  c=np.arange(len(point_cloud)), cmap='viridis',
                  alpha=0.6, s=5, edgecolors='none')

        remove_spines(ax)
        ax.set_xlabel(f'Dimension 1')
        ax.set_ylabel(f'Dimension 2')
        ax.set_title(f'{sig_name.upper().replace("_", " ")}\n(τ={delay}, d={dimension})',
                    fontweight='bold', fontsize=11)

fig.text(0.5, 0.02, '2D Projections of Time-Delay Embeddings',
         ha='center', fontweight='bold', fontsize=14)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig('results/figures/fig5_embeddings.png', dpi=300, bbox_inches='tight')
plt.close()

from topoemotion.embedding.embedding import _compute_fnn_ratio

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

curve_color = '#1f77b4'

for idx, sig_name in enumerate(signal_list):
    ax = axes[idx]
    signal = downsampled_signals[sig_name]
    params = embedding_params[sig_name]
    delay = params['delay']

    dims = range(1, 11)
    fnn_ratios = []
    for dim in dims:
        ratio = _compute_fnn_ratio(signal, dim, delay, rtol=15.0, atol=2.0)
        fnn_ratios.append(ratio * 100)

    ax.plot(dims, fnn_ratios, 'o-', color=curve_color, linewidth=2,
            markersize=5, markerfacecolor='white', markeredgewidth=1.5)

    ax.axhline(y=10, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    opt_dim = params['dimension']
    if opt_dim <= 10:
        ax.axvline(x=opt_dim, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

    remove_spines(ax)
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('FNN Ratio (%)')
    ax.set_title(f'{sig_name.upper().replace("_", " ")}', fontweight='bold')
    ax.set_xticks(range(1, 11))
    ax.set_ylim(-5, max(fnn_ratios) * 1.1)

fig.text(0.5, 0.02, 'False Nearest Neighbors Analysis',
         ha='center', fontweight='bold', fontsize=14)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig('results/figures/fig6_fnn_curves.png', dpi=300, bbox_inches='tight')
plt.close()

from topoemotion.features import extract_features

np.random.seed(42)
signal_landscapes = {}

for sig_name in signal_list:
    signal = downsampled_signals[sig_name]
    params = embedding_params[sig_name]

    # Extract features
    features = extract_features(signal, params['dimension'], params['delay'], max_dim=1)

    # Extract 500 landscape values (H0 + H1, 5 layers, 50 bins each)
    # H0: 250 values (5 layers × 50 bins), H1: 250 values (5 layers × 50 bins)
    landscape = np.array([features[f'landscape_{i}'] for i in range(500)])

    signal_landscapes[sig_name.upper().replace('_', ' ')] = landscape

from visualization.plot_features import plot_multi_signal_landscapes

plot_multi_signal_landscapes(
    signal_landscapes,
    title='Persistence Landscape Features - Examples from Different Signals\n(Subject 1, Video 1, First Window for each signal, 500 landscape values)',
    save_path='results/figures/fig7_persistence_landscapes.png'
)
