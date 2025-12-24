import matplotlib.pyplot as plt
import numpy as np
from topoemotion.utils.visualization import setup_plot_style
from topoemotion.config import SIGNAL_COLORS


def plot_signals_comparison(raw_signal, clean_signal, downsampled_signal,
                            signal_type, sampling_rates, save_path=None):
    setup_plot_style()

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
    color = SIGNAL_COLORS.get(f'{signal_type}_clean', '#0173B2')

    t_raw = np.arange(len(raw_signal)) / sampling_rates['raw']
    t_clean = np.arange(len(clean_signal)) / sampling_rates['clean']
    t_down = np.arange(len(downsampled_signal)) / sampling_rates['downsampled']

    axes[0].plot(t_raw, raw_signal, color=color, alpha=0.7, linewidth=0.8)
    axes[0].set_ylabel('Raw Signal')
    axes[0].set_title(f'{signal_type.upper()} Signal Processing Pipeline', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_clean, clean_signal, color=color, linewidth=1.0)
    axes[1].set_ylabel('Cleaned Signal')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_down, downsampled_signal, color=color, linewidth=1.2)
    axes[2].set_ylabel('Downsampled Signal')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
