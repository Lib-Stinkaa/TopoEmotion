import matplotlib.pyplot as plt
import numpy as np
from topoemotion.utils.visualization import setup_plot_style
from topoemotion.config import COLORS


def plot_persistence_diagram(dgms, max_dim=1, save_path=None):
    setup_plot_style()

    fig, axes = plt.subplots(1, max_dim + 1, figsize=(6 * (max_dim + 1), 5))
    if max_dim == 0:
        axes = [axes]

    colors = [COLORS['primary'], COLORS['secondary']]

    for dim in range(max_dim + 1):
        ax = axes[dim]

        if dim < len(dgms) and len(dgms[dim]) > 0:
            pairs = dgms[dim]
            finite_pairs = pairs[np.isfinite(pairs).all(axis=1)]

            if len(finite_pairs) > 0:
                births = finite_pairs[:, 0]
                deaths = finite_pairs[:, 1]

                # Plot points
                ax.scatter(births, deaths, c=colors[dim], alpha=0.6,
                          s=50, edgecolors='white', linewidth=0.5)

                # Diagonal line
                max_val = max(births.max(), deaths.max())
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1.5)

                ax.set_xlabel('Birth')
                ax.set_ylabel('Death')
                ax.set_title(f'H{dim} Persistence Diagram', fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
            else:
                ax.text(0.5, 0.5, 'No features', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'H{dim} Persistence Diagram', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No features', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(f'H{dim} Persistence Diagram', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_persistence_landscape(landscapes, n_layers=5, max_dim=1, save_path=None):
    setup_plot_style()

    n_bins = len(landscapes) // (n_layers * (max_dim + 1))
    fig, axes = plt.subplots(1, max_dim + 1, figsize=(6 * (max_dim + 1), 5))
    if max_dim == 0:
        axes = [axes]

    colors = [COLORS['primary'], COLORS['secondary']]

    for dim in range(max_dim + 1):
        ax = axes[dim]
        start_idx = dim * n_layers * n_bins
        end_idx = (dim + 1) * n_layers * n_bins
        dim_landscapes = landscapes[start_idx:end_idx].reshape(n_layers, n_bins)

        x = np.arange(n_bins)
        for layer in range(n_layers):
            alpha = 1.0 - (layer / n_layers) * 0.7
            ax.plot(x, dim_landscapes[layer], color=colors[dim],
                   alpha=alpha, linewidth=1.5, label=f'Layer {layer+1}')

        ax.set_xlabel('Position')
        ax.set_ylabel('Landscape Value')
        ax.set_title(f'H{dim} Persistence Landscape', fontweight='bold')
        ax.grid(True, alpha=0.3)
        if dim == max_dim:
            ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_multi_signal_landscapes(signal_landscapes_dict, title=None, save_path=None):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
    })

    def remove_spines(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

    n_signals = len(signal_landscapes_dict)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for idx, (sig_name, landscape) in enumerate(signal_landscapes_dict.items()):
        if idx >= 8:
            break

        ax = axes[idx]
        x = np.arange(len(landscape))

        # Plot filled area
        ax.fill_between(x, 0, landscape, color='#0173B2', alpha=0.6,
                        edgecolor='white', linewidth=0.5)
        ax.plot(x, landscape, color='#0173B2', linewidth=1.5, alpha=0.9)

        remove_spines(ax)
        ax.set_xlabel('Landscape Index', fontweight='bold')
        ax.set_ylabel('Landscape Value', fontweight='bold')
        ax.set_title(sig_name, fontweight='bold', pad=8)

        # Stats box
        max_val = np.max(landscape)
        mean_val = np.mean(landscape)
        nonzero = np.sum(landscape > 0.001)

        stats_text = f'Max: {max_val:.3f}\nMean: {mean_val:.3f}\nNonzero: {nonzero}/{len(landscape)}'
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                          edgecolor='gray', linewidth=0.5))

        ax.set_xlim(0, len(landscape))

    if title is None:
        title = 'Persistence Landscape Features - Examples from Different Signals\n(Subject 1, Video 1, First Window for each signal, 500 landscape values)'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
