"""Results visualization for LOSO validation."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from topoemotion.utils.visualization import setup_plot_style
from topoemotion.config import COLORS


def plot_loso_results(results_dict, save_path=None):
    """
    Plot LOSO cross-validation results.

    Args:
        results_dict: Dict with 'valence' and 'arousal' results
        save_path: Path to save figure (optional)
    """
    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    metrics = ['accuracy', 'f1', 'precision', 'recall']
    colors = [COLORS['primary'], COLORS['secondary']]

    for idx, (target, results) in enumerate(results_dict.items()):
        ax = axes[idx]

        fold_details = results['fold_details']
        subjects = [f['test_subject'] for f in fold_details]

        # Extract metrics for each fold
        data = {metric: [f[metric] for f in fold_details] for metric in metrics}

        x = np.arange(len(metrics))
        means = [np.mean(data[m]) for m in metrics]
        stds = [np.std(data[m]) for m in metrics]

        # Bar plot with error bars
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                     color=colors[idx], edgecolor='white', linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.set_ylabel('Score')
        ax.set_title(f'{target.capitalize()} Classification Performance', fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels=['Low', 'High'], save_path=None):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        save_path: Path to save figure (optional)
    """
    setup_plot_style()

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Normalized Frequency'},
                linewidths=1, linecolor='white', ax=ax)

    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()
