"""Visualization utilities with unified styling."""

import matplotlib.pyplot as plt
import seaborn as sns
from ..config import COLORS, FIGURE_DPI


def setup_plot_style():
    """
    Set up unified plotting style.
    Uses colorblind-friendly palette and professional formatting.
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    # Set matplotlib parameters
    plt.rcParams.update({
        'figure.dpi': FIGURE_DPI,
        'savefig.dpi': FIGURE_DPI,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'lines.linewidth': 1.5,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'grid.alpha': 0.3,
    })


def get_color_palette(n_colors=None):
    """
    Get colorblind-friendly color palette.

    Args:
        n_colors: Number of colors needed

    Returns:
        List of color codes
    """
    base_palette = [
        COLORS['primary'],
        COLORS['secondary'],
        COLORS['success'],
        COLORS['danger'],
        COLORS['warning'],
        COLORS['info'],
        COLORS['dark'],
        COLORS['red']
    ]

    if n_colors is None or n_colors <= len(base_palette):
        return base_palette[:n_colors] if n_colors else base_palette
    else:
        # Extend palette if more colors needed
        return sns.color_palette("husl", n_colors)
