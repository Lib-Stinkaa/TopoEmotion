import numpy as np
import matplotlib.pyplot as plt
from topoemotion.preprocessing import preprocess_signal, downsample_signal
from topoemotion.embedding import compute_delay, compute_dimension, takens_embedding
from topoemotion.features import extract_features
from topoemotion.utils.visualization import setup_plot_style
from visualization.plot_features import plot_persistence_diagram, plot_persistence_landscape
from ripser import ripser


def generate_example_signal(n_points=8000, fs=1000):
    t = np.arange(n_points) / fs
    signal = np.sin(2 * np.pi * 1.2 * t)
    signal += 0.3 * np.sin(2 * np.pi * 0.3 * t)
    signal += 0.1 * np.random.randn(n_points)
    return signal, t


def main():
    setup_plot_style()

    raw_signal, t = generate_example_signal()
    clean_signal = preprocess_signal(raw_signal, 'ecg', sampling_rate=1000)
    downsampled = downsample_signal(clean_signal, original_fs=1000, target_fs=100)

    delay = compute_delay(downsampled, max_delay=50)
    dimension = compute_dimension(downsampled, delay, max_dim=6)

    features = extract_features(downsampled, dimension, delay, max_dim=1)

    point_cloud = takens_embedding(downsampled, dimension, delay)
    result = ripser(point_cloud, maxdim=1)

    plot_persistence_diagram(result['dgms'], max_dim=1,
                            save_path='results/figures/persistence_diagram_example.png')

    landscape_features = np.array([features[f'landscape_{i}'] for i in range(500)])

    plot_persistence_landscape(landscape_features, n_layers=5, max_dim=1,
                              save_path='results/figures/persistence_landscape_example.png')


if __name__ == "__main__":
    main()
