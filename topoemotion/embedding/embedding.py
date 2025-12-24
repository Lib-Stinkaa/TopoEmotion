import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors


def compute_delay(signal, max_delay=100):
    signal = signal[:min(50000, len(signal))]
    max_delay = min(max_delay, len(signal) // 2)

    mi_values = []
    for delay in range(1, max_delay):
        original = signal[:-delay].reshape(-1, 1)
        delayed = signal[delay:]

        if len(original) > 100:
            mi = mutual_info_regression(original, delayed, random_state=42)[0]
            mi_values.append(mi)
        else:
            mi_values.append(0)

    for i in range(1, len(mi_values) - 1):
        if mi_values[i] < mi_values[i-1] and mi_values[i] < mi_values[i+1]:
            return i + 1

    return int(np.argmin(mi_values)) + 1


def compute_dimension(signal, delay, max_dim=10, rtol=15.0, atol=2.0, threshold=0.10):
    for dim in range(1, max_dim + 1):
        fnn_ratio = _compute_fnn_ratio(signal, dim, delay, rtol, atol)
        if fnn_ratio < threshold:
            return max(dim + 1, 3)
    return 3


def _compute_fnn_ratio(signal, dim, delay, rtol, atol):
    if len(signal) > 5000:
        step = len(signal) // 5000
        signal = signal[::step]
        delay = max(1, delay // step)

    try:
        emb_d = takens_embedding(signal, dim, delay)
        emb_d1 = takens_embedding(signal, dim + 1, delay)
    except ValueError:
        return 0.0

    n_points = min(len(emb_d), len(emb_d1))
    if n_points < 10:
        return 0.0

    emb_d = emb_d[:n_points]
    emb_d1 = emb_d1[:n_points]

    signal_std = np.std(signal)
    if signal_std == 0:
        return 0.0

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto')
    nbrs.fit(emb_d)
    distances, indices = nbrs.kneighbors(emb_d)

    false_neighbors = 0
    for i in range(n_points):
        nn_idx = indices[i, 1]
        dist_d = distances[i, 1]

        if dist_d == 0:
            continue

        dist_d1 = np.linalg.norm(emb_d1[i] - emb_d1[nn_idx])
        rel_increase = abs(dist_d1 - dist_d) / dist_d
        abs_distance = dist_d1 / signal_std

        if rel_increase > rtol or abs_distance > atol:
            false_neighbors += 1

    return false_neighbors / n_points


def takens_embedding(signal, dim, delay):
    n = len(signal)
    m = n - (dim - 1) * delay

    if m <= 0:
        raise ValueError("Signal too short for embedding")

    embedded = np.zeros((m, dim))
    for i in range(dim):
        embedded[:, i] = signal[i * delay : i * delay + m]

    return embedded
