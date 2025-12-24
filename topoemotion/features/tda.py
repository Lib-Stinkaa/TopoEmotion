import numpy as np
from ripser import ripser
from ..embedding.embedding import takens_embedding
from ..config import MAX_DIM, N_LANDSCAPES, N_BINS


def extract_features(signal, dim, delay, max_dim=MAX_DIM):
    point_cloud = takens_embedding(signal, dim, delay)

    if len(point_cloud) < 10:
        return _get_empty_features()

    try:
        result = ripser(point_cloud, maxdim=max_dim)
        dgms = result['dgms']

        features = {}
        landscapes = compute_persistence_landscape(dgms, max_dim=max_dim)
        for i, val in enumerate(landscapes):
            features[f'landscape_{i}'] = val

        stats = _compute_persistence_statistics(dgms, max_dim=max_dim)
        features.update(stats)

        return features
    except:
        return _get_empty_features()


def compute_persistence_landscape(dgms, n_layers=N_LANDSCAPES, n_bins=N_BINS, max_dim=MAX_DIM):
    all_landscapes = []

    for dim in range(max_dim + 1):
        if dim < len(dgms) and len(dgms[dim]) > 0:
            pairs = dgms[dim]
            finite_pairs = pairs[np.isfinite(pairs).all(axis=1)]

            if len(finite_pairs) > 0:
                landscapes = _compute_landscape_for_diagram(finite_pairs, n_layers, n_bins)
                all_landscapes.append(landscapes.flatten())
            else:
                all_landscapes.append(np.zeros(n_layers * n_bins))
        else:
            all_landscapes.append(np.zeros(n_layers * n_bins))

    return np.concatenate(all_landscapes)


def _compute_landscape_for_diagram(pairs, n_layers, n_bins):
    births = pairs[:, 0]
    deaths = pairs[:, 1]

    x_min = births.min()
    x_max = deaths.max()

    if x_max <= x_min:
        return np.zeros((n_layers, n_bins))

    x = np.linspace(x_min, x_max, n_bins)
    x_expanded = x[:, np.newaxis]
    mids = (births + deaths) / 2

    left_mask = (x_expanded >= births) & (x_expanded <= mids)
    left_vals = x_expanded - births

    right_mask = (x_expanded > mids) & (x_expanded <= deaths)
    right_vals = deaths - x_expanded

    tent_matrix = np.where(left_mask, left_vals, 0.0)
    tent_matrix = np.where(right_mask, right_vals, tent_matrix)

    landscapes = np.zeros((n_layers, n_bins))
    n_pairs = tent_matrix.shape[1]

    if n_pairs >= n_layers:
        for i in range(n_bins):
            row = tent_matrix[i]
            idx = np.argpartition(row, -n_layers)[-n_layers:]
            top_vals = row[idx]
            top_vals.sort()
            landscapes[:, i] = top_vals[::-1]
    else:
        for i in range(n_bins):
            row = tent_matrix[i]
            sorted_vals = np.sort(row)[::-1]
            landscapes[:len(sorted_vals), i] = sorted_vals

    return landscapes


def _compute_persistence_statistics(dgms, max_dim=MAX_DIM):
    stats = {}

    for dim in range(max_dim + 1):
        prefix = f'H{dim}_'

        if dim < len(dgms) and len(dgms[dim]) > 0:
            pairs = dgms[dim]
            finite_pairs = pairs[np.isfinite(pairs).all(axis=1)]

            if len(finite_pairs) > 0:
                lifetimes = finite_pairs[:, 1] - finite_pairs[:, 0]
                births = finite_pairs[:, 0]
                deaths = finite_pairs[:, 1]

                stats[prefix + 'n_features'] = len(lifetimes)
                stats[prefix + 'sum_lifetime'] = np.sum(lifetimes)
                stats[prefix + 'mean_lifetime'] = np.mean(lifetimes)
                stats[prefix + 'std_lifetime'] = np.std(lifetimes)
                stats[prefix + 'max_lifetime'] = np.max(lifetimes)
                stats[prefix + 'min_lifetime'] = np.min(lifetimes)
                stats[prefix + 'mean_birth'] = np.mean(births)
                stats[prefix + 'mean_death'] = np.mean(deaths)

                if np.sum(lifetimes) > 0:
                    probs = lifetimes / np.sum(lifetimes)
                    probs = probs[probs > 0]
                    stats[prefix + 'entropy'] = -np.sum(probs * np.log(probs))
                else:
                    stats[prefix + 'entropy'] = 0.0
            else:
                stats.update(_get_empty_stats_for_dim(prefix))
        else:
            stats.update(_get_empty_stats_for_dim(prefix))

    return stats


def _get_empty_stats_for_dim(prefix):
    return {
        prefix + 'n_features': 0.0,
        prefix + 'sum_lifetime': 0.0,
        prefix + 'mean_lifetime': 0.0,
        prefix + 'std_lifetime': 0.0,
        prefix + 'max_lifetime': 0.0,
        prefix + 'min_lifetime': 0.0,
        prefix + 'mean_birth': 0.0,
        prefix + 'mean_death': 0.0,
        prefix + 'entropy': 0.0
    }


def _get_empty_features():
    features = {}
    for i in range(N_LANDSCAPES * N_BINS * (MAX_DIM + 1)):
        features[f'landscape_{i}'] = 0.0

    for dim in range(MAX_DIM + 1):
        features.update(_get_empty_stats_for_dim(f'H{dim}_'))

    return features
