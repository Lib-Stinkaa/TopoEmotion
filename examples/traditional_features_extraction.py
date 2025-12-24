#!/usr/bin/env python3
"""
Traditional Time-Frequency Feature Extraction

Extracts standard time-domain and frequency-domain features from physiological signals:
- ECG/BVP: HRV time-domain and frequency-domain features
- GSR: SCL/SCR time-domain features
- RSP: Respiratory time-domain features
- EMG: Time-domain statistics and frequency-domain features
- SKT: Temperature time-domain features
"""

import pandas as pd
import numpy as np
import os
from scipy import signal, stats
from scipy.signal import find_peaks, welch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import SIGNAL_TYPES

# Configuration
INPUT_DIR = 'sup_exp/downsampled_data'
OUTPUT_DIR = 'sup_exp/traditional_features'
SAMPLING_RATE = 100


def detect_r_peaks(ecg_signal, fs=100):
    """Simple R-peak detection algorithm"""
    sos = signal.butter(4, [5, 15], btype='band', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, ecg_signal)
    squared = filtered ** 2
    window_size = int(0.12 * fs)
    integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    min_distance = int(0.6 * fs)
    height_threshold = np.mean(integrated) * 0.5
    peaks, _ = find_peaks(integrated, distance=min_distance, height=height_threshold)
    return peaks


def extract_hrv_time_domain(rr_intervals):
    """Extract HRV time-domain features"""
    if len(rr_intervals) < 2:
        return {'mean_rr': np.nan, 'std_rr': np.nan, 'sdnn': np.nan,
                'rmssd': np.nan, 'pnn50': np.nan, 'mean_hr': np.nan, 'std_hr': np.nan}

    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals, ddof=1)
    sdnn = std_rr
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    nn50 = np.sum(np.abs(diff_rr) > 0.05)
    pnn50 = (nn50 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0
    hr = 60.0 / rr_intervals
    mean_hr = np.mean(hr)
    std_hr = np.std(hr, ddof=1)

    return {'mean_rr': mean_rr, 'std_rr': std_rr, 'sdnn': sdnn,
            'rmssd': rmssd, 'pnn50': pnn50, 'mean_hr': mean_hr, 'std_hr': std_hr}


def extract_hrv_frequency_domain(rr_intervals):
    """Extract HRV frequency-domain features"""
    if len(rr_intervals) < 10:
        return {'vlf_power': np.nan, 'lf_power': np.nan, 'hf_power': np.nan,
                'total_power': np.nan, 'lf_hf_ratio': np.nan,
                'lf_norm': np.nan, 'hf_norm': np.nan}

    fs_resample = 4.0
    time_rr = np.cumsum(rr_intervals) - np.cumsum(rr_intervals)[0]
    time_uniform = np.arange(0, time_rr[-1], 1/fs_resample)
    rr_uniform = np.interp(time_uniform, time_rr, rr_intervals)
    freqs, psd = welch(rr_uniform, fs=fs_resample, nperseg=min(256, len(rr_uniform)))

    vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)

    vlf_power = np.trapz(psd[vlf_mask], freqs[vlf_mask]) if np.any(vlf_mask) else 0
    lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
    hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0
    total_power = vlf_power + lf_power + hf_power
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan
    lf_norm = lf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else 0
    hf_norm = hf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else 0

    return {'vlf_power': vlf_power, 'lf_power': lf_power, 'hf_power': hf_power,
            'total_power': total_power, 'lf_hf_ratio': lf_hf_ratio,
            'lf_norm': lf_norm, 'hf_norm': hf_norm}


def extract_ecg_features(ecg_signal, fs=100):
    """Extract complete HRV features from ECG/BVP"""
    r_peaks = detect_r_peaks(ecg_signal, fs)
    if len(r_peaks) < 2:
        empty_td = extract_hrv_time_domain([])
        empty_fd = extract_hrv_frequency_domain([])
        return {**empty_td, **empty_fd}

    rr_intervals = np.diff(r_peaks) / fs
    valid_mask = (rr_intervals >= 0.3) & (rr_intervals <= 2.0)
    rr_intervals = rr_intervals[valid_mask]

    if len(rr_intervals) < 2:
        empty_td = extract_hrv_time_domain([])
        empty_fd = extract_hrv_frequency_domain([])
        return {**empty_td, **empty_fd}

    td_features = extract_hrv_time_domain(rr_intervals)
    fd_features = extract_hrv_frequency_domain(rr_intervals)
    return {**td_features, **fd_features}


def extract_gsr_features(gsr_signal, fs=100):
    """Extract GSR/EDA time-domain features"""
    sos_tonic = signal.butter(4, 0.05, btype='low', fs=fs, output='sos')
    scl = signal.sosfiltfilt(sos_tonic, gsr_signal)
    scr = gsr_signal - scl

    mean_scl = np.mean(scl)
    std_scl = np.std(scl, ddof=1)

    threshold = np.std(scr) * 0.5
    peaks, properties = find_peaks(scr, height=threshold, distance=int(0.5*fs))
    n_scr = len(peaks)

    if n_scr > 0:
        scr_amplitudes = properties['peak_heights']
        mean_scr = np.mean(scr_amplitudes)
        std_scr = np.std(scr_amplitudes)
        max_scr = np.max(scr_amplitudes)
        min_scr = np.min(scr_amplitudes)
    else:
        mean_scr = std_scr = max_scr = min_scr = 0

    gsr_diff = np.diff(gsr_signal)
    mean_derivative = np.mean(gsr_diff)
    std_derivative = np.std(gsr_diff, ddof=1)

    return {'mean_scl': mean_scl, 'std_scl': std_scl, 'mean_scr': mean_scr,
            'std_scr': std_scr, 'n_scr': n_scr, 'max_scr': max_scr,
            'min_scr': min_scr, 'mean_derivative': mean_derivative,
            'std_derivative': std_derivative}


def extract_rsp_features(rsp_signal, fs=100):
    """Extract respiratory time-domain features"""
    sos = signal.butter(4, [0.1, 0.5], btype='band', fs=fs, output='sos')
    rsp_filtered = signal.sosfiltfilt(sos, rsp_signal)
    peaks, _ = find_peaks(rsp_filtered, distance=int(1.0*fs))
    troughs, _ = find_peaks(-rsp_filtered, distance=int(1.0*fs))

    if len(peaks) > 1:
        breath_intervals = np.diff(peaks) / fs
        breathing_rate = 60.0 / np.mean(breath_intervals)
        std_breath_interval = np.std(breath_intervals)
    else:
        breathing_rate = std_breath_interval = np.nan

    if len(peaks) > 0 and len(troughs) > 0:
        amplitudes = []
        for peak in peaks:
            closest_trough_idx = np.argmin(np.abs(troughs - peak))
            if closest_trough_idx < len(troughs):
                amp = rsp_filtered[peak] - rsp_filtered[troughs[closest_trough_idx]]
                amplitudes.append(amp)
        mean_amplitude = np.mean(amplitudes) if amplitudes else np.nan
        std_amplitude = np.std(amplitudes) if amplitudes else np.nan
    else:
        mean_amplitude = std_amplitude = np.nan

    return {'breathing_rate': breathing_rate, 'std_breath_interval': std_breath_interval,
            'mean_amplitude': mean_amplitude, 'std_amplitude': std_amplitude}


def extract_emg_features(emg_signal, fs=100):
    """Extract EMG time-domain and frequency-domain features"""
    emg_abs = np.abs(emg_signal)
    mean_emg = np.mean(emg_abs)
    std_emg = np.std(emg_signal, ddof=1)
    rms_emg = np.sqrt(np.mean(emg_signal ** 2))
    max_emg = np.max(emg_abs)
    min_emg = np.min(emg_abs)
    variance_emg = np.var(emg_signal, ddof=1)
    skewness_emg = stats.skew(emg_signal)
    kurtosis_emg = stats.kurtosis(emg_signal)
    waveform_length = np.sum(np.abs(np.diff(emg_signal)))

    freqs, psd = welch(emg_signal, fs=fs, nperseg=min(256, len(emg_signal)))
    valid_idx = freqs > 0
    freqs = freqs[valid_idx]
    psd = psd[valid_idx]

    if len(psd) > 0:
        mean_freq = np.sum(freqs * psd) / np.sum(psd)
        cumulative_power = np.cumsum(psd)
        total_power = cumulative_power[-1]
        median_freq_idx = np.argmin(np.abs(cumulative_power - total_power/2))
        median_freq = freqs[median_freq_idx]
        total_power_emg = np.sum(psd)
        peak_freq = freqs[np.argmax(psd)]
    else:
        mean_freq = median_freq = total_power_emg = peak_freq = np.nan

    return {'mean': mean_emg, 'std': std_emg, 'rms': rms_emg, 'max': max_emg,
            'min': min_emg, 'variance': variance_emg, 'skewness': skewness_emg,
            'kurtosis': kurtosis_emg, 'mav': mean_emg, 'waveform_length': waveform_length,
            'mean_freq': mean_freq, 'median_freq': median_freq,
            'total_power': total_power_emg, 'peak_freq': peak_freq}


def extract_skt_features(skt_signal):
    """Extract skin temperature time-domain features"""
    mean_temp = np.mean(skt_signal)
    std_temp = np.std(skt_signal, ddof=1)
    max_temp = np.max(skt_signal)
    min_temp = np.min(skt_signal)
    range_temp = max_temp - min_temp
    x = np.arange(len(skt_signal))
    slope = np.polyfit(x, skt_signal, 1)[0] if len(x) > 1 else 0

    return {'mean_temp': mean_temp, 'std_temp': std_temp, 'max_temp': max_temp,
            'min_temp': min_temp, 'range_temp': range_temp, 'slope': slope}


def extract_features_from_window(window_signal, signal_type, fs=100):
    """Extract features from a single window"""
    if signal_type in ['ecg_clean', 'bvp_clean']:
        features = extract_ecg_features(window_signal, fs)
        prefix = 'ecg_' if signal_type == 'ecg_clean' else 'bvp_'
    elif signal_type == 'gsr_clean':
        features = extract_gsr_features(window_signal, fs)
        prefix = 'gsr_'
    elif signal_type == 'rsp_clean':
        features = extract_rsp_features(window_signal, fs)
        prefix = 'rsp_'
    elif 'emg' in signal_type:
        features = extract_emg_features(window_signal, fs)
        muscle = signal_type.replace('_clean', '').replace('emg_', '')
        prefix = f'emg_{muscle}_'
    elif signal_type == 'skt_clean':
        features = extract_skt_features(window_signal)
        prefix = 'skt_'
    else:
        return {}

    return {f'{prefix}{k}': v for k, v in features.items()}


def process_single_file(file_path, signal_type):
    """Process a single CSV file and extract features from all windows"""
    df = pd.read_csv(file_path)
    window_ids = df['window_id'].unique()
    all_features = []

    for window_id in window_ids:
        window_data = df[df['window_id'] == window_id]
        metadata = {
            'subject_id': window_data['subject_id'].iloc[0],
            'video_id': window_data['video_id'].iloc[0],
            'window_id': window_id,
            'valence': window_data['valence'].iloc[0],
            'arousal': window_data['arousal'].iloc[0],
            'signal_type': signal_type
        }
        window_signal = window_data[signal_type].values
        features = extract_features_from_window(window_signal, signal_type, SAMPLING_RATE)
        all_features.append({**metadata, **features})

    return pd.DataFrame(all_features)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for signal_type in SIGNAL_TYPES:
        files = [f for f in os.listdir(INPUT_DIR) if f.endswith(f'_{signal_type}.csv')]
        if not files:
            continue

        all_dfs = []
        for file in tqdm(files, desc=f"{signal_type}"):
            try:
                df = process_single_file(os.path.join(INPUT_DIR, file), signal_type)
                all_dfs.append(df)
            except:
                continue

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            output_file = os.path.join(OUTPUT_DIR, f'{signal_type}_features.csv')
            combined_df.to_csv(output_file, index=False)

    all_features = []
    for signal_type in SIGNAL_TYPES:
        feature_file = os.path.join(OUTPUT_DIR, f'{signal_type}_features.csv')
        if os.path.exists(feature_file):
            all_features.append(pd.read_csv(feature_file))

    if all_features:
        combined = pd.concat(all_features, ignore_index=True)
        combined.to_csv(os.path.join(OUTPUT_DIR, 'all_traditional_features.csv'), index=False)


if __name__ == "__main__":
    main()
