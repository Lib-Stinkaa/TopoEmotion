import neurokit2 as nk
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_signal(signal, signal_type, sampling_rate=1000):
    if signal_type == 'ecg':
        return nk.ecg_clean(signal, sampling_rate=sampling_rate, method='neurokit')
    elif signal_type == 'gsr':
        return nk.signal_filter(signal, sampling_rate=sampling_rate,
                               lowcut=0.05, highcut=5.0,
                               method='butterworth', order=4)
    elif signal_type == 'bvp':
        return nk.ppg_clean(signal, sampling_rate=sampling_rate, method='elgendi')
    elif signal_type == 'rsp':
        return nk.rsp_clean(signal, sampling_rate=sampling_rate, method='khodadad2018')
    elif signal_type == 'skt':
        return nk.signal_filter(signal, sampling_rate=sampling_rate,
                               lowcut=None, highcut=0.5,
                               method='butterworth', order=4)
    elif signal_type in ['emg_zygo', 'emg_coru', 'emg_trap']:
        return nk.signal_filter(signal, sampling_rate=sampling_rate,
                               lowcut=20, highcut=450,
                               method='butterworth', order=4)
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")


def preprocess_subject(df, sampling_rate=1000):
    df_clean = df.copy()
    signal_types = ['ecg', 'gsr', 'bvp', 'rsp', 'skt', 'emg_zygo', 'emg_coru', 'emg_trap']

    for sig_type in signal_types:
        if sig_type in df.columns:
            df_clean[f'{sig_type}_clean'] = preprocess_signal(
                df[sig_type].values, sig_type, sampling_rate
            )

    scaler = StandardScaler()
    clean_cols = [c for c in df_clean.columns if c.endswith('_clean')]
    if clean_cols:
        df_clean[clean_cols] = scaler.fit_transform(df_clean[clean_cols])

    return df_clean
