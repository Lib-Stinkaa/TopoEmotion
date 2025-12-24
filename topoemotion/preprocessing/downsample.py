from scipy import signal
import numpy as np


def downsample_signal(signal_data, original_fs, target_fs):
    factor = int(original_fs / target_fs)
    if factor == 1:
        return signal_data
    try:
        return signal.decimate(signal_data, factor, ftype='fir', zero_phase=True)
    except:
        return signal_data[::factor]
