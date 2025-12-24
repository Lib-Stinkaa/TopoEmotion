import os
import pandas as pd
from pathlib import Path


def check_directory(path, create=True):
    path = Path(path)
    if not path.exists() and create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def load_data(file_path, **kwargs):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path, **kwargs)


def save_results(data, output_path, **kwargs):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    data.to_csv(output_path, index=False, **kwargs)
