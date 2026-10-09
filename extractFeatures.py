import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft

def extract_features(window):
    """Extract time and frequency domain features from a single signal window."""
    features = {}
    features['mean'] = np.mean(window)
    features['std'] = np.std(window)
    features['min'] = np.min(window)
    features['max'] = np.max(window)
    features['range'] = features['max'] - features['min']
    features['skew'] = skew(window)
    features['kurtosis'] = kurtosis(window)

    mean_centered = window - features['mean']
    features['zero_crossing_rate'] = np.sum(np.diff(np.sign(mean_centered)) != 0)
    features['sma'] = np.sum(np.abs(window)) / len(window)

    fft_vals = np.abs(fft(window))
    fft_vals = fft_vals[:len(fft_vals)//2]
    features['dominant_freq_magnitude'] = np.max(fft_vals)
    features['spectral_energy'] = np.sum(fft_vals**2) / len(fft_vals)

    return features


def build_feature_table(raw_signals_dict, signal_types, num_windows):
    """Build a full feature DataFrame from a dict of {signal_name: raw_dataframe}."""
    import pandas as pd
    all_features = []
    for i in range(num_windows):
        row_features = {}
        for sig in signal_types:
            window = raw_signals_dict[sig].iloc[i].values
            feats = extract_features(window)
            for key, value in feats.items():
                row_features[f"{sig}_{key}"] = value
        all_features.append(row_features)
    return pd.DataFrame(all_features)