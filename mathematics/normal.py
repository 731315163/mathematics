import numpy as np


def zscore(series):
    series = np.asarray(series)
    return (series - np.mean (series)) / np.std(series)



def min_max(series):
    series = np.asarray(series)
    min_val, max_val = np.min(series), np.max(series)
    normalized = (series - min_val) / (max_val - min_val)
    return normalized, min_val, max_val