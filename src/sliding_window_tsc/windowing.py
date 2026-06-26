import numpy as np


def make_window_dataset(X, y, window_size: int, stride: int):
    """
    Convert aeon-format series into fixed-length sliding windows.

    Input X shape is expected to be (n_cases, n_channels, n_timepoints).
    Returns X_windows, y_windows, series_ids.
    """
    if window_size <= 0:
        raise ValueError("window_size must be greater than 0.")
    if stride <= 0:
        raise ValueError("stride must be greater than 0.")
    if X.ndim != 3:
        raise ValueError("X must have shape (n_cases, n_channels, n_timepoints).")

    n_cases = X.shape[0]
    series_length = X.shape[-1]

    if window_size > series_length:
        raise ValueError(
            f"window_size={window_size} is larger than series length={series_length}."
        )

    X_windows = []
    y_windows = []
    series_ids = []

    for series_id in range(n_cases):
        for start in range(0, series_length - window_size + 1, stride):
            end = start + window_size
            X_windows.append(X[series_id, :, start:end])
            y_windows.append(y[series_id])
            series_ids.append(series_id)

    return np.asarray(X_windows), np.asarray(y_windows), np.asarray(series_ids)


def parse_window_sizes(X, window_sizes, percentages: bool = False):
    """Convert absolute window sizes or fractions of series length to integers."""
    series_length = X.shape[-1]

    if percentages:
        sizes = [max(1, int(series_length * float(p))) for p in window_sizes]
    else:
        sizes = [int(w) for w in window_sizes]

    sizes = sorted({w for w in sizes if 0 < w <= series_length})

    if not sizes:
        raise ValueError("No valid window sizes were produced.")

    return sizes
