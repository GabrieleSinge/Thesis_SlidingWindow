import numpy as np


def make_window_dataset(X, y, window_size: int, stride: int):
    """
    Convert a time-series classification dataset into a windowed dataset.

    Input X is expected in aeon format:
        (n_cases, n_channels, n_timepoints)

    Output:
        X_windows: (n_windows, n_channels, window_size)
        y_windows: label inherited from the original full series
        series_ids: index of the original time series
    """

    X_windows = []
    y_windows = []
    series_ids = []

    n_cases = X.shape[0]
    series_length = X.shape[-1]

    if window_size > series_length:
        raise ValueError(
            f"window_size={window_size} is larger than series length={series_length}."
        )

    for i in range(n_cases):
        start = 0

        while start + window_size <= series_length:
            end = start + window_size

            X_windows.append(X[i, :, start:end])
            y_windows.append(y[i])
            series_ids.append(i)

            start += stride

    X_windows = np.asarray(X_windows)
    y_windows = np.asarray(y_windows)
    series_ids = np.asarray(series_ids)

    return X_windows, y_windows, series_ids


def parse_window_sizes(X, window_sizes, percentages=False):
    """
    Convert either absolute window sizes or percentages into integer window sizes.
    """

    series_length = X.shape[-1]

    if percentages:
        sizes = [int(series_length * p) for p in window_sizes]
    else:
        sizes = [int(w) for w in window_sizes]

    sizes = sorted(set([w for w in sizes if w > 0 and w <= series_length]))

    if len(sizes) == 0:
        raise ValueError("No valid window sizes were produced.")

    return sizes