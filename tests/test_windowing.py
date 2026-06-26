import numpy as np
import pytest

from sliding_window_tsc.windowing import make_window_dataset


def test_make_window_dataset_single_series():
    X = np.array([
        [[1, 2, 3, 4, 5, 6]]
    ])

    y = np.array(["A"])

    X_windows, y_windows, series_ids = make_window_dataset(
        X,
        y,
        window_size=3,
        stride=2,
    )

    assert X_windows.shape == (2, 1, 3)

    np.testing.assert_array_equal(
        X_windows[0],
        np.array([[1, 2, 3]])
    )

    np.testing.assert_array_equal(
        X_windows[1],
        np.array([[3, 4, 5]])
    )

    np.testing.assert_array_equal(y_windows, np.array(["A", "A"]))
    np.testing.assert_array_equal(series_ids, np.array([0, 0]))

def test_make_window_dataset_multiple_series():
    X = np.array([
        [[1, 2, 3, 4]],
        [[5, 6, 7, 8]],
    ])

    y = np.array(["A", "B"])

    X_windows, y_windows, series_ids = make_window_dataset(
        X,
        y,
        window_size=2,
        stride=2,
    )

    assert X_windows.shape == (4, 1, 2)

    np.testing.assert_array_equal(y_windows, np.array(["A", "A", "B", "B"]))
    np.testing.assert_array_equal(series_ids, np.array([0, 0, 1, 1]))


def test_make_window_dataset_window_too_large():
    X = np.array([
        [[1, 2, 3]]
    ])

    y = np.array(["A"])

    with pytest.raises(ValueError):
        make_window_dataset(
            X,
            y,
            window_size=10,
            stride=1,
        )
