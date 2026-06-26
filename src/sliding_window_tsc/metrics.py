from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def compute_classification_metrics(y_true, y_pred, prefix: str = ""):
    """Compute accuracy, balanced accuracy, and macro F1."""
    names = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }

    if not prefix:
        return names

    return {f"{prefix}_{name}": value for name, value in names.items()}


def aggregate_predictions_by_series(y_true_window, y_pred_window, series_ids):
    """
    Aggregate window-level predictions to one label per original series.

    The true label is taken as the first true window label for each series.
    The predicted label is selected by majority voting, with deterministic
    tie-breaking by label string representation.
    """
    y_true_window = np.asarray(y_true_window)
    y_pred_window = np.asarray(y_pred_window)
    series_ids = np.asarray(series_ids)

    true_by_series = []
    pred_by_series = []

    for series_id in np.unique(series_ids):
        mask = series_ids == series_id
        true_by_series.append(y_true_window[mask][0])

        counts = Counter(y_pred_window[mask])
        pred_label = max(counts.items(), key=lambda item: (item[1], str(item[0])))[0]
        pred_by_series.append(pred_label)

    return np.asarray(true_by_series), np.asarray(pred_by_series)
