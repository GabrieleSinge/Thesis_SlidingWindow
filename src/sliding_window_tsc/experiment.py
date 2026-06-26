import json
import time
from itertools import product
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from sliding_window_tsc.classifiers import get_classifier_by_name
from sliding_window_tsc.data_loader import load_dataset
from sliding_window_tsc.metrics import (
    aggregate_predictions_by_series,
    compute_classification_metrics,
)
from sliding_window_tsc.windowing import make_window_dataset, parse_window_sizes


RESULT_COLUMNS = [
    "dataset",
    "classifier",
    "classifier_hyperparameters",
    "window_size",
    "window_percentage",
    "stride",
    "stride_ratio",
    "n_train_windows",
    "n_test_windows",
    "window_accuracy",
    "window_balanced_accuracy",
    "window_macro_f1",
    "series_accuracy",
    "series_balanced_accuracy",
    "series_macro_f1",
    "fit_time_sec",
    "predict_time_sec",
    "total_time_sec",
    "random_state",
    "status",
    "error",
]


def _base_result(
    dataset_name,
    classifier_name,
    hyperparameters,
    window_size,
    series_length,
    stride,
    stride_ratio,
    random_state,
):
    return {
        "dataset": dataset_name,
        "classifier": classifier_name,
        "classifier_hyperparameters": json.dumps(hyperparameters or {}, sort_keys=True),
        "window_size": window_size,
        "window_percentage": window_size / series_length,
        "stride": stride,
        "stride_ratio": stride_ratio,
        "n_train_windows": None,
        "n_test_windows": None,
        "window_accuracy": None,
        "window_balanced_accuracy": None,
        "window_macro_f1": None,
        "series_accuracy": None,
        "series_balanced_accuracy": None,
        "series_macro_f1": None,
        "fit_time_sec": None,
        "predict_time_sec": None,
        "total_time_sec": None,
        "random_state": random_state,
        "status": "failed",
        "error": None,
    }


def run_experiment(
    dataset_folder: str | Path,
    classifier_name: str,
    window_sizes,
    stride_ratio: float = 0.5,
    stride_ratios=None,
    percentages: bool = False,
    random_state: int = 42,
    hyperparameters: dict | None = None,
):
    """Run one classifier over sliding window sizes and stride ratios."""
    X_train, y_train, X_test, y_test, dataset_name = load_dataset(dataset_folder)
    series_length = X_train.shape[-1]

    parsed_window_sizes = parse_window_sizes(
        X_train,
        window_sizes=window_sizes,
        percentages=percentages,
    )
    parsed_stride_ratios = list(stride_ratios) if stride_ratios is not None else [stride_ratio]
    configurations = list(product(parsed_window_sizes, parsed_stride_ratios))

    results = []

    progress = tqdm(
        configurations,
        desc=f"{dataset_name} / {classifier_name}",
        unit="config",
    )

    for window_size, current_stride_ratio in progress:
        stride = max(1, int(window_size * current_stride_ratio))
        progress.set_postfix(
            window_size=window_size,
            stride=stride,
            stride_ratio=current_stride_ratio,
            refresh=False,
        )
        result = _base_result(
            dataset_name=dataset_name,
            classifier_name=classifier_name,
            hyperparameters=hyperparameters,
            window_size=window_size,
            series_length=series_length,
            stride=stride,
            stride_ratio=current_stride_ratio,
            random_state=random_state,
        )

        total_start = time.perf_counter()

        try:
            X_train_w, y_train_w, _train_series_ids = make_window_dataset(
                X_train,
                y_train,
                window_size=window_size,
                stride=stride,
            )
            X_test_w, y_test_w, test_series_ids = make_window_dataset(
                X_test,
                y_test,
                window_size=window_size,
                stride=stride,
            )

            clf = get_classifier_by_name(
                classifier_name=classifier_name,
                random_state=random_state,
                hyperparameters=hyperparameters,
            )

            fit_start = time.perf_counter()
            clf.fit(X_train_w, y_train_w)
            fit_time = time.perf_counter() - fit_start

            predict_start = time.perf_counter()
            y_pred_w = clf.predict(X_test_w)
            predict_time = time.perf_counter() - predict_start

            y_test_series, y_pred_series = aggregate_predictions_by_series(
                y_test_w,
                y_pred_w,
                test_series_ids,
            )

            result.update(
                {
                    "n_train_windows": len(X_train_w),
                    "n_test_windows": len(X_test_w),
                    "fit_time_sec": fit_time,
                    "predict_time_sec": predict_time,
                    "total_time_sec": time.perf_counter() - total_start,
                    "status": "ok",
                    "error": None,
                }
            )
            result.update(compute_classification_metrics(y_test_w, y_pred_w, "window"))
            result.update(
                compute_classification_metrics(y_test_series, y_pred_series, "series")
            )

        except Exception as exc:
            result["total_time_sec"] = time.perf_counter() - total_start
            result["status"] = "failed"
            result["error"] = str(exc)

        progress.set_postfix(
            window_size=window_size,
            stride=stride,
            stride_ratio=current_stride_ratio,
            status=result["status"],
            refresh=False,
        )
        results.append(result)

    return pd.DataFrame(results, columns=RESULT_COLUMNS)
