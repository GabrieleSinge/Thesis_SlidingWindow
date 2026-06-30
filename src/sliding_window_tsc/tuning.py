import json
from dataclasses import dataclass
from pathlib import Path

import optuna
import pandas as pd

from sliding_window_tsc.data_loader import load_dataset
from sliding_window_tsc.experiment import run_experiment
from sliding_window_tsc.windowing import parse_window_sizes
from sliding_window_tsc.utils import _clean_error, _score_from_result_row, _save_json, _best_hyperparameters_filename, _prepare_output_paths
from sliding_window_tsc.hyperparameters_search_space import suggest_hyperparameters


@dataclass(frozen=True)
class WindowStrideConfig:
    """
    One experimental configuration.

    A configuration is uniquely identified by:
        dataset + classifier + window_size + stride_ratio
    """

    dataset_name: str
    classifier_name: str
    window_size: int
    window_percentage: float
    stride_ratio: float
    stride: int


@dataclass(frozen=True)
class TuningOutputPaths:
    """
    Output folders used by the tuning process.
    """

    root: Path
    best_hyperparameters_dir: Path
    trials_dir: Path
    trial_results_dir: Path
    index_file: Path


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------



def _build_window_stride_configs(
    dataset_name: str,
    classifier_name: str,
    window_sizes,
    stride_ratios,
    series_length: int,
) -> list[WindowStrideConfig]:
    """
    Create all window/stride configurations.
    """

    configs = []

    for window_size in window_sizes:
        for stride_ratio in stride_ratios:
            stride = max(1, int(window_size * stride_ratio))
            window_percentage = window_size / series_length

            configs.append(
                WindowStrideConfig(
                    dataset_name=dataset_name,
                    classifier_name=classifier_name,
                    window_size=int(window_size),
                    window_percentage=float(window_percentage),
                    stride_ratio=float(stride_ratio),
                    stride=int(stride),
                )
            )

    return configs


# ---------------------------------------------------------------------
# Single configuration tuning
# ---------------------------------------------------------------------


def _run_single_configuration(
    dataset_folder: str | Path,
    config: WindowStrideConfig,
    hyperparameters: dict,
    random_state: int,
) -> pd.DataFrame:
    """
    Run one experiment for one window_size and one stride_ratio.
    """

    return run_experiment(
        dataset_folder=dataset_folder,
        classifier_name=config.classifier_name,
        window_sizes=[config.window_size],
        stride_ratios=[config.stride_ratio],
        percentages=False,
        random_state=random_state,
        hyperparameters=hyperparameters,
    )


def _set_single_config_trial_attrs(
    trial,
    config: WindowStrideConfig,
    row: pd.Series,
    hyperparameters: dict,
    score: float,
) -> None:
    """
    Store useful metadata inside the Optuna trial.
    """

    trial.set_user_attr("hyperparameters", hyperparameters)
    trial.set_user_attr("dataset", config.dataset_name)
    trial.set_user_attr("classifier", config.classifier_name)
    trial.set_user_attr("window_size", config.window_size)
    trial.set_user_attr("window_percentage", config.window_percentage)
    trial.set_user_attr("stride", config.stride)
    trial.set_user_attr("stride_ratio", config.stride_ratio)
    trial.set_user_attr("status", row["status"])
    trial.set_user_attr("error", _clean_error(row["error"]))
    trial.set_user_attr("score", score)


def _objective_single_configuration(
    trial,
    dataset_folder: str | Path,
    config: WindowStrideConfig,
    metric: str,
    random_state: int,
) -> float:
    """
    Optuna objective for one specific window/stride configuration.

    One trial evaluates one hyperparameter set on exactly one:
        window_size × stride_ratio
    """

    hyperparameters = suggest_hyperparameters(
        trial=trial,
        classifier_name=config.classifier_name,
    )

    results_df = _run_single_configuration(
        dataset_folder=dataset_folder,
        config=config,
        hyperparameters=hyperparameters,
        random_state=random_state,
    )

    if metric not in results_df.columns:
        raise ValueError(f"Unknown metric column: {metric}")

    row = results_df.iloc[0]
    score = _score_from_result_row(row, metric)

    _set_single_config_trial_attrs(
        trial=trial,
        config=config,
        row=row,
        hyperparameters=hyperparameters,
        score=score,
    )

    return score


def _build_best_result_dict(
    config: WindowStrideConfig,
    study,
    metric: str,
    n_trials: int,
) -> dict:
    """
    Build the dictionary saved as best-hyperparameter JSON.
    """

    best_trial = study.best_trial

    return {
        "dataset": config.dataset_name,
        "classifier": config.classifier_name,
        "window_size": config.window_size,
        "window_percentage": config.window_percentage,
        "stride_ratio": config.stride_ratio,
        "stride": config.stride,
        "metric": metric,
        "best_value": float(study.best_value),
        "best_params": best_trial.params,
        "best_trial": int(best_trial.number),
        "best_status": best_trial.user_attrs.get("status"),
        "best_error": best_trial.user_attrs.get("error"),
        "n_trials": int(n_trials),
    }


def _build_index_row(
    config: WindowStrideConfig,
    study,
    metric: str,
    best_file: Path,
    trials_file: Path,
) -> dict:
    """
    Build one row of best_hyperparameters_index.csv.
    """

    best_trial = study.best_trial

    return {
        "dataset": config.dataset_name,
        "classifier": config.classifier_name,
        "window_size": config.window_size,
        "window_percentage": config.window_percentage,
        "stride_ratio": config.stride_ratio,
        "stride": config.stride,
        "metric": metric,
        "best_value": float(study.best_value),
        "best_params": json.dumps(best_trial.params, sort_keys=True),
        "best_trial": int(best_trial.number),
        "best_status": best_trial.user_attrs.get("status"),
        "best_error": best_trial.user_attrs.get("error"),
        "best_file": str(best_file),
        "trials_file": str(trials_file),
    }


def tune_classifier_per_configuration(
    dataset_folder: str | Path,
    classifier_name: str,
    window_sizes,
    stride_ratios,
    percentages: bool = False,
    random_state: int = 42,
    metric: str = "series_macro_f1",
    n_trials: int = 20,
    output_dir: str | Path = "results/tuning/per_config",
) -> pd.DataFrame:
    """
    Tune hyperparameters separately for each window_size and stride_ratio.

    This function creates one JSON file for each configuration:
        dataset + classifier + window_size + stride_ratio
    """

    output_paths = _prepare_output_paths(output_dir)

    X_train, _y_train, _X_test, _y_test, dataset_name = load_dataset(dataset_folder)
    series_length = X_train.shape[-1]

    parsed_window_sizes = parse_window_sizes(
        X_train,
        window_sizes=window_sizes,
        percentages=percentages,
    )

    configs = _build_window_stride_configs(
        dataset_name=dataset_name,
        classifier_name=classifier_name,
        window_sizes=parsed_window_sizes,
        stride_ratios=list(stride_ratios),
        series_length=series_length,
    )

    index_rows = []

    for config_index, config in enumerate(configs, start=1):
        print()
        print(
            f"[{config_index}/{len(configs)}] "
            f"Tuning {config.dataset_name} / {config.classifier_name} / "
            f"window_size={config.window_size} / "
            f"stride_ratio={config.stride_ratio}"
        )

        study = optuna.create_study(direction="maximize")

        study.optimize(
            lambda trial: _objective_single_configuration(
                trial=trial,
                dataset_folder=dataset_folder,
                config=config,
                metric=metric,
                random_state=random_state,
            ),
            n_trials=n_trials,
        )

        base_filename = _best_hyperparameters_filename(config)

        best_file = output_paths.best_hyperparameters_dir / base_filename
        trials_file = output_paths.trials_dir / base_filename.replace(
            ".json",
            "_trials.csv",
        )

        trials_df = study.trials_dataframe()
        trials_df.to_csv(trials_file, index=False)

        best_result = _build_best_result_dict(
            config=config,
            study=study,
            metric=metric,
            n_trials=n_trials,
        )

        _save_json(best_file, best_result)

        index_row = _build_index_row(
            config=config,
            study=study,
            metric=metric,
            best_file=best_file,
            trials_file=trials_file,
        )

        index_rows.append(index_row)

        index_df = pd.DataFrame(index_rows)
        index_df.to_csv(output_paths.index_file, index=False)

    return pd.DataFrame(index_rows)


# ---------------------------------------------------------------------
# Global tuning across many windows/strides
# ---------------------------------------------------------------------


def _set_global_trial_attrs(
    trial,
    results_df: pd.DataFrame,
    hyperparameters: dict,
    metric: str,
) -> float:
    """
    Store metadata for a global tuning trial and return its score.

    A global trial evaluates one hyperparameter set across many
    window/stride configurations and returns the best valid score.
    """

    trial.set_user_attr("hyperparameters", hyperparameters)

    n_ok = int((results_df["status"] == "ok").sum())
    n_failed = int((results_df["status"] == "failed").sum())

    trial.set_user_attr("n_ok", n_ok)
    trial.set_user_attr("n_failed", n_failed)

    failed_results = results_df[results_df["status"] == "failed"]

    if not failed_results.empty:
        unique_errors = (
            failed_results["error"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        trial.set_user_attr("failed_errors", unique_errors[:5])

    valid_results = results_df[results_df["status"] == "ok"]

    if valid_results.empty:
        trial.set_user_attr("best_score", 0.0)
        trial.set_user_attr("best_window_size", None)
        trial.set_user_attr("best_stride_ratio", None)
        return 0.0

    best_row = valid_results.sort_values(metric, ascending=False).iloc[0]
    best_score = float(best_row[metric])

    trial.set_user_attr("best_score", best_score)
    trial.set_user_attr("best_window_size", int(best_row["window_size"]))
    trial.set_user_attr("best_stride_ratio", float(best_row["stride_ratio"]))

    return best_score


def _objective_global(
    trial,
    dataset_folder: str | Path,
    classifier_name: str,
    window_sizes,
    stride_ratios,
    percentages: bool,
    random_state: int,
    metric: str,
    output_paths: TuningOutputPaths,
) -> float:
    """
    Optuna objective for global tuning.

    One trial evaluates one hyperparameter set over many:
        window_size × stride_ratio
    configurations.
    """

    hyperparameters = suggest_hyperparameters(
        trial=trial,
        classifier_name=classifier_name,
    )

    results_df = run_experiment(
        dataset_folder=dataset_folder,
        classifier_name=classifier_name,
        window_sizes=window_sizes,
        stride_ratios=stride_ratios,
        percentages=percentages,
        random_state=random_state,
        hyperparameters=hyperparameters,
    )

    if metric not in results_df.columns:
        raise ValueError(f"Unknown metric column: {metric}")

    results_df["trial_number"] = trial.number
    results_df["trial_hyperparameters"] = json.dumps(
        hyperparameters,
        sort_keys=True,
    )

    trial_results_file = output_paths.trial_results_dir / f"trial_{trial.number:04d}.csv"
    results_df.to_csv(trial_results_file, index=False)

    trial.set_user_attr("trial_results_file", str(trial_results_file))

    return _set_global_trial_attrs(
        trial=trial,
        results_df=results_df,
        hyperparameters=hyperparameters,
        metric=metric,
    )


def tune_classifier(
    dataset_folder: str | Path,
    classifier_name: str,
    window_sizes,
    stride_ratios=None,
    percentages: bool = False,
    random_state: int = 42,
    metric: str = "series_macro_f1",
    n_trials: int = 20,
    output_dir: str | Path = "results/tuning",
):
    """
    Run global Optuna hyperparameter search for one classifier on one dataset.

    This optimizes one hyperparameter set across all provided
    window_size × stride_ratio configurations.
    """

    if stride_ratios is None:
        stride_ratios = [0.5]

    output_paths = _prepare_output_paths(output_dir)

    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: _objective_global(
            trial=trial,
            dataset_folder=dataset_folder,
            classifier_name=classifier_name,
            window_sizes=window_sizes,
            stride_ratios=stride_ratios,
            percentages=percentages,
            random_state=random_state,
            metric=metric,
            output_paths=output_paths,
        ),
        n_trials=n_trials,
    )

    dataset_name = Path(dataset_folder).name

    trials_file = output_paths.root / f"{dataset_name}_{classifier_name}_optuna_trials.csv"
    best_file = output_paths.root / f"{dataset_name}_{classifier_name}_best_params.json"

    trials_df = study.trials_dataframe()
    trials_df.to_csv(trials_file, index=False)

    best_result = {
        "dataset": dataset_name,
        "classifier": classifier_name,
        "metric": metric,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "best_trial": int(study.best_trial.number),
        "best_window_size": study.best_trial.user_attrs.get("best_window_size"),
        "best_stride_ratio": study.best_trial.user_attrs.get("best_stride_ratio"),
        "n_ok": study.best_trial.user_attrs.get("n_ok"),
        "n_failed": study.best_trial.user_attrs.get("n_failed"),
    }

    _save_json(best_file, best_result)

    return study