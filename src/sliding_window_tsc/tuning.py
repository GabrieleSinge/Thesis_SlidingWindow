import json
from pathlib import Path

import optuna

from sliding_window_tsc.experiment import run_experiment


def suggest_hyperparameters(trial, classifier_name: str):
    """Define Optuna search spaces for supported classifiers."""
    if classifier_name == "MiniRocketClassifier":
        return {
            "num_kernels": trial.suggest_categorical(
                "num_kernels", [1000, 5000, 10000]
            )
        }

    if classifier_name == "KNeighborsTimeSeriesClassifier":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 7),
            "distance": trial.suggest_categorical("distance", ["euclidean", "dtw"]),
        }

    if classifier_name == "TimeSeriesForestClassifier":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        }

    if classifier_name == "RandomIntervalClassifier":
        return {
            "n_intervals": trial.suggest_categorical(
                "n_intervals", ["sqrt", "log", 50, 100]
            ),
        }

    if classifier_name == "DrCIFClassifier":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "n_intervals": trial.suggest_int("n_intervals", 4, 16),
        }

    if classifier_name in {"Catch22Classifier", "SummaryClassifier"}:
        return {}

    raise ValueError(f"No Optuna search space defined for classifier: {classifier_name}")


def objective(
    trial,
    dataset_folder: str,
    classifier_name: str,
    window_sizes,
    stride_ratio: float,
    percentages: bool,
    random_state: int,
    metric: str,
):
    """Optuna objective: evaluate one hyperparameter configuration."""
    hyperparameters = suggest_hyperparameters(trial, classifier_name)

    results_df = run_experiment(
        dataset_folder=dataset_folder,
        classifier_name=classifier_name,
        window_sizes=window_sizes,
        stride_ratio=stride_ratio,
        percentages=percentages,
        random_state=random_state,
        hyperparameters=hyperparameters,
    )

    if metric not in results_df.columns:
        raise ValueError(f"Unknown metric column: {metric}")

    valid_results = results_df[results_df["status"] == "ok"]
    if valid_results.empty:
        return 0.0

    best_row = valid_results.sort_values(metric, ascending=False).iloc[0]
    best_score = float(best_row[metric])

    trial.set_user_attr("hyperparameters", hyperparameters)
    trial.set_user_attr("best_score", best_score)
    trial.set_user_attr("best_window_size", int(best_row["window_size"]))

    return best_score


def tune_classifier(
    dataset_folder: str,
    classifier_name: str,
    window_sizes,
    stride_ratio: float = 0.5,
    percentages: bool = False,
    random_state: int = 42,
    metric: str = "series_macro_f1",
    n_trials: int = 20,
    output_dir: str = "results/tuning",
):
    """Run Optuna hyperparameter search for one classifier on one dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(
            trial=trial,
            dataset_folder=dataset_folder,
            classifier_name=classifier_name,
            window_sizes=window_sizes,
            stride_ratio=stride_ratio,
            percentages=percentages,
            random_state=random_state,
            metric=metric,
        ),
        n_trials=n_trials,
    )

    trials_df = study.trials_dataframe()
    dataset_name = Path(dataset_folder).name

    trials_file = output_path / f"{dataset_name}_{classifier_name}_optuna_trials.csv"
    best_file = output_path / f"{dataset_name}_{classifier_name}_best_params.json"

    trials_df.to_csv(trials_file, index=False)

    best_result = {
        "dataset": dataset_name,
        "classifier": classifier_name,
        "metric": metric,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial": study.best_trial.number,
        "best_window_size": study.best_trial.user_attrs.get("best_window_size"),
    }

    with best_file.open("w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=4)

    return study
