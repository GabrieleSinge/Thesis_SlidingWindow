import json
from pathlib import Path
import pandas as pd


def expand_range(start: float, end: float, step: float) -> list[float]:
    """
    Expand START END STEP into an inclusive range.
    """

    if step <= 0:
        raise ValueError("STEP must be greater than 0.")

    if end < start:
        raise ValueError("END must be greater than or equal to START.")

    values = []
    current = start

    while current <= end + 1e-12:
        values.append(round(current, 10))
        current += step

    return values

def _safe_float_for_filename(value: float) -> str:
    """
    Convert a float into a filesystem-safe string.

    Example:
        0.5 -> 0p5
        0.25 -> 0p25
    """

    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")

def load_hyperparameters_from_json(path: str | None) -> dict:
    """
    Load classifier hyperparameters from a JSON file.

    Supported formats:

    1. Explicit format:
        {
          "hyperparameters": {
            "param_name": value
          }
        }

    2. Direct format:
        {
          "param_name": value
        }
    """

    if path is None:
        return {}

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Hyperparameter file not found: {path}")

    if file_path.suffix.lower() != ".json":
        raise ValueError(
            f"Unsupported hyperparameter file format: {file_path.suffix}. "
            "Use a .json file."
        )

    with open(file_path, "r") as f:
        content = json.load(f)

    if content is None:
        return {}

    if not isinstance(content, dict):
        raise ValueError(
            f"Hyperparameter file must contain a JSON object, got {type(content)}."
        )

    if "hyperparameters" in content:
        hyperparameters = content["hyperparameters"]
    else:
        hyperparameters = content

    if hyperparameters is None:
        return {}

    if not isinstance(hyperparameters, dict):
        raise ValueError("`hyperparameters` must be a JSON object.")

    return hyperparameters




def _prepare_output_paths(output_dir: str | Path) -> TuningOutputPaths:
    """
    Create and return all output directories used by tuning.
    """

    root = Path(output_dir)

    paths = TuningOutputPaths(
        root=root,
        best_hyperparameters_dir=root / "best_hyperparameters",
        trials_dir=root / "trials",
        trial_results_dir=root / "trial_results",
        index_file=root / "best_hyperparameters_index.csv",
    )

    paths.best_hyperparameters_dir.mkdir(parents=True, exist_ok=True)
    paths.trials_dir.mkdir(parents=True, exist_ok=True)
    paths.trial_results_dir.mkdir(parents=True, exist_ok=True)

    return paths


def _best_hyperparameters_filename(config: WindowStrideConfig) -> str:
    """
    Build the JSON filename for one best-hyperparameter file.
    """

    stride_text = _safe_float_for_filename(config.stride_ratio)

    return (
        f"{config.dataset_name}_{config.classifier_name}"
        f"_window_{config.window_size}"
        f"_stride_{stride_text}"
        f".json"
    )


def _save_json(path: Path, content: dict) -> None:
    """
    Save a dictionary as a pretty JSON file.
    """

    with path.open("w", encoding="utf-8") as f:
        json.dump(content, f, indent=4)


def _clean_error(error):
    """
    Convert pandas NaN errors into None.
    """

    if pd.isna(error):
        return None

    return str(error)


def _score_from_result_row(row: pd.Series, metric: str) -> float:
    """
    Extract the optimization score from one experiment result row.

    Failed rows or NaN metrics receive score 0.0.
    """

    if row["status"] != "ok":
        return 0.0

    value = row[metric]

    if pd.isna(value):
        return 0.0

    return float(value)

IDEAL_CLASSIFIERS = [
    "MiniRocketClassifier",
    "KNeighborsTimeSeriesClassifier",
    "WEASEL",
    "Catch22Classifier",
    "DrCIFClassifier",
    "RDSTClassifier",
    "InceptionTimeClassifier",
]

FAST_TRAINING_CLASSIFIERS = [
    "MiniRocketClassifier",
    "SummaryClassifier",
    "Catch22Classifier",
    "TimeSeriesForestClassifier",
    "RandomIntervalClassifier",
    "KNeighborsTimeSeriesClassifier",
]
