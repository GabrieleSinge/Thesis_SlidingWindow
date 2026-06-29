import json
from pathlib import Path


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
