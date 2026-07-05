def suggest_hyperparameters(trial, classifier_name: str) -> dict:
    """
    Define the Optuna search space for each supported classifier.
    """

    if classifier_name == "MiniRocketClassifier":
        return {
            "n_kernels": trial.suggest_int(
                "n_kernels",
                500,
                3000,
                step=50,
            ),
            "max_dilations_per_kernel": trial.suggest_categorical(
                "max_dilations_per_kernel",
                [16, 32, 64],
            ),
        }

    if classifier_name == "KNeighborsTimeSeriesClassifier":
        return {
            "n_neighbors": trial.suggest_categorical(
                "n_neighbors",
                [1, 3, 5, 7, 9],
            ),
            "weights": trial.suggest_categorical(
                "weights",
                ["uniform", "distance"],
            ),
            "distance": trial.suggest_categorical(
                "distance",
                ["euclidean", "dtw", "ddtw"],
            ),
            "n_jobs": -1,
        }

    if classifier_name == "TimeSeriesForestClassifier":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                50,
                300,
                step=50,
            ),
        }

    if classifier_name == "RandomIntervalClassifier":
        return {
            "n_intervals": trial.suggest_categorical(
                "n_intervals",
                ["sqrt", "log", 50, 100],
            ),
        }

    if classifier_name == "DrCIFClassifier":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                50,
                300,
                step=50,
            ),
            "n_intervals": trial.suggest_int(
                "n_intervals",
                4,
                16,
            ),
        }

    if classifier_name in {"Catch22Classifier", "SummaryClassifier"}:
        return {}

    raise ValueError(
        f"No Optuna search space defined for classifier: {classifier_name}"
    )