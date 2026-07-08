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
    
    if classifier_name == "WEASEL":
        return {
            "window_inc": trial.suggest_categorical(
                "window_inc",
                [1, 2, 3, 4],
            ),
            "alphabet_size": trial.suggest_categorical(
                "alphabet_size",
                [4, 6, 8],
            ),
            "p_threshold": trial.suggest_categorical(
                "p_threshold",
                [0.01, 0.025, 0.05, 0.1, 0.2],
            ),
            "anova": trial.suggest_categorical(
                "anova",
                [True, False],
            ),
            "bigrams": trial.suggest_categorical(
                "bigrams",
                [True, False],
            ),
            "binning_strategy": trial.suggest_categorical(
                "binning_strategy",
                [
                    "information-gain",
                    "equi-depth",
                    "equi-width",
                ],
            ),
            "feature_selection": "chi2",
            "support_probabilities": False,
            "n_jobs": 4,
        }
    if classifier_name == "InceptionTimeClassifier":
        return {
            "n_classifiers": trial.suggest_categorical(
                "n_classifiers",
                [1, 3, 5],
            ),
            "n_filters": trial.suggest_categorical(
                "n_filters",
                [16, 32, 64],
            ),
            "kernel_size": trial.suggest_categorical(
                "kernel_size",
                [20, 40, 60],
            ),
            "depth": trial.suggest_categorical(
                "depth",
                [4, 6, 8],
            ),
            "n_conv_per_layer": trial.suggest_categorical(
                "n_conv_per_layer",
                [2, 3],
            ),
            "bottleneck_size": trial.suggest_categorical(
                "bottleneck_size",
                [16, 32, 64],
            ),
            "use_residual": trial.suggest_categorical(
                "use_residual",
                [True, False],
            ),
            "use_bottleneck": trial.suggest_categorical(
                "use_bottleneck",
                [True, False],
            ),
            "use_max_pooling": trial.suggest_categorical(
                "use_max_pooling",
                [True, False],
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size",
                [16, 32, 64],
            ),
            "n_epochs": trial.suggest_categorical(
                "n_epochs",
                [50, 100, 150],
            ),
            "verbose": False,
            "random_state": 42,
            "save_last_model": False,
            "save_best_model": False,
            "save_init_model": False,
        }
    
    if classifier_name == "TimeSeriesForestClassifier":
        return {
            "n_estimators": trial.suggest_categorical(
                "n_estimators",
                [100, 200, 300],
            ),
            "n_intervals": trial.suggest_categorical(
                "n_intervals",
                ["sqrt", "log", 20, 50, 100],
            ),
            "min_interval_length": trial.suggest_categorical(
                "min_interval_length",
                [3, 5, 10],
            ),
            "n_jobs": 4,
            "random_state": 42,
        }

    if classifier_name in {"Catch22Classifier", "SummaryClassifier"}:
        return {}

    raise ValueError(
        f"No Optuna search space defined for classifier: {classifier_name}"
    )