import argparse

from sliding_window_tsc.tuning import tune_classifier


def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter search for an aeon classifier."
    )

    parser.add_argument(
        "--dataset-folder",
        type=str,
        required=True,
        help="Dataset folder containing *_TRAIN.ts and *_TEST.ts files.",
    )

    parser.add_argument(
        "--classifier",
        type=str,
        required=True,
        help="Aeon classifier name.",
    )

    parser.add_argument(
        "--window-sizes",
        type=float,
        nargs="+",
        required=True,
        help="Window sizes or percentages.",
    )

    parser.add_argument(
        "--percentages",
        action="store_true",
        help="Interpret window sizes as percentages.",
    )

    parser.add_argument(
        "--stride-ratio",
        type=float,
        default=0.5,
        help="Stride ratio with respect to window size.",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="series_macro_f1",
        help="Metric to maximize.",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tuning",
        help="Output directory for tuning results.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    study = tune_classifier(
        dataset_folder=args.dataset_folder,
        classifier_name=args.classifier,
        window_sizes=args.window_sizes,
        stride_ratio=args.stride_ratio,
        percentages=args.percentages,
        random_state=args.random_state,
        metric=args.metric,
        n_trials=args.n_trials,
        output_dir=args.output_dir,
    )

    print("\nBest trial")
    print(f"Value: {study.best_value}")
    print(f"Params: {study.best_params}")
    print(f"Trial number: {study.best_trial.number}")


if __name__ == "__main__":
    main()