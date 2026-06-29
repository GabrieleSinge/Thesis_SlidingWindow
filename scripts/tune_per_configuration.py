import argparse

from sliding_window_tsc.tuning import tune_classifier_per_configuration
from sliding_window_tsc.utils import expand_range


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Tune classifier hyperparameters separately for each "
            "window size and stride ratio."
        )
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
        nargs=3,
        required=True,
        metavar=("START", "END", "STEP"),
        help=(
            "Window size range as START END STEP. "
            "Absolute values by default, percentages if --percentages is used."
        ),
    )

    parser.add_argument(
        "--percentages",
        action="store_true",
        help="Interpret window sizes as percentages.",
    )

    parser.add_argument(
        "--stride-ratios",
        type=float,
        nargs="+",
        required=True,
        help="List of stride ratios to evaluate.",
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
        help="Number of Optuna trials for each window/stride configuration.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tuning/per_config",
        help="Output directory for tuning results.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    window_sizes = expand_range(
        start=args.window_sizes[0],
        end=args.window_sizes[1],
        step=args.window_sizes[2],
    )

    index_df = tune_classifier_per_configuration(
        dataset_folder=args.dataset_folder,
        classifier_name=args.classifier,
        window_sizes=window_sizes,
        stride_ratios=args.stride_ratios,
        percentages=args.percentages,
        random_state=args.random_state,
        metric=args.metric,
        n_trials=args.n_trials,
        output_dir=args.output_dir,
    )

    print()
    print("Tuning completed.")
    print(f"Saved {len(index_df)} best hyperparameter files.")
    print(f"Index file: {args.output_dir}/best_hyperparameters_index.csv")


if __name__ == "__main__":
    main()