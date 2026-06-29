import argparse
from pathlib import Path
from datetime import datetime
import json

from sliding_window_tsc.utils import load_hyperparameters_from_json
from sliding_window_tsc.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Run aeon time-series classification experiments with sliding windows."
    )

    parser.add_argument(
        "--dataset-folder",
        type=str,
        required=True,
        help="Path to dataset folder containing *_TRAIN.ts and *_TEST.ts files.",
    )

    parser.add_argument(
        "--classifier",
        type=str,
        required=True,
        help="Name of the aeon classifier to use.",
    )

    parser.add_argument(
        "--window-sizes",
        type=float,
        nargs="+",
        required=True,
        help="List of window sizes. Absolute values by default, percentages if --percentages is used.",
    )

    parser.add_argument(
        "--percentages",
        action="store_true",
        help="Interpret window sizes as percentages of the original series length.",
    )

    parser.add_argument(
        "--stride-ratio",
        type=float,
        default=0.5,
        help="Stride as a fraction of the window size. Used when --stride-ratios is not provided.",
    )

    parser.add_argument(
        "--stride-ratios",
        type=float,
        nargs="+",
        default=None,
        help="List of stride ratios to evaluate for each window size.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where results will be saved.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--hyperparameters-file",
        type=str,
        default=None,
        help="Path to a JSON file containing classifier hyperparameters.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hyperparameters = load_hyperparameters_from_json(args.hyperparameters_file)

    results_df = run_experiment(
        dataset_folder=args.dataset_folder,
        classifier_name=args.classifier,
        window_sizes=args.window_sizes,
        stride_ratio=args.stride_ratio,
        stride_ratios=args.stride_ratios,
        percentages=args.percentages,
        random_state=args.random_state,
        hyperparameters=hyperparameters,
    )

    dataset_name = Path(args.dataset_folder).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = output_dir / f"{dataset_name}_{args.classifier}_{timestamp}.csv"

    results_df.to_csv(output_file, index=False)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
