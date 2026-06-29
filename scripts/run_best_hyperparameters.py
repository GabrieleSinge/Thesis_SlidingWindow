import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from sliding_window_tsc.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run experiments using the best hyperparameters saved by "
            "tune_per_configuration.py."
        )
    )

    parser.add_argument(
        "--best-dir",
        type=str,
        required=True,
        help="Directory containing best hyperparameter JSON files.",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw",
        help="Root directory containing dataset folders.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where final experiment results will be saved.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--skip-failed-best",
        action="store_true",
        help="Skip JSON files whose best_status is not ok.",
    )

    args = parser.parse_args()

    best_dir = Path(args.best_dir)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(best_dir.glob("*.json"))

    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON files found in {best_dir}")

    all_results = []

    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as f:
            config = json.load(f)

        if args.skip_failed_best and config.get("best_status") != "ok":
            print(f"Skipping failed best configuration: {json_file}")
            continue

        dataset = config["dataset"]
        classifier = config["classifier"]
        window_size = int(config["window_size"])
        stride_ratio = float(config["stride_ratio"])
        hyperparameters = config["best_params"]

        dataset_folder = data_root / dataset

        print()
        print(
            f"Running {dataset} / {classifier} / "
            f"window_size={window_size} / stride_ratio={stride_ratio} / "
            f"params={hyperparameters}"
        )

        result_df = run_experiment(
            dataset_folder=dataset_folder,
            classifier_name=classifier,
            window_sizes=[window_size],
            stride_ratios=[stride_ratio],
            percentages=False,
            random_state=args.random_state,
            hyperparameters=hyperparameters,
        )

        result_df["best_hyperparameters_file"] = str(json_file)
        result_df["tuning_best_value"] = config.get("best_value")
        result_df["tuning_best_trial"] = config.get("best_trial")

        all_results.append(result_df)

    if len(all_results) == 0:
        raise ValueError("No experiments were executed.")

    final_df = pd.concat(all_results, ignore_index=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"best_hyperparameters_run_{timestamp}.csv"

    final_df.to_csv(output_file, index=False)

    print()
    print(f"Saved final results to: {output_file}")


if __name__ == "__main__":
    main()