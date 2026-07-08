import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "thesis_sliding_window_matplotlib"),
)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASSIFIER_LABELS = {
    "KNeighborsTimeSeriesClassifier": "KNN",
    "MiniRocketClassifier": "MiniRocket",
    "MiniRocket": "MiniRocket",
    "TimeSeriesForestClassifier": "TSF",
    "WEASEL": "WEASEL",
    "InceptionTimeClassifier": "InceptionTime",
}

RUN_TYPE_STYLES = {
    "Untuned": "--",
    "Tuned": "-",
}


def _latest_complete_comparison_dir(classifier_dir: Path) -> Path | None:
    comparison_dirs = sorted(
        [path for path in classifier_dir.glob("comparison*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for comparison_dir in comparison_dirs:
        has_macro = any(comparison_dir.glob("*_plot_input_macro_f1.csv"))
        has_phase_times = any(comparison_dir.glob("*_phase_total_times.csv"))
        has_trial_times = any(comparison_dir.glob("*_tuning_trial_total_times.csv"))

        if has_macro and has_phase_times and has_trial_times:
            return comparison_dir

    return None


def discover_comparison_dirs(results_dir: Path) -> dict[str, Path]:
    comparison_dirs = {}

    for classifier_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        comparison_dir = _latest_complete_comparison_dir(classifier_dir)

        if comparison_dir is not None:
            comparison_dirs[classifier_dir.name] = comparison_dir

    return comparison_dirs


def _classifier_label(classifier: str) -> str:
    return CLASSIFIER_LABELS.get(classifier, classifier)


def load_macro_f1(comparison_dirs: dict[str, Path]) -> pd.DataFrame:
    frames = []

    for classifier_dir_name, comparison_dir in comparison_dirs.items():
        macro_files = sorted(comparison_dir.glob("*_plot_input_macro_f1.csv"))

        for macro_file in macro_files:
            frame = pd.read_csv(macro_file)
            frame = frame[frame["status"] == "ok"].copy()
            frame["comparison_dir"] = str(comparison_dir)
            frame["classifier_dir"] = classifier_dir_name
            frame["classifier_label"] = frame["classifier"].map(_classifier_label)
            frames.append(frame)

    if not frames:
        raise FileNotFoundError("No *_plot_input_macro_f1.csv files found.")

    return pd.concat(frames, ignore_index=True)


def load_phase_times(comparison_dirs: dict[str, Path]) -> pd.DataFrame:
    frames = []

    for classifier_dir_name, comparison_dir in comparison_dirs.items():
        phase_files = sorted(comparison_dir.glob("*_phase_total_times.csv"))

        for phase_file in phase_files:
            frame = pd.read_csv(phase_file)
            frame["classifier_dir"] = classifier_dir_name
            frame["classifier_label"] = _classifier_label(classifier_dir_name)
            frame["comparison_dir"] = str(comparison_dir)
            frames.append(frame)

    if not frames:
        raise FileNotFoundError("No *_phase_total_times.csv files found.")

    return pd.concat(frames, ignore_index=True)


def load_tuning_trial_times(comparison_dirs: dict[str, Path]) -> pd.DataFrame:
    frames = []

    for classifier_dir_name, comparison_dir in comparison_dirs.items():
        trial_files = sorted(comparison_dir.glob("*_tuning_trial_total_times.csv"))

        for trial_file in trial_files:
            frame = pd.read_csv(trial_file)
            frame["classifier_dir"] = classifier_dir_name
            frame["classifier_label"] = _classifier_label(classifier_dir_name)
            frame["comparison_dir"] = str(comparison_dir)
            frames.append(frame)

    if not frames:
        raise FileNotFoundError("No *_tuning_trial_total_times.csv files found.")

    return pd.concat(frames, ignore_index=True)


def plot_macro_f1_by_stride(macro_df: pd.DataFrame, output_dir: Path) -> Path:
    strides = sorted(macro_df["stride_ratio"].dropna().unique())
    classifiers = sorted(macro_df["classifier_label"].dropna().unique())
    color_map = dict(zip(classifiers, plt.cm.tab10(np.linspace(0, 1, len(classifiers)))))

    fig, axes = plt.subplots(
        1,
        len(strides),
        figsize=(6 * len(strides), 5),
        sharey=True,
        constrained_layout=True,
    )

    if len(strides) == 1:
        axes = [axes]

    for ax, stride in zip(axes, strides):
        stride_df = macro_df[macro_df["stride_ratio"] == stride].copy()

        for (classifier, run_type), group in stride_df.groupby(
            ["classifier_label", "run_type"]
        ):
            group = group.sort_values("window_percentage")
            ax.plot(
                group["window_percentage"] * 100,
                group["series_macro_f1"],
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                linestyle=RUN_TYPE_STYLES.get(run_type, "-"),
                color=color_map[classifier],
                label=f"{classifier} {run_type}",
            )

        ax.set_title(f"Stride {stride:g}")
        ax.set_xlabel("Window size (%)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Series macro-F1")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        ncol=1,
        bbox_to_anchor=(1.01, 0.5),
    )

    output_path = output_dir / "overall_macro_f1_by_stride.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_phase_time_bars(phase_df: pd.DataFrame, output_dir: Path) -> Path:
    plot_df = (
        phase_df.pivot_table(
            index="classifier_label",
            columns="phase",
            values="total_time_hour",
            aggfunc="sum",
        )
        .fillna(0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    phases = list(plot_df.columns)
    positions = np.arange(len(plot_df.index))
    width = 0.8 / max(1, len(phases))
    colors = plt.cm.Set2(np.linspace(0, 1, len(phases)))

    for index, phase in enumerate(phases):
        offset = (index - (len(phases) - 1) / 2) * width
        ax.bar(
            positions + offset,
            plot_df[phase],
            width=width,
            label=phase,
            color=colors[index],
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(plot_df.index)
    ax.set_ylabel("Total time (hours)")
    ax.set_title("Total execution time by classifier and phase")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Phase")
    fig.tight_layout()

    output_path = output_dir / "overall_phase_total_times_hours.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_per_configuration_time_histograms(macro_df: pd.DataFrame, output_dir: Path) -> Path:
    classifiers = sorted(macro_df["classifier_label"].dropna().unique())
    fig, axes = plt.subplots(
        len(classifiers),
        1,
        figsize=(10, 3.2 * len(classifiers)),
        sharex=False,
        constrained_layout=True,
    )

    if len(classifiers) == 1:
        axes = [axes]

    for ax, classifier in zip(axes, classifiers):
        classifier_df = macro_df[macro_df["classifier_label"] == classifier]

        for run_type, group in classifier_df.groupby("run_type"):
            ax.hist(
                group["total_time_sec"].dropna() / 60,
                bins=12,
                alpha=0.55,
                label=run_type,
            )

        ax.set_title(classifier)
        ax.set_xlabel("Per-configuration total time (minutes)")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

    output_path = output_dir / "overall_per_configuration_time_histograms.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_tuning_trial_time_histograms(trial_df: pd.DataFrame, output_dir: Path) -> Path:
    classifiers = sorted(trial_df["classifier_label"].dropna().unique())
    fig, axes = plt.subplots(
        len(classifiers),
        1,
        figsize=(10, 3.2 * len(classifiers)),
        sharex=False,
        constrained_layout=True,
    )

    if len(classifiers) == 1:
        axes = [axes]

    for ax, classifier in zip(axes, classifiers):
        classifier_df = trial_df[trial_df["classifier_label"] == classifier]
        ax.hist(classifier_df["total_duration_sec"].dropna() / 60, bins=12, alpha=0.75)
        ax.set_title(classifier)
        ax.set_xlabel("Tuning trial total time (minutes)")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)

    output_path = output_dir / "overall_tuning_trial_time_histograms.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_summary_tables(
    macro_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    trial_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    macro_columns = [
        "dataset",
        "classifier",
        "classifier_label",
        "run_type",
        "window_size",
        "window_percentage",
        "stride_ratio",
        "series_macro_f1",
        "fit_time_sec",
        "predict_time_sec",
        "total_time_sec",
    ]
    macro_out = macro_df[macro_columns].sort_values(
        ["classifier_label", "stride_ratio", "run_type", "window_percentage"]
    )

    best_macro = (
        macro_out.sort_values("series_macro_f1", ascending=False)
        .groupby(["classifier_label", "run_type", "stride_ratio"], as_index=False)
        .head(1)
        .sort_values(["classifier_label", "stride_ratio", "run_type"])
    )

    phase_out = phase_df.sort_values(["classifier_label", "phase"])
    trial_summary = (
        trial_df.groupby("classifier_label", as_index=False)
        .agg(
            n_configurations=("trial_file", "count"),
            total_tuning_time_min=("total_duration_sec", lambda values: values.sum() / 60),
            mean_trial_time_min=("total_duration_sec", lambda values: values.mean() / 60),
            median_trial_time_min=("total_duration_sec", lambda values: values.median() / 60),
        )
        .sort_values("classifier_label")
    )

    paths = [
        output_dir / "overall_macro_f1_all_points.csv",
        output_dir / "overall_macro_f1_best_by_stride.csv",
        output_dir / "overall_phase_total_times.csv",
        output_dir / "overall_tuning_trial_time_summary.csv",
    ]

    macro_out.to_csv(paths[0], index=False)
    best_macro.to_csv(paths[1], index=False)
    phase_out.to_csv(paths[2], index=False)
    trial_summary.to_csv(paths[3], index=False)

    return paths


def build_overall_plots(results_dir: Path, plots_dir: Path, tables_dir: Path) -> dict[str, list[Path]]:
    comparison_dirs = discover_comparison_dirs(results_dir)

    if not comparison_dirs:
        raise FileNotFoundError(f"No complete comparison directories found in {results_dir}")

    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    macro_df = load_macro_f1(comparison_dirs)
    phase_df = load_phase_times(comparison_dirs)
    trial_df = load_tuning_trial_times(comparison_dirs)

    plot_paths = [
        plot_macro_f1_by_stride(macro_df, plots_dir),
        plot_phase_time_bars(phase_df, plots_dir),
        plot_per_configuration_time_histograms(macro_df, plots_dir),
        plot_tuning_trial_time_histograms(trial_df, plots_dir),
    ]
    table_paths = save_summary_tables(macro_df, phase_df, trial_df, tables_dir)

    selected_dirs_path = tables_dir / "overall_selected_comparison_dirs.csv"
    pd.DataFrame(
        [
            {"classifier_dir": classifier, "comparison_dir": str(comparison_dir)}
            for classifier, comparison_dir in comparison_dirs.items()
        ]
    ).to_csv(selected_dirs_path, index=False)
    table_paths.append(selected_dirs_path)

    return {"plots": plot_paths, "tables": table_paths}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create overall Fish macro-F1 and timing plots from comparison CSVs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/Fish"),
        help="Fish results directory.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("plots/Fish/overall"),
        help="Directory where PNG plots are saved.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("results/Fish/overall"),
        help="Directory where summary CSV tables are saved.",
    )
    args = parser.parse_args()

    outputs = build_overall_plots(args.results_dir, args.plots_dir, args.tables_dir)

    print("Saved plots:")
    for path in outputs["plots"]:
        print(path)

    print("\nSaved tables:")
    for path in outputs["tables"]:
        print(path)


if __name__ == "__main__":
    main()
