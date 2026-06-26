import argparse
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "thesis_sliding_window_matplotlib"),
)
os.environ.setdefault("MPLBACKEND", "Agg")

from sliding_window_tsc.plotting import load_all_results, plot_results


def main():
    parser = argparse.ArgumentParser(
        description="Plot experimental results with flexible grouping."
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing experiment CSV files.",
    )

    parser.add_argument(
        "--x",
        type=str,
        required=True,
        help="Column to use on the x-axis.",
    )

    parser.add_argument(
        "--y",
        type=str,
        required=True,
        help="Column to use on the y-axis.",
    )

    parser.add_argument(
        "--group-by",
        type=str,
        default=None,
        help="Column used to group lines/bars/points.",
    )

    parser.add_argument(
        "--facet-by",
        type=str,
        default=None,
        help="Column used to create separate plots.",
    )

    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Pandas query string used to filter rows.",
    )

    parser.add_argument(
        "--plot-type",
        type=str,
        default="line",
        choices=["line", "bar", "scatter"],
        help="Type of plot to create.",
    )

    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "median", "max", "min"],
        help="Aggregation method when multiple rows share the same x/group/facet.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory where plots will be saved.",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional base filename for the saved plot.",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot window.",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the plot.",
    )

    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Search only CSV files directly inside results-dir.",
    )

    args = parser.parse_args()

    df = load_all_results(
        results_dir=args.results_dir,
        recursive=not args.non_recursive,
    )

    saved_files = plot_results(
        df=df,
        x=args.x,
        y=args.y,
        group_by=args.group_by,
        facet_by=args.facet_by,
        filter_query=args.filter,
        plot_type=args.plot_type,
        aggregation=args.aggregation,
        output_dir=args.output_dir,
        output_name=args.output_name,
        show=not args.no_show,
        save=not args.no_save,
    )

    if saved_files:
        print("\nSaved plots:")
        for file in saved_files:
            print(file)


if __name__ == "__main__":
    main()
