from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_all_results(results_dir: str, recursive: bool = True) -> pd.DataFrame:
    """
    Load all CSV result files from a directory.

    Parameters
    ----------
    results_dir : str
        Directory containing experiment CSV files.
    recursive : bool
        If True, also search inside subdirectories.

    Returns
    -------
    pd.DataFrame
        Concatenated results.
    """

    results_path = Path(results_dir)

    if recursive:
        csv_files = sorted(results_path.glob("**/*.csv"))
    else:
        csv_files = sorted(results_path.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")

    dataframes = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df["source_file"] = csv_file.name
        df["source_path"] = str(csv_file)
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


def filter_results(
    df: pd.DataFrame,
    filter_query: str | None = None,
    keep_only_ok: bool = True,
) -> pd.DataFrame:
    """
    Filter experiment results.

    Parameters
    ----------
    df : pd.DataFrame
        Input results.
    filter_query : str | None
        Optional pandas query string.
    keep_only_ok : bool
        If True, keep only rows with status == 'ok'.

    Returns
    -------
    pd.DataFrame
        Filtered results.
    """

    df = df.copy()

    if keep_only_ok and "status" in df.columns:
        df = df[df["status"] == "ok"]

    if filter_query is not None:
        df = df.query(filter_query)

    return df


def aggregate_results(
    df: pd.DataFrame,
    x: str,
    y: str,
    group_by: str | None = None,
    facet_by: str | None = None,
    aggregation: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate results before plotting.

    This is useful when multiple runs share the same:
        dataset, classifier, window_size, stride_ratio, etc.

    Parameters
    ----------
    df : pd.DataFrame
        Input results.
    x : str
        Column for x-axis.
    y : str
        Column for y-axis.
    group_by : str | None
        Column used for grouping lines/bars.
    facet_by : str | None
        Column used to create separate plots.
    aggregation : str
        Aggregation function: mean, median, max, min.

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe.
    """

    required_columns = [x, y]

    if group_by is not None:
        required_columns.append(group_by)

    if facet_by is not None:
        required_columns.append(facet_by)

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing columns in results dataframe: {missing_columns}\n"
            f"Available columns are: {list(df.columns)}"
        )

    df = df.dropna(subset=[x, y]).copy()

    group_columns = []

    if facet_by is not None:
        group_columns.append(facet_by)

    if group_by is not None:
        group_columns.append(group_by)

    group_columns.append(x)

    if aggregation == "mean":
        aggregated = (
            df.groupby(group_columns, dropna=False)[y]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        aggregated = aggregated.rename(columns={"mean": y})

    elif aggregation == "median":
        aggregated = (
            df.groupby(group_columns, dropna=False)[y]
            .agg(["median", "count"])
            .reset_index()
        )
        aggregated = aggregated.rename(columns={"median": y})

    elif aggregation == "max":
        aggregated = (
            df.groupby(group_columns, dropna=False)[y]
            .agg(["max", "count"])
            .reset_index()
        )
        aggregated = aggregated.rename(columns={"max": y})

    elif aggregation == "min":
        aggregated = (
            df.groupby(group_columns, dropna=False)[y]
            .agg(["min", "count"])
            .reset_index()
        )
        aggregated = aggregated.rename(columns={"min": y})

    else:
        raise ValueError(
            f"Unsupported aggregation: {aggregation}. "
            "Use one of: mean, median, max, min."
        )

    return aggregated


def safe_filename(value) -> str:
    """
    Convert a value into a filesystem-safe filename component.
    """

    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")


def sort_for_plot(df: pd.DataFrame, x: str) -> pd.DataFrame:
    """
    Sort dataframe by x-axis column.
    Works for both numeric and categorical x values.
    """

    df = df.copy()

    try:
        df["_x_sort"] = pd.to_numeric(df[x])
        df = df.sort_values("_x_sort")
        df = df.drop(columns=["_x_sort"])
    except Exception:
        df = df.sort_values(x)

    return df


def plot_results(
    df: pd.DataFrame,
    x: str,
    y: str,
    group_by: str | None = None,
    facet_by: str | None = None,
    filter_query: str | None = None,
    plot_type: str = "line",
    aggregation: str = "mean",
    output_dir: str = "plots",
    output_name: str | None = None,
    show: bool = True,
    save: bool = True,
):
    """
    Generic plotting function for experiment results.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment results.
    x : str
        Column for x-axis.
    y : str
        Column for y-axis.
    group_by : str | None
        Column used to group lines/bars.
    facet_by : str | None
        Column used to create separate plots.
    filter_query : str | None
        Pandas query string.
    plot_type : str
        One of: line, bar, scatter.
    aggregation : str
        Aggregation function: mean, median, max, min.
    output_dir : str
        Directory where plots are saved.
    output_name : str | None
        Optional output name.
    show : bool
        If True, show plots interactively.
    save : bool
        If True, save plots as PNG.

    Returns
    -------
    list[Path]
        Paths of saved plot files.
    """

    if plot_type not in {"line", "bar", "scatter"}:
        raise ValueError("plot_type must be one of: line, bar, scatter")

    filtered = filter_results(
        df,
        filter_query=filter_query,
        keep_only_ok=True,
    )

    if len(filtered) == 0:
        raise ValueError("No rows available after filtering.")

    plot_df = aggregate_results(
        filtered,
        x=x,
        y=y,
        group_by=group_by,
        facet_by=facet_by,
        aggregation=aggregation,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    if facet_by is None:
        facet_values = [None]
    else:
        facet_values = sorted(plot_df[facet_by].dropna().unique())

    for facet_value in facet_values:
        if facet_by is None:
            current_df = plot_df.copy()
            title = f"{y} by {x}"
        else:
            current_df = plot_df[plot_df[facet_by] == facet_value].copy()
            title = f"{y} by {x} | {facet_by} = {facet_value}"

        fig, ax = plt.subplots(figsize=(12, 6))

        if plot_type == "line":
            _plot_line(
                ax=ax,
                df=current_df,
                x=x,
                y=y,
                group_by=group_by,
            )

        elif plot_type == "bar":
            _plot_bar(
                ax=ax,
                df=current_df,
                x=x,
                y=y,
                group_by=group_by,
            )

        elif plot_type == "scatter":
            _plot_scatter(
                ax=ax,
                df=current_df,
                x=x,
                y=y,
                group_by=group_by,
            )

        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.3)

        if group_by is not None:
            ax.legend(title=group_by)

        fig.tight_layout()

        if save:
            if output_name is None:
                file_name = f"{plot_type}_{y}_by_{x}"
            else:
                file_name = output_name

            if facet_by is not None:
                file_name = f"{file_name}_{facet_by}_{safe_filename(facet_value)}"

            file_path = output_path / f"{file_name}.png"
            fig.savefig(file_path, dpi=300, bbox_inches="tight")
            saved_files.append(file_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

    return saved_files


def _plot_line(ax, df: pd.DataFrame, x: str, y: str, group_by: str | None):
    if group_by is None:
        df = sort_for_plot(df, x)
        ax.plot(df[x], df[y], marker="o")
    else:
        for group_value, group_df in df.groupby(group_by):
            group_df = sort_for_plot(group_df, x)
            ax.plot(
                group_df[x],
                group_df[y],
                marker="o",
                label=str(group_value),
            )


def _plot_scatter(ax, df: pd.DataFrame, x: str, y: str, group_by: str | None):
    if group_by is None:
        ax.scatter(df[x], df[y])
    else:
        for group_value, group_df in df.groupby(group_by):
            ax.scatter(
                group_df[x],
                group_df[y],
                label=str(group_value),
            )


def _plot_bar(ax, df: pd.DataFrame, x: str, y: str, group_by: str | None):
    df = sort_for_plot(df, x)

    if group_by is None:
        ax.bar(df[x].astype(str), df[y])
        ax.tick_params(axis="x", rotation=45)
        return

    x_values = list(df[x].dropna().unique())
    group_values = list(df[group_by].dropna().unique())

    positions = np.arange(len(x_values))
    width = 0.8 / max(1, len(group_values))

    for i, group_value in enumerate(group_values):
        group_df = df[df[group_by] == group_value]

        heights = []

        for x_value in x_values:
            row = group_df[group_df[x] == x_value]

            if len(row) == 0:
                heights.append(np.nan)
            else:
                heights.append(row[y].iloc[0])

        offset = (i - (len(group_values) - 1) / 2) * width

        ax.bar(
            positions + offset,
            heights,
            width=width,
            label=str(group_value),
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(value) for value in x_values], rotation=45)