# Usage Guide

This document explains how to use the main scripts of the `Thesis_SlidingWindow` project.

The goal of the project is to run reproducible experiments on the effect of sliding window size in time-series classification using `aeon`.

All commands must be executed from the root folder of the repository:

```bash
cd Thesis_SlidingWindow
```

Before running any script, activate the virtual environment:

```bash
source .venv/bin/activate
```

Check that the correct Python version is being used:

```bash
python --version
```

Expected version:

```text
Python 3.12.x
```

---

## 1. Project structure

The expected project structure is:

```text
Thesis_SlidingWindow/
├── data/
│   ├── raw/
│   └── processed/
├── results/
├── plots/
├── notebooks/
├── scripts/
│   ├── run_experiment.py
│   ├── run_many.py
│   ├── plot_results.py
│   └── tune_classifier.py
├── src/
│   └── sliding_window_tsc/
└── tests/
```

Main folders:

| Folder                    | Purpose                           |
| ------------------------- | --------------------------------- |
| `data/raw/`               | Original datasets                 |
| `data/processed/`         | Optional processed datasets       |
| `results/`                | CSV files produced by experiments |
| `plots/`                  | Generated figures                 |
| `scripts/`                | Command-line scripts              |
| `src/sliding_window_tsc/` | Main Python package               |
| `tests/`                  | Unit and smoke tests              |

---

## 2. Dataset format

Datasets must be placed inside `data/raw/`.

Each dataset must have one train file and one test file in `.ts` format.

Example:

```text
data/raw/Fish/
├── Fish_TRAIN.ts
└── Fish_TEST.ts
```

The expected aeon format is:

```text
X shape = (n_cases, n_channels, n_timepoints)
y shape = (n_cases,)
```

Example with the Fish dataset:

```text
X_train shape = (175, 1, 463)
y_train shape = (175,)
```

---

## 3. Run a single experiment

The main script for running one experiment is:

```bash
python scripts/run_experiment.py
```

Example:

```bash
python scripts/run_experiment.py \
  --dataset-folder data/raw/Fish \
  --classifier SummaryClassifier \
  --window-sizes 0.10 \
  --percentages \
  --stride-ratio 0.5 \
  --output-dir results/smoke
```

This command:

* loads the `Fish` dataset;
* uses `SummaryClassifier`;
* uses a sliding window equal to 10% of the original time-series length;
* uses a stride equal to 50% of the window size;
* saves the result in `results/smoke`.

The output file has a name similar to:

```text
Fish_SummaryClassifier_YYYYMMDD_HHMMSS.csv
```

---

## 4. Arguments of `run_experiment.py`

### `--dataset-folder`

Path to the dataset folder.

Example:

```bash
--dataset-folder data/raw/Fish
```

The folder must contain:

```text
Fish_TRAIN.ts
Fish_TEST.ts
```

---

### `--classifier`

Name of the aeon classifier.

Example:

```bash
--classifier MiniRocketClassifier
```

Useful classifiers for this project:

```text
SummaryClassifier
Catch22Classifier
MiniRocketClassifier
TimeSeriesForestClassifier
RandomIntervalClassifier
KNeighborsTimeSeriesClassifier
DrCIFClassifier
RDSTClassifier
WEASEL
InceptionTimeClassifier
```

---

### `--window-sizes`

List of window sizes.

Absolute window sizes:

```bash
--window-sizes 50 100 150
```

Percentage window sizes:

```bash
--window-sizes 0.05 0.10 0.20
--percentages
```

If `--percentages` is used, window sizes are interpreted as fractions of the original series length.

Example:

```text
series length = 463
window size = 0.10
actual window size = 46
```

---

### `--percentages`

Interprets `--window-sizes` as percentages of the original time-series length.

Example:

```bash
--window-sizes 0.10
--percentages
```

Without `--percentages`, `0.10` would be interpreted as an absolute window size and would not be valid.

---

### `--stride-ratio`

Stride as a fraction of the window size.

Example:

```bash
--stride-ratio 0.5
```

If:

```text
window_size = 46
stride_ratio = 0.5
```

then:

```text
stride = 23
```

Common values:

| `stride_ratio` | Meaning     |
| -------------: | ----------- |
|          `1.0` | No overlap  |
|          `0.5` | 50% overlap |
|         `0.25` | 75% overlap |

---

### `--output-dir`

Directory where the result CSV file is saved.

Example:

```bash
--output-dir results/fish
```

---

### `--random-state`

Random seed.

Example:

```bash
--random-state 42
```

---

### `--hyperparameters`

Optional classifier hyperparameters as a JSON string.

Example:

```bash
--hyperparameters '{"num_kernels": 1000}'
```

Full example:

```bash
python scripts/run_experiment.py \
  --dataset-folder data/raw/Fish \
  --classifier MiniRocketClassifier \
  --window-sizes 0.10 0.20 \
  --percentages \
  --stride-ratio 0.5 \
  --hyperparameters '{"num_kernels": 1000}' \
  --output-dir results/fish
```

Another example with KNN:

```bash
python scripts/run_experiment.py \
  --dataset-folder data/raw/Fish \
  --classifier KNeighborsTimeSeriesClassifier \
  --window-sizes 0.10 0.20 \
  --percentages \
  --stride-ratio 0.5 \
  --hyperparameters '{"n_neighbors": 3, "distance": "dtw"}' \
  --output-dir results/fish
```

---

## 5. Output CSV columns

Each experiment produces one row for each tested window size.

Important columns:

```text
dataset
classifier
classifier_hyperparameters
window_size
window_percentage
stride
stride_ratio
n_train_windows
n_test_windows
window_accuracy
window_balanced_accuracy
window_macro_f1
series_accuracy
series_balanced_accuracy
series_macro_f1
fit_time_sec
predict_time_sec
total_time_sec
random_state
status
error
```

The most important metric for the thesis is usually:

```text
series_macro_f1
```

because the original task is classification of full time series, not classification of individual windows.

A successful experiment has:

```text
status = ok
```

A failed experiment has:

```text
status = failed
```

and the error message is stored in:

```text
error
```

---

## 6. Run multiple window sizes

Example:

```bash
python scripts/run_experiment.py \
  --dataset-folder data/raw/Fish \
  --classifier SummaryClassifier \
  --window-sizes 0.05 0.10 0.20 0.30 0.50 \
  --percentages \
  --stride-ratio 0.5 \
  --output-dir results/fish
```

This tests five different window sizes.

---

## 7. Run multiple classifiers manually

Example:

```bash
python scripts/run_experiment.py \
  --dataset-folder data/raw/Fish \
  --classifier SummaryClassifier \
  --window-sizes 0.05 0.10 0.20 \
  --percentages \
  --stride-ratio 0.5 \
  --output-dir results/fish

python scripts/run_experiment.py \
  --dataset-folder data/raw/Fish \
  --classifier Catch22Classifier \
  --window-sizes 0.05 0.10 0.20 \
  --percentages \
  --stride-ratio 0.5 \
  --output-dir results/fish

python scripts/run_experiment.py \
  --dataset-folder data/raw/Fish \
  --classifier MiniRocketClassifier \
  --window-sizes 0.05 0.10 0.20 \
  --percentages \
  --stride-ratio 0.5 \
  --hyperparameters '{"num_kernels": 1000}' \
  --output-dir results/fish
```

---

## 8. Suggested classifier sets

### Fast preliminary set

This set is useful for quick exploration.

```python
FAST_TRAINING_CLASSIFIERS = [
    "MiniRocketClassifier",
    "SummaryClassifier",
    "Catch22Classifier",
    "TimeSeriesForestClassifier",
    "RandomIntervalClassifier",
    "KNeighborsTimeSeriesClassifier",
]
```

Note: `KNeighborsTimeSeriesClassifier` has almost no training time, but prediction can be slow. For this reason, always consider:

```text
fit_time_sec
predict_time_sec
total_time_sec
```

---

### Scientific diversity set

This set is useful for the main experimental analysis.

```python
IDEAL_CLASSIFIERS = [
    "MiniRocketClassifier",
    "KNeighborsTimeSeriesClassifier",
    "WEASEL",
    "Catch22Classifier",
    "DrCIFClassifier",
    "RDSTClassifier",
    "InceptionTimeClassifier",
]
```

This set covers different methodological families:

| Family            | Classifier                       |
| ----------------- | -------------------------------- |
| Convolution-based | `MiniRocketClassifier`           |
| Distance-based    | `KNeighborsTimeSeriesClassifier` |
| Dictionary-based  | `WEASEL`                         |
| Feature-based     | `Catch22Classifier`              |
| Interval-based    | `DrCIFClassifier`                |
| Shapelet-based    | `RDSTClassifier`                 |
| Deep learning     | `InceptionTimeClassifier`        |

---

## 9. Run Optuna hyperparameter tuning

The main script for hyperparameter tuning is:

```bash
python scripts/tune_classifier.py
```

Example smoke test:

```bash
python scripts/tune_classifier.py \
  --dataset-folder data/raw/Fish \
  --classifier MiniRocketClassifier \
  --window-sizes 0.10 \
  --percentages \
  --stride-ratio 0.5 \
  --metric series_macro_f1 \
  --n-trials 2 \
  --output-dir results/tuning_smoke
```

This runs a very small Optuna search with 2 trials.

For real experiments, use more trials:

```bash
--n-trials 20
```

or:

```bash
--n-trials 50
```

The tuning output is saved inside the selected output directory.

Typical files:

```text
Fish_MiniRocketClassifier_optuna_trials.csv
Fish_MiniRocketClassifier_best_params.json
```

---

## 10. Plot results

The main plotting script is:

```bash
python scripts/plot_results.py
```

Basic example:

```bash
python scripts/plot_results.py \
  --results-dir results \
  --x window_size \
  --y series_macro_f1 \
  --plot-type line \
  --output-dir plots/performance
```

This plots `series_macro_f1` as a function of `window_size`.

---

## 11. Plot with one line per classifier

```bash
python scripts/plot_results.py \
  --results-dir results \
  --x window_size \
  --y series_macro_f1 \
  --group-by classifier \
  --plot-type line \
  --output-dir plots/performance
```

---

## 12. Plot with one line per stride ratio

```bash
python scripts/plot_results.py \
  --results-dir results \
  --x window_size \
  --y series_macro_f1 \
  --group-by stride_ratio \
  --plot-type line \
  --output-dir plots/performance
```

---

## 13. Plot separated by dataset

```bash
python scripts/plot_results.py \
  --results-dir results \
  --x window_size \
  --y series_macro_f1 \
  --group-by classifier \
  --facet-by dataset \
  --plot-type line \
  --output-dir plots/performance
```

This creates one plot for each dataset.

---

## 14. Plot only one dataset

Use the `--filter` argument.

Example:

```bash
python scripts/plot_results.py \
  --results-dir results \
  --x window_size \
  --y series_macro_f1 \
  --group-by classifier \
  --filter "dataset == 'Fish'" \
  --plot-type line \
  --output-dir plots/fish
```

---

## 15. Plot only one classifier

```bash
python scripts/plot_results.py \
  --results-dir results \
  --x window_size \
  --y series_macro_f1 \
  --group-by stride_ratio \
  --filter "classifier == 'SummaryClassifier'" \
  --plot-type line \
  --output-dir plots/summary_classifier
```

---

## 16. Plot training time

```bash
python scripts/plot_results.py \
  --results-dir results \
  --x window_size \
  --y fit_time_sec \
  --group-by classifier \
  --facet-by dataset \
  --plot-type line \
  --output-dir plots/training_time
```

---

## 17. Plot total time

```bash
python scripts/plot_results.py \
  --results-dir results \
  --x window_size \
  --y total_time_sec \
  --group-by classifier \
  --facet-by dataset \
  --plot-type line \
  --output-dir plots/total_time
```

This is important because some classifiers have fast training but slow prediction.

---

## 18. Bar plot: best score per classifier

```bash
python scripts/plot_results.py \
  --results-dir results \
  --x classifier \
  --y series_macro_f1 \
  --facet-by dataset \
  --plot-type bar \
  --aggregation max \
  --output-dir plots/best_scores
```

Here:

```bash
--aggregation max
```

means that for each classifier the plot uses the best observed `series_macro_f1`.

---

## 19. Recommended plots for the thesis

### Performance vs window size

```bash
python scripts/plot_results.py \
  --results-dir results \
  --x window_size \
  --y series_macro_f1 \
  --group-by classifier \
  --facet-by dataset \
  --plot-type line \
  --output-dir plots/performance
```

---

### Training time vs window size

```bash
python scripts/plot_results.py \
  --results-dir results \
  --x window_size \
  --y fit_time_sec \
  --group-by classifier \
  --facet-by dataset \
  --plot-type line \
  --output-dir plots/training_time
```

---

### Total time vs window size

```bash
python scripts/plot_results.py \
  --results-dir results \
  --x window_size \
  --y total_time_sec \
  --group-by classifier \
  --facet-by dataset \
  --plot-type line \
  --output-dir plots/total_time
```

---

### Best score per classifier

```bash
python scripts/plot_results.py \
  --results-dir results \
  --x classifier \
  --y series_macro_f1 \
  --facet-by dataset \
  --plot-type bar \
  --aggregation max \
  --output-dir plots/best_scores
```

---

## 20. Run tests

Run all tests:

```bash
pytest -v
```

Run only windowing tests:

```bash
pytest tests/test_windowing.py -v
```

Run only metrics tests:

```bash
pytest tests/test_metrics.py -v
```

Run only classifier tests:

```bash
pytest tests/test_classifiers.py -v
```

Run Fish smoke test:

```bash
pytest tests/test_fish_smoke.py -v
```

If the Fish dataset is not available in `data/raw/Fish`, the smoke test should be skipped.

---

## 21. Useful help commands

Each script should provide a command-line help message.

```bash
python scripts/run_experiment.py --help
python scripts/tune_classifier.py --help
python scripts/plot_results.py --help
```

Use these commands to inspect all available arguments.

---

## 22. Minimal smoke test workflow

This is the minimal workflow to verify that the framework works.

Run one experiment:

```bash
python scripts/run_experiment.py \
  --dataset-folder data/raw/Fish \
  --classifier SummaryClassifier \
  --window-sizes 0.10 \
  --percentages \
  --stride-ratio 0.5 \
  --output-dir results/smoke
```

Then plot:

```bash
python scripts/plot_results.py \
  --results-dir results/smoke \
  --x window_size \
  --y series_macro_f1 \
  --group-by classifier \
  --plot-type line \
  --output-dir plots/smoke
```

Check generated files:

```bash
ls results/smoke
ls plots/smoke
```

The experiment is successful only if the CSV contains:

```text
status = ok
```

and valid values for:

```text
series_macro_f1
fit_time_sec
predict_time_sec
total_time_sec
```

---

## 23. Troubleshooting

### `No such file or directory: scripts/run_experiment.py`

The file does not exist or the command is being executed from the wrong folder.

Check current folder:

```bash
pwd
```

You should be inside:

```text
Thesis_SlidingWindow
```

Check that the script exists:

```bash
ls scripts
```

---

### `ModuleNotFoundError: No module named 'sliding_window_tsc'`

The project is probably not installed in editable mode.

Run:

```bash
pip install -e ".[dev]"
```

Then retry the command.

---

### `Package requires a different Python`

Check Python version:

```bash
python --version
```

If it is Python 3.13 or Python 3.14, recreate the virtual environment with Python 3.12:

```bash
deactivate
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

### `too many values to unpack (expected 2)`

This usually means a function returns more values than the caller expects.

In this project, `make_window_dataset` should return three values:

```python
X_windows, y_windows, series_ids
```

Therefore, in `experiment.py`, calls should look like:

```python
X_train_w, y_train_w, train_series_ids = make_window_dataset(...)
X_test_w, y_test_w, test_series_ids = make_window_dataset(...)
```

not:

```python
X_train_w, y_train_w = make_window_dataset(...)
```

---

## 24. Files that should not be committed

Do not commit:

```text
.venv/
data/raw/*
data/processed/*
results/*
plots/*
*.egg-info/
__pycache__/
.DS_Store
```

The repository should contain code, documentation, configuration files, and lightweight tests, not local environments, datasets, or generated results.
