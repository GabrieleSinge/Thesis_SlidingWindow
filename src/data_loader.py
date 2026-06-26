from pathlib import Path
from aeon.datasets import load_from_ts_file


def find_train_test_files(dataset_folder: str):
    """
    Find TRAIN and TEST .ts files inside a dataset folder.
    Expected format:
        DatasetName_TRAIN.ts
        DatasetName_TEST.ts
    """

    dataset_path = Path(dataset_folder)

    train_files = list(dataset_path.glob("*_TRAIN.ts"))
    test_files = list(dataset_path.glob("*_TEST.ts"))

    if len(train_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *_TRAIN.ts file in {dataset_folder}, found {len(train_files)}."
        )

    if len(test_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *_TEST.ts file in {dataset_folder}, found {len(test_files)}."
        )

    return train_files[0], test_files[0]


def load_dataset(dataset_folder: str):
    """
    Load train and test data from a dataset folder.
    """

    train_file, test_file = find_train_test_files(dataset_folder)

    X_train, y_train = load_from_ts_file(str(train_file))
    X_test, y_test = load_from_ts_file(str(test_file))

    dataset_name = Path(dataset_folder).name

    return X_train, y_train, X_test, y_test, dataset_name