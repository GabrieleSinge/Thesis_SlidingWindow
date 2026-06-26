import inspect

from aeon.utils.discovery import all_estimators


def get_classifier_by_name(
    classifier_name: str,
    random_state: int = 42,
    hyperparameters: dict | None = None,
):
    """Instantiate an aeon classifier by name with optional hyperparameters."""
    classifiers = dict(all_estimators(type_filter="classifier", return_names=True))

    if classifier_name not in classifiers:
        available = ", ".join(sorted(classifiers.keys()))
        raise ValueError(
            f"Unknown classifier: {classifier_name}. "
            f"Available classifiers are: {available}"
        )

    classifier_cls = classifiers[classifier_name]
    params = dict(hyperparameters or {})

    signature = inspect.signature(classifier_cls)
    if "random_state" in signature.parameters and "random_state" not in params:
        params["random_state"] = random_state

    return classifier_cls(**params)
