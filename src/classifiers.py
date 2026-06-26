from aeon.utils.discovery import all_estimators


def get_classifier_by_name(classifier_name: str, random_state: int = 42):
    """
    Instantiate an aeon classifier by name.

    Example:
        MiniRocketClassifier
        RocketClassifier
        DrCIFClassifier
        TimeSeriesForestClassifier
    """

    classifiers = dict(all_estimators(type_filter="classifier", return_names=True))

    if classifier_name not in classifiers:
        available = "\n".join(sorted(classifiers.keys()))
        raise ValueError(
            f"Unknown classifier: {classifier_name}\n\n"
            f"Available classifiers are:\n{available}"
        )

    classifier_cls = classifiers[classifier_name]

    try:
        return classifier_cls(random_state=random_state)
    except TypeError:
        return classifier_cls()