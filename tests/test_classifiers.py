import pytest

from sliding_window_tsc.classifiers import get_classifier_by_name


def test_get_classifier_by_name_valid_classifier():
    clf = get_classifier_by_name("MiniRocketClassifier")

    assert clf is not None
    assert clf.__class__.__name__ == "MiniRocketClassifier"


def test_get_classifier_by_name_invalid_classifier():
    with pytest.raises(ValueError):
        get_classifier_by_name("ClassifierCheNonEsiste")
