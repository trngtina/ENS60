import numpy as np
import pandas as pd
import pytest

from utils.data_loader import merge_target, create_submission


def _config():
    return {"features": {"target_col": "target"}, "data": {}}


def test_merge_target_id_alignment_success():
    x_train = pd.DataFrame(
        {
            "ID": [11, 12, 13],
            "pid": [1, 1, 2],
            "NLV": [0.1, 0.2, 0.3],
        }
    )
    y_train = pd.DataFrame(
        {
            "ID": [13, 11, 12],
            "target": [1.3, 1.1, 1.2],
        }
    )

    merged = merge_target(x_train, y_train, config=_config(), align_mode="id", strict=True)

    assert merged["ID"].tolist() == [11, 12, 13]
    assert merged["target"].tolist() == [1.1, 1.2, 1.3]


def test_merge_target_id_alignment_strict_raises_on_missing_ids():
    x_train = pd.DataFrame({"ID": [1, 2, 3], "NLV": [0.1, 0.2, 0.3]})
    y_train = pd.DataFrame({"ID": [1, 2, 4], "target": [1.0, 2.0, 4.0]})

    with pytest.raises(ValueError):
        merge_target(x_train, y_train, config=_config(), align_mode="id", strict=True)


def test_merge_target_positional_fallback_without_id():
    x_train = pd.DataFrame({"pid": [1, 1, 2], "NLV": [0.1, 0.2, 0.3]})
    y_train = pd.DataFrame({"target": [1.0, 2.0, 3.0]})

    merged = merge_target(x_train, y_train, config=_config(), align_mode="auto", strict=True)
    assert merged["target"].tolist() == [1.0, 2.0, 3.0]


def test_create_submission_uses_test_ids():
    x_test = pd.DataFrame({"ID": [100, 101, 102], "pid": [1, 1, 2]})
    preds = np.array([0.01, 0.02, 0.03])

    submission = create_submission(
        predictions=preds,
        X_test=x_test,
        config=_config(),
        validate_against_example=False,
    )

    assert submission["ID"].tolist() == [100, 101, 102]
    assert submission["target"].tolist() == [0.01, 0.02, 0.03]


def test_create_submission_raises_on_length_mismatch():
    x_test = pd.DataFrame({"ID": [100, 101, 102], "pid": [1, 1, 2]})
    preds = np.array([0.01, 0.02])

    with pytest.raises(ValueError):
        create_submission(predictions=preds, X_test=x_test, config=_config())


def test_create_submission_raises_on_duplicate_ids():
    x_test = pd.DataFrame({"ID": [100, 100, 102], "pid": [1, 1, 2]})
    preds = np.array([0.01, 0.02, 0.03])

    with pytest.raises(ValueError):
        create_submission(predictions=preds, X_test=x_test, config=_config())
