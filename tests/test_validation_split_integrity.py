import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from utils.validation import TimeSeriesValidator, cross_validate


class RecordingRegressor(BaseEstimator, RegressorMixin):
    fit_columns_log = []
    fit_kwargs_log = []

    def fit(self, X, y, **kwargs):
        self.__class__.fit_columns_log.append(list(X.columns))
        self.__class__.fit_kwargs_log.append(kwargs)
        self._mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self._mean)


def _config():
    return {"features": {"day_col": "day"}}


def test_cross_validate_uses_split_df_without_day_leakage():
    RecordingRegressor.fit_columns_log = []
    RecordingRegressor.fit_kwargs_log = []

    x = pd.DataFrame({"feat_1": np.arange(12), "feat_2": np.arange(12) * 0.1})
    split_df = pd.DataFrame({"day": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]})
    y = pd.Series(np.linspace(0.0, 1.0, len(x)))

    validator = TimeSeriesValidator(n_splits=2, config=_config())
    results = cross_validate(
        model=RecordingRegressor(),
        X=x,
        y=y,
        validator=validator,
        split_df=split_df,
        fit_kwargs={"categorical_feature": ["pid"]},
        config=_config(),
        verbose=False,
    )

    assert len(results["fold_metrics"]) == 2
    assert all("day" not in cols for cols in RecordingRegressor.fit_columns_log)
    assert all(
        kwargs.get("categorical_feature") == ["pid"]
        for kwargs in RecordingRegressor.fit_kwargs_log
    )


def test_cross_validate_raises_on_split_length_mismatch():
    x = pd.DataFrame({"feat_1": np.arange(10)})
    split_df = pd.DataFrame({"day": np.arange(9)})
    y = pd.Series(np.linspace(0.0, 1.0, len(x)))

    validator = TimeSeriesValidator(n_splits=2, config=_config())
    with pytest.raises(ValueError):
        cross_validate(
            model=RecordingRegressor(),
            X=x,
            y=y,
            validator=validator,
            split_df=split_df,
            config=_config(),
            verbose=False,
        )
