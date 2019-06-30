#   encoding: utf8
#   filename: cli.py

import numpy as np
import pandas as pd

from typing import Optional

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from .baseline import BaselineRegressor


class LinearCorrectionRegressor(BaseEstimator, RegressorMixin):
    """Class LinearCorrectionRegressor реализует идею о том, что можно
    построить линейную регрессию между координатами пользовательской и истинной
    разметок. В качестве признаков берутся параметры прямоугольника (нижний
    угол, площадь, длинны сторон).

    Применяется подель сходным с Baseline образом, но сначала корректируется
    пользовательская разметка с помощью обученного регрессора. Из-за того, что
    регрессия линейная корректировку приходиться делить на 100.

    :param n_jobs: Число поток для обучения линейного регрессора.
    """

    def __init__(self, n_jobs: Optional[int] = None):
        super().__init__()

        self.reg = LinearRegression(normalize=True, n_jobs=n_jobs)
        self.bas = BaselineRegressor()

    def fit(self,
            X: pd.DataFrame,
            y: pd.DataFrame) -> 'LinearCorrectionRegressor':
        full = pd.merge(X, y, left_on='item_id', right_on='item_id')
        full.drop(['user_id', 'item_id'], axis=1, inplace=True)
        data = full.values

        correction = data[:, 4:] - data[:, :4]
        features = self._prepare_features(data)

        self.reg.fit(features, correction)
        self.bas.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        features = self._prepare_features(X.drop('item_id', axis=1).values)
        correction = self.reg.predict(features) / 100  # NOTE: Wtf!?
        Y = X.copy()
        Y[['x_min', 'y_min', 'x_max', 'y_max']] += correction
        return self.bas.predict(Y)

    def _prepare_features(self, data: np.ndarray) -> np.ndarray:
        x0 = data[:, 0]
        y0 = data[:, 1]
        dx = data[:, 2] - data[:, 0]
        dy = data[:, 3] - data[:, 1]
        area = dx * dy
        features = np.stack([x0, y0, dx, dy, area]).T
        return features
