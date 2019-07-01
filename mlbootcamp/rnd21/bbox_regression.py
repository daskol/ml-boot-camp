#   encoding: utf8
#   filename: bbox_regression.py

import numpy as np
import pandas as pd

from typing import Optional

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from .baseline import BaselineRegressor


class BoundingBoxRegressor(BaseEstimator, RegressorMixin):
    """Class BoundingBoxRegressor решает задачу регрессия на пользовательских
    разметках. Целевыми переменными являются координаты истинной разметки, а
    признаками координаты пользовательских разметок. При это пользовательских
    разметки могут быть усреднены до или после построения регрессии, за что
    отвечает параметр :avg_mode:.

    :param avg_mode: Когда усреднять пользовательские разметки.

    :param n_jobs: Число поток для обучения линейного регрессора.
    """

    def __init__(self, avg_mode: str = 'before', n_jobs: Optional[int] = None):
        super().__init__()

        if avg_mode == 'before':
            self.avg_mode = avg_mode
        elif avg_mode == 'after':
            self.avg_mode = avg_mode
        else:
            raise ValueError('Averaging of bounding box coordinates must be '
                             'done either before or after regression.')

        self.reg = LinearRegression(normalize=True, n_jobs=n_jobs)

    def fit(self,
            X: pd.DataFrame,
            y: pd.DataFrame) -> 'BoundingBoxRegressor':
        if self.avg_mode == 'after':
            joined = pd.merge(X, y, left_on='item_id', right_on='item_id')
            features = joined[joined.columns[2:6]].values
            targets = joined[joined.columns[6:]].values
            self.reg.fit(features, targets)
        elif self.avg_mode == 'before':
            Y = X.copy() \
                .drop('user_id', axis=1) \
                .groupby('item_id') \
                .mean()
            self.reg.fit(Y.values, y.set_index('item_id').values)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.avg_mode == 'after':
            Y = X.copy() \
                .drop('user_id', axis=1) \
                .set_index('item_id')
        elif self.avg_mode == 'before':
            Y = X.copy() \
                .drop('user_id', axis=1) \
                .groupby('item_id') \
                .mean()

        Y[['x_min', 'y_min', 'x_max', 'y_max']] = self.reg.predict(Y.values)

        if self.avg_mode == 'after':
            return Y \
                .reset_index('item_id') \
                .groupby('item_id') \
                .mean() \
                .reset_index()
        elif self.avg_mode == 'before':
            return Y.reset_index()
