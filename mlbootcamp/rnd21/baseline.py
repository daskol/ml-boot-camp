#   encoding: utf8
#   filename: baseline.py

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin


class BaselineRegressor(BaseEstimator, RegressorMixin):

    def __init__(self):
        super().__init__()

    def fit(self, X=None, y=None) -> 'BaselineRegressor':
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        # In general case prediction is a means of all avaliable markups.
        Y = X.copy() \
            .set_index(['item_id', 'user_id']) \
            .groupby(level=0) \
            .mean() \
            .reset_index()

        if 146 not in Y.item_id.values:
            return Y

        # In case multiobject markup we generate random markup.
        Z = X[X.item_id == 146]

        min_x = Z.x_min.min()
        min_y = Z.y_min.min()
        max_x = Z.x_max.max()
        max_y = Z.y_max.max()

        coords = np.zeros((2, 2))
        coords[:, 0] = np.random.randint(min_x, max_x, 2)
        coords[:, 1] = np.random.randint(min_y, max_y, 2)
        coords.sort(axis=0)

        Y.loc[Y.item_id == 146, ['x_min', 'y_min', 'x_max', 'y_max']] = \
            coords.reshape(-1)

        return Y
