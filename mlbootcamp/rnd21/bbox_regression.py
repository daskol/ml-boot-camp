#   encoding: utf8
#   filename: bbox_regression.py

import logging
import numpy as np
import pandas as pd
import torch as T
import torch.utils.data

from typing import Optional

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


class NNRegression(T.nn.Module):

    def __init__(self, noepoches: int = 512, nosamples: int = 256):
        super().__init__()

        if T.cuda.device_count():
            self.device = T.device('gpu')
        else:
            self.device = T.device('cpu')

        self.noepoches = noepoches
        self.nosamples = nosamples
        self.loss = T.nn.MSELoss(reduction='mean')
        self.model = T.nn.Sequential(
            T.nn.Linear(4, 4),
            T.nn.ELU(),
        ).to(self.device)

    def forward(self, inputs):
        return self.model(inputs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Initial approximation.
        reg = LinearRegression(copy_X=True).fit(X, y)
        self.model[0].weight.data = T.tensor(reg.coef_, dtype=T.float)
        self.model[0].bias.data = T.tensor(reg.intercept_, dtype=T.float)

        # Use SGD in order to fit neural network.
        opt = T.optim.Adam(self.parameters(), lr=1e-4)

        tensor_X = T.tensor(X, dtype=T.float).to(self.device)
        tensor_y = T.tensor(y, dtype=T.float).to(self.device)
        dataset = T.utils.data.TensorDataset(tensor_X, tensor_y)
        loader = T.utils.data.DataLoader(dataset,
                                         batch_size=self.nosamples,
                                         shuffle=True)

        for e in range(self.noepoches):
            for b, (features, targets) in enumerate(loader):
                opt.zero_grad()
                predictions = self(features)
                criterion = self.loss(predictions, targets)
                criterion.backward()
                opt.step()

            if e % 50 == 0:
                logging.debug('%03d:%03d mse = %11.4f', e, b, criterion.item())

        logging.debug('%03d:%03d mse = %11.4f', e, b, criterion.item())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self(T.tensor(X, dtype=T.float)).detach().numpy()


class BoundingBoxRegressor(BaseEstimator, RegressorMixin):
    """Class BoundingBoxRegressor решает задачу регрессия на пользовательских
    разметках. Целевыми переменными являются координаты истинной разметки, а
    признаками координаты пользовательских разметок. При это пользовательских
    разметки могут быть усреднены до или после построения регрессии, за что
    отвечает параметр :avg_mode:.

    :param avg_mode: Когда усреднять пользовательские разметки.

    :param denoising: Apply heuristic algorithm in order to remove outliers.

    :param denoising_level: Threshold value for side of bounding box.

    :param extend: Aggregate markup to increase area or just calculate means by
                   coordinates.

    :param regressor: Use common Linear Regression or Neural Network with ELU.

    :param n_jobs: Число поток для обучения линейного регрессора.
    """

    def __init__(self, avg_mode: str = 'before',
                 denoising: bool = True,
                 denoising_level: int = 10,
                 extend: bool = True,
                 regressor: str = 'sklearn',
                 n_jobs: Optional[int] = None):
        super().__init__()

        if avg_mode == 'before':
            self.avg_mode = avg_mode
        elif avg_mode == 'after':
            self.avg_mode = avg_mode
        else:
            raise ValueError('Averaging of bounding box coordinates must be '
                             'done either before or after regression.')

        if regressor == 'sklearn':
            self.reg = LinearRegression(normalize=True, n_jobs=n_jobs)
        elif regressor == 'pytorch':
            self.reg = NNRegression()
        else:
            raise ValueError(f'Unknown type of regression model: {regressor}.')

        self.denoising = denoising
        self.denoising_level = denoising_level
        self.aggfuncs = {}

        if extend:
            self.aggfuncs['x_min'] = np.min
            self.aggfuncs['y_min'] = np.min
            self.aggfuncs['x_max'] = np.max
            self.aggfuncs['y_max'] = np.max
        else:
            self.aggfuncs['x_min'] = np.mean
            self.aggfuncs['y_min'] = np.mean
            self.aggfuncs['x_max'] = np.mean
            self.aggfuncs['y_max'] = np.mean

    def fit(self,
            X: pd.DataFrame,
            y: pd.DataFrame) -> 'BoundingBoxRegressor':
        if self.denoising:
            X, y = self._filter_outliers(X, y)

        if self.avg_mode == 'after':
            joined = pd.merge(X, y, left_on='item_id', right_on='item_id')
            features = joined[joined.columns[2:6]].values
            targets = joined[joined.columns[6:]].values
        elif self.avg_mode == 'before':
            Y = X.copy() \
                .drop('user_id', axis=1) \
                .groupby('item_id') \
                .aggregate(self.aggfuncs)
            features = Y.values
            targets = y.set_index('item_id').values

        self.reg.fit(features, targets)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.denoising:
            X = self._filter_outliers(X)

        if self.avg_mode == 'after':
            Y = X.copy() \
                .drop('user_id', axis=1) \
                .set_index('item_id')
        elif self.avg_mode == 'before':
            Y = X.copy() \
                .drop('user_id', axis=1) \
                .groupby('item_id') \
                .aggregate(self.aggfuncs)

        Y[['x_min', 'y_min', 'x_max', 'y_max']] = self.reg.predict(Y.values)

        if self.avg_mode == 'after':
            return Y \
                .reset_index('item_id') \
                .groupby('item_id') \
                .aggregate(self.aggfuncs) \
                .reset_index()
        elif self.avg_mode == 'before':
            return Y.reset_index()

    def _filter_outliers(self, X: pd.DataFrame,
                         y: Optional[pd.DataFrame] = None):
        X = X.drop(X[(X.x_max - X.x_min) < self.denoising_level].index)
        X = X.drop(X[(X.y_max - X.y_min) < self.denoising_level].index)
        if y is None:
            return X
        y = y.set_index('item_id').loc[X.item_id.unique()].reset_index()
        return X, y
