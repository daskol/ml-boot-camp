#   encoding: utf8
#   filename: util.py

import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse

from dataclasses import dataclass
from os.path import join
from typing import Tuple

from scipy.sparse import spmatrix
from sklearn.model_selection import KFold


def iou(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Function iou estimates value of Intersection-over-Union (IoU) metric.

    :param lhs: Coordinates of the first bounding box.

    :param rhs: Coordinates of the second bounding box.

    :return: Value of IoU.
    """
    lhs_area = (lhs[2] - lhs[0]) * (lhs[3] - lhs[1])
    rhs_area = (rhs[2] - rhs[0]) * (rhs[3] - rhs[1])

    if lhs_area == 0 or rhs_area == 0:
        return 0.0

    x_min = max(lhs[0], rhs[0])
    y_min = max(lhs[1], rhs[1])
    x_max = min(lhs[2], rhs[2])
    y_max = min(lhs[3], rhs[3])

    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
    iou = intersection / (lhs_area + rhs_area - intersection)
    return iou


def miou(lhs: pd.DataFrame, rhs: pd.DataFrame) -> float:
    """Function miou estimates mean value of IoU.

    :param lhs: Coordinates of bounding boxes of the first frame.

    :param rhs: Coordinates of bounding boxes of the second frame.

    :return: Value of mIoU.
    """
    joined: pd.DataFrame = pd.merge(lhs, rhs, on='item_id')
    joined.drop('item_id', axis=1, inplace=True)
    ious = np.empty(len(joined), dtype=np.float)

    for i, row in joined.iterrows():
        ious[i] = iou(row.values[:4], row.values[4:])

    return ious.mean()


def evaluate(X: pd.DataFrame, y: pd.DataFrame, estimator, params=None,
             nofolds: int = 5, seed: int = 42) -> Tuple[float, float]:
    """Function evaluate does k-fold cross-validation.
    """
    if params is None:
        params = {}

    X = X.set_index('item_id')
    y = y.set_index('item_id')

    mious = np.empty(nofolds, dtype=np.float)
    kfold = KFold(nofolds, shuffle=True, random_state=seed)
    kfold.get_n_splits(y=y)

    for i, (train, test) in enumerate(kfold.split(y.index.values)):
        train_X = X.loc[y.index[train]].reset_index()
        train_y = y.loc[y.index[train]].reset_index()

        test_X = X.loc[y.index[test]].reset_index()
        test_y = y.loc[y.index[test]].reset_index()

        pred_y = estimator(**params) \
            .fit(train_X, train_y) \
            .predict(test_X)

        if pred_y.item_id.nunique() != test_y.item_id.nunique():
            raise RuntimeError('Wrong number of predicted markups.')

        mious[i] = miou(test_y, pred_y)

    return mious.mean(), mious.std()


@dataclass
class Dataset:

    # Minimal coordinates (x_min, y_min).
    cmin: Tuple[spmatrix, spmatrix]

    # Maximal coordinates (x_min, y_min).
    cmax: Tuple[spmatrix, spmatrix]

    # Estimated IoU values for each known box.
    ioum: spmatrix

    # Sorted list of item identifiers.
    iids: np.ndarray

    # Sorted list of user identifiers.
    uids: np.ndarray

    ind_data: np.ndarray
    ind_subm: np.ndarray

    @staticmethod
    def load(indir: str) -> 'Dataset':
        def load_npz(filename):
            return sp.sparse.load_npz(join(indir, filename))

        def load(filename):
            return np.load(join(indir, filename))

        return Dataset(cmin=(load_npz('xmin.npz'), load_npz('ymin.npz')),
                       cmax=(load_npz('xmax.npz'), load_npz('ymax.npz')),
                       ioum=load_npz('ioum.npz'),
                       iids=load('item-ids.npy'),
                       uids=load('user-ids.npy'),
                       ind_data=load('indices-dataset.npy'),
                       ind_subm=load('indices-submset.npy'))
