#   encoding: utf8
#   filename: util.py

import numpy as np
import scipy as sp
import scipy.sparse

from dataclasses import dataclass
from os.path import join
from typing import Tuple

from scipy.sparse import spmatrix


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
