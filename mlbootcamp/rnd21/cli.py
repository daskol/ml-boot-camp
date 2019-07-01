#   encoding: utf8
#   filename: cli.py

import click
import logging
import numpy as np
import pandas as pd

from .baseline import BaselineRegressor
from .linear_correction import LinearCorrectionRegressor
from .bbox_regression import BoundingBoxRegressor
from .util import evaluate


@click.group()
def main():
    """Модератор "Одноклассников"
    """


@main.command()
@click.argument('train-data', type=click.Path(exists=True, dir_okay=False))
@click.argument('train-target', type=click.Path(exists=True, dir_okay=False))
@click.argument('test-data', type=click.Path(exists=True, dir_okay=False))
@click.argument('test-target', type=click.Path(exists=False, dir_okay=False))
def baseline(train_data: str, train_target: str,
             test_data: str, test_target: str):
    """Среднее разметок.
    """
    logging.info('load train set and inference set')

    train_X = pd.read_parquet(train_data)
    train_y = pd.read_parquet(train_target)
    test_X = pd.read_parquet(test_data)

    logging.info('evaluate model on %d items', len(train_y))

    miou_mean, miou_std = evaluate(train_X, train_y, BaselineRegressor)

    logging.info('miou = %.7f ± %.7f', miou_mean, miou_std)
    logging.info('fit model and apply model to inference set')

    test_y = BaselineRegressor().fit(train_X, train_y).predict(test_X)
    test_y.to_csv(test_target, header=False, index=False)

    logging.info('save predictions to %s', test_target)
    logging.info('done.')


@main.command()
@click.argument('train-data', type=click.Path(exists=True, dir_okay=False))
@click.argument('train-target', type=click.Path(exists=True, dir_okay=False))
@click.argument('test-data', type=click.Path(exists=True, dir_okay=False))
@click.argument('test-target', type=click.Path(exists=False, dir_okay=False))
def linear_correction(train_data: str, train_target: str,
                      test_data: str, test_target: str):
    """Линейная регрессия для коррекции пользовательских разметок.
    """
    logging.info('load train set and inference set')

    train_X = pd.read_parquet(train_data)
    train_y = pd.read_parquet(train_target)
    test_X = pd.read_parquet(test_data)

    logging.info('evaluate model on %d items', len(train_y))

    miou_mean, miou_std = evaluate(train_X, train_y, LinearCorrectionRegressor)

    logging.info('miou = %.7f ± %.7f', miou_mean, miou_std)
    logging.info('fit model and apply model to inference set')

    test_y = LinearCorrectionRegressor().fit(train_X, train_y).predict(test_X)
    test_y.to_csv(test_target, header=False, index=False)

    logging.info('save predictions to %s', test_target)
    logging.info('done.')


@main.command()
@click.option('--avg-mode',
              default='before',
              type=click.Choice(['after', 'before']),
              help='How to avarage user markups.')
@click.argument('train-data', type=click.Path(exists=True, dir_okay=False))
@click.argument('train-target', type=click.Path(exists=True, dir_okay=False))
@click.argument('test-data', type=click.Path(exists=True, dir_okay=False))
@click.argument('test-target', type=click.Path(exists=False, dir_okay=False))
def linear_correction(avg_mode: str,
                      train_data: str, train_target: str,
                      test_data: str, test_target: str):
    """Линейная регрессия для коррекции пользовательских разметок.
    """
    def fabricate(**kwargs):
        return BoundingBoxRegressor(avg_mode=avg_mode, **kwargs)

    logging.info('load train set and inference set')

    train_X = pd.read_parquet(train_data)
    train_y = pd.read_parquet(train_target)
    test_X = pd.read_parquet(test_data)

    logging.info('evaluate model on %d items', len(train_y))

    miou_mean, miou_std = evaluate(train_X, train_y, fabricate)

    logging.info('miou = %.7f ± %.7f', miou_mean, miou_std)
    logging.info('fit model and apply model to inference set')

    test_y = fabricate().fit(train_X, train_y).predict(test_X)
    test_y.to_csv(test_target, header=False, index=False)

    logging.info('save predictions to %s', test_target)
    logging.info('done.')


@main.command()
@click.argument('src', type=click.Path(dir_okay=False, exists=True))
@click.argument('dst', type=click.Path(dir_okay=False))
def dropout(src: str, dst: str):
    """Занулить половину координат исходного решения.

    Данная команда случайно выбирает 314 прямоугольника и зануляет их. Это
    сделано с тем, чтобы попробовать поспользоваться утечкой информации о части
    набора данных, который используется для проверки решения.
    """
    logging.info('load submission from %s', src)
    names = ['item_id', 'x_min', 'y_min', 'x_max', 'y_max']
    frame = pd.read_csv(src, names=names)

    logging.info('fabricate permutation without one exclusive element')
    exclusive = frame[frame.item_id == 146].index.values[0]
    indices = np.arange(len(frame) - 1)
    indices[exclusive:] += 1
    perm = np.random.permutation(indices)
    half = perm[:len(frame) // 2 - 1]

    logging.info('mutate submission and write to %s', dst)
    frame.loc[half, names[1:]] = 0.0
    frame.to_csv(dst, header=False, index=False)
    logging.info('done.')
