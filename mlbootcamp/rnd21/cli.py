#   encoding: utf8
#   filename: cli.py

import click
import logging
import pandas as pd

from .baseline import BaselineRegressor
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
