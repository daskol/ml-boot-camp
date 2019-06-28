#   encoding: utf8
#   filename: cli.py

import click
import logging
import pandas as pd

from os.path import basename

from .client import Client


@click.group()
def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)


@main.command()
@click.option('-c', '--comment', default='')
@click.option('-f', '--force',
              default=False,
              is_flag=True,
              help='Force to obtain new session identifier.')
@click.option('-p', '--password', metavar='pass', envvar='MLBOOTCAMP_PASS')
@click.option('-u', '--username', metavar='user', envvar='MLBOOTCAMP_USER')
@click.argument('task-id', type=int)
@click.argument('filename', type=click.Path(dir_okay=False, exists=True))
def submit(comment: str, force: bool, password: str, username: str,
           task_id: int, filename: str):
    """Отправить решение в проверочную систему.
    """
    logging.info('instantiate mlbootcamp client')
    cli = Client()

    if force or (password and username):
        logging.info('sign in with username `%s` to the system', username)
        cli.login(password, username, force)

    logging.info('submit solution')
    logging.info('    from file `%s`', basename(filename))
    logging.info('    on task %d', task_id)

    if comment:
        logging.info('    with comment `%s`', comment)
    else:
        logging.info('    without any comment')

    with open(filename, 'rb') as fin:
        cli.submit(task_id, fin, comment, basename(filename))

    logging.info('done.')


@main.command()
@click.option('-f', '--force',
              default=False,
              is_flag=True,
              help='Force to obtain new session identifier.')
@click.option('-p', '--password', metavar='pass', envvar='MLBOOTCAMP_PASS')
@click.option('-u', '--username', metavar='user', envvar='MLBOOTCAMP_USER')
@click.argument('task-id', type=int)
def history(force: bool, password: str, username: str, task_id: int):
    """
    """
    logging.info('instantiate mlbootcamp client')
    cli = Client()

    if force or (password and username):
        logging.info('sign in with username `%s` to the system', username)
        cli.login(password, username, force)

    logging.info('request history for task %d', task_id)
    json = cli.history(task_id)
    history = next(iter(json['history'].values()))
    solutions = history['solutions']

    logging.info('process submission list')

    def format_place(value):
        if isinstance(value, list):
            return f'{value[0]}-{value[1]}'
        else:
            return str(value)

    names = ['score', 'place', 'chosen', 'filename', 'accepted at', 'comment']
    selected_columns = ['id', 'rd_score', 'rd_place', 'chosen', 'file_name',
                        'accepted', 'comment']

    frame = pd.DataFrame(solutions)
    frame = frame[frame.state == 'ОК']
    frame = frame[selected_columns]
    frame = frame.set_index('id').sort_index()
    frame.columns = names
    frame['place'] = frame.place.apply(format_place)

    del frame.index.name

    print()
    print(frame)
    print()

    logging.info('done.')


from mlbootcamp.rnd21 import cli as rnd21  # noqa


main.add_command(rnd21.main, name='rnd21')
