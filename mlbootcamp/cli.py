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
    """История посылок в проверяющую систему.
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


@main.command()
@click.option('-f', '--force',
              default=False,
              is_flag=True,
              help='Force to obtain new session identifier.')
@click.option('-N', '--nickname', type=str, help='Nickname to highlight.')
@click.option('-n', '--lines', default=0, type=int)
@click.option('-o', '--output',
              type=click.Path(dir_okay=False),
              help='Filename where to save leaderboard.')
@click.option('-p', '--password', metavar='pass', envvar='MLBOOTCAMP_PASS')
@click.option('-u', '--username', metavar='user', envvar='MLBOOTCAMP_USER')
@click.argument('task-id', type=int)
def leaderboard(force: bool, lines: int, nickname: str, output: str,
                password: str, username: str, task_id: int):
    """Рейтинг участников.
    """
    logging.info('instantiate mlbootcamp client')
    cli = Client()

    if force or (password and username):
        logging.info('sign in with username `%s` to the system', username)
        cli.login(password, username, force)

    logging.info('request leaderboard page')
    html = cli.leaderboard(task_id)

    logging.info('parse html content of the page')
    lis = html.find_all('li')
    records = []

    for li in lis:
        spans = li.find_all('span')

        if len(spans) >= 1:
            name = spans[0].text

        if len(spans) >= 2:
            nick = spans[1].text
            nick = nick[1:-1].strip()
            if nick.startswith('&nbsp'):
                nick = nick[5:]
        else:
            nick = ''

        divs = li.find_all('div')

        if len(divs) > 0:
            ordinal = int(divs[0].text)

        if len(divs) > 4:
            score = float(divs[4].text)

        record = (ordinal, name, score, nick)
        records.append(record)

    # Combind all parsed record to data frame.
    frame: pd.DataFrame = pd.DataFrame(
        data=records,
        columns=['ordinal', 'name', 'score', 'nick'],
    ).set_index('ordinal')

    del frame.index.name

    # Keep only head or tail of origin frame if filter is specified.
    if lines > 0:
        frame = frame[:lines]
    if lines < 0:
        frame = frame[lines:]

    # Save leaderboard to file if filename of output file is specified.
    if output:
        logging.info('save leaderboard to file `%s`', output)
        frame.to_csv(output)

    # Serialize to human-readable representaion with some display settings.
    with pd.option_context('display.max_rows', 200, 'display.precision', 7):
        lines = str(frame).splitlines()

    # If nickname is given that we need to find row in a subframe in order to
    # get relative position of row with the nickname in a subframe.
    if nickname:
        reseted = frame.reset_index()
        selected = reseted[reseted.nick == nickname]

        # If there is a row with a given nickname then highlight it!
        if len(selected) > 0:
            index = selected.index[0] + 1
            lines[index] = f'\033[1;30;43m{lines[index]}\033[0m'

    print()
    print('\n'.join(lines))
    print()

    logging.info('done.')


from mlbootcamp.rnd21 import cli as rnd21  # noqa


main.add_command(rnd21.main, name='rnd21')
