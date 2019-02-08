#   encoding: utf8
#   filename: client.py

from io import BytesIO
from os import getenv
from typing import Iterable

from bs4 import BeautifulSoup
from requests import Session


class Client:
    """Client to ML Boot Camp competition facility. It has not any API so this
    client works over HTML. It is built on top of requests libraray and bs4
    parser. In fact we should perform at least two request on each call in
    order to pass CSRF token.

    :param session: Use provided Session object and not create a new one.

    >>> cli = Client()
    >>> cli.login(email='admin@example.org', password='notarealpassword')
    >>> cli.submit(task_id=15, solution=[1, 0, 0.5], comment='doctest.')
    """

    URL = 'https://mlbootcamp.ru{endpoint}'

    __slot__ = ['session', 'session_id']

    def __init__(self, session: Session=None, session_id: str=None):
        self.session = session or Session()
        self.session_id = session_id or self._load_session_id()

    def history(self, task_id: int):
        """This function requests submition history for a given task.

        :param task_id: Task identifier which could be found in page URI.
        :return: JSON object containing submition history.
        """
        url = self.URL.format(endpoint='/round/%d/my-history/' % task_id)
        res = self.session.get(url)

        if not res.ok:
            message = 'Failed to get submition history for task #%d.' % task_id
            raise RuntimeError(message)

        return res.json()

    def login(self, email: str, password: str):
        """Function login signs in with email and password. It is needed to set
        sessionid cookie.

        :param email: User email.
        :param password: User password.
        :return: Value of sessionid cookie.
        """
        url = self.URL.format(endpoint='/login/')
        res = self.session.get(url)

        if not res.ok:
            raise RuntimeError('Request was failed.')

        data = dict(csrfmiddlewaretoken=self._extract_csrf_token(res.text),
                    email=email,
                    password=password)
        res = self.session.post(url, data=data)

        if not res.ok or not res.cookies.get('sessionid'):
            return False

        return res.cookies.get('sessionid')

    def submit(self, task_id: int, solution: Iterable, comment: str=None,
               filename: str=None):
        """Funcdtion submit adds new solution to a task. In this implimentation
        we assume that solution is iterable which should be serialized to a
        column of floating points. Also, user could provide a comment to the
        solution.

        :param task_id: Task identifier which could be found in page URI.
        :param solution: Iterable of integers which marshals to single column.
        :param comment: Comment on a solution.
        :param filename: What filename use in submition.
        """
        url = self.URL.format(endpoint='/round/%s/tasks/' % task_id)
        res = self.session.get(url)

        if not res.ok:
            raise RuntimeError('Request was failed.')

        solution = '\n'.join(str(x) for x in solution) + '\n'
        stream = BytesIO()
        stream.write(solution.encode('utf8'))
        stream.seek(0)

        data = dict(csrfmiddlewaretoken=self._extract_csrf_token(res.text),
                    comment=comment,
                    task=self._extract_task_id(res.text))
        filename = filename or 'solution.csv'
        files = dict(file=(filename, stream, 'text/csv; charset=utf-8'))

        url = self.URL.format(endpoint='/round/%d/solution/add/' % task_id)
        res = self.session.post(url, data=data, files=files)

        if not res.ok or res.json().get('status') == 'ERROR':
            raise RuntimeError('Failed submit solution for task %s.' % task_id)

    def _extract_csrf_token(self, content: str) -> str:
        """This function helps to find and extract CSRF token from HTML body.

        :param content: HTML content.
        :return: CSRF token.
        """
        html = BeautifulSoup(content, 'html.parser')
        inputs = html.find_all('input', attrs=dict(
            type='hidden',
            name='csrfmiddlewaretoken',
        ))

        if len(inputs) == 0:
            raise RuntimeError('There is no CSRF token.')

        return inputs[0].get('value')

    def _extract_task_id(self, content: str) -> int:
        """This function helps to find real task id.

        :param content: HTML content.
        :return: Real task id for solution submition.
        """
        html = BeautifulSoup(content, 'html.parser')
        buttons = html.find_all('button')
        buttons = [button for button in buttons if button.get('data-task')]

        if len(buttons) == 0:
            raise RuntimeError('There is tag with data-task attribute.')

        return int(buttons[0].get('data-task'))

    def _load_session_id(self) -> str:
        confdir = getenv('XDG_CONFIG_HOME', '~/.config')
        confdir = join(confdir, 'ml-bootcamp')

        with open(join(confdir, 'session')) as fin:
            self.session_id = fin.read().strip()
