# ML Bootcamp API

*HTML API to ML Bootcamp competition platform*

## Overview

### Usage Example

Assume that the password in environment variable `MLBOOTCAMP_PASSWORD` then one
can submit solution as the following.

```python
from os import getenv
from mlbootcamp import Client

cli = Client()
cli.login('daskol@example.org', getenv('MLBOOTCAMP_PASSWORD'))
cli.submit(task_id=15, solution=[0] * 948)
```


## Round 21

In order to reproduce experiments or implement his/her own model, one need to
run the following commands.

```bash
$ git clone git@github.com:daskol/ml-bootcamp.git
$ git checkout round/21
$ dvc pull
```

The last command requires access to remote SSH storage which is private in
fact.
