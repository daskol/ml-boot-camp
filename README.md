# ML Boot Camp

*HTML API to ML Boot Camp competition platform*

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
