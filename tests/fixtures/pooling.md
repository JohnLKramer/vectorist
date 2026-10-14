# Connection Pooling

## Overview

Connection pooling reuses database connections to reduce latency and resource churn.

```python
def get_connection(pool):
    return pool.acquire()
```

| Strategy | Description |
| --- | --- |
| eager | Pre-warm connections at startup |
| lazy | Create connections on demand |

![Pooling diagram](images/pooling.png)
