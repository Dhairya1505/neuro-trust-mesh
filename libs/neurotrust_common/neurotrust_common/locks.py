import asyncio
from collections import defaultdict


class KeyedLock:
    """One asyncio.Lock per key (e.g. agent_id), created on first use.

    aio-pika dispatches each consumed message as its own asyncio.Task
    (see aio_pika.queue.consumer: `create_task(callback, message)`), so two
    events for the SAME agent arriving close together on the same or
    different subscriptions run as genuinely concurrent tasks in this
    process. Every service handler here follows a read-modify-write
    pattern (read a JSON list column, append, write it back) which is not
    safe under that concurrency -- without serializing per agent_id, one
    update can silently clobber another with no error, no crash, just a
    quietly dropped event.
    """

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def get(self, key: str) -> asyncio.Lock:
        return self._locks[key]
