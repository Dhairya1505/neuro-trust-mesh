"""Thin wrapper over aio-pika giving every service the same topic-exchange
pub/sub pattern described in the system architecture doc (§2, §12): one
exchange, routing keys like "agent.heartbeat", each service binds only the
routing keys it cares about.

Convention for REST vs. events: state MUTATIONS (role changes, task
reassignment, isolation, reinstatement) flow through this bus as events
(e.g. agent.role_change_requested), never a direct synchronous PATCH --
matching the system doc's "async event bus, not direct request-response"
requirement. Read-only QUERIES (fetching current agent state, candidate
lists, or federated_learning's bulky model-weight export/import) stay REST
-- a documented, deliberate exception: modeling every read as a
request/reply event or a locally-cached read-model would add real
complexity for no behavioral benefit at this system's scale.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

import aio_pika
from aio_pika.abc import AbstractIncomingMessage
from pydantic import BaseModel

from neurotrust_common.logging import ServiceLoggerAdapter

EXCHANGE_NAME = "neurotrust.events"

T = TypeVar("T", bound=BaseModel)
Handler = Callable[[T], Awaitable[None]]


class EventBus:
    def __init__(self, url: str, service_name: str):
        self._url = url
        self._service_name = service_name
        self._connection: aio_pika.RobustConnection | None = None
        self._channel: aio_pika.abc.AbstractChannel | None = None
        self._exchange: aio_pika.abc.AbstractExchange | None = None

    async def connect(self, retries: int = 10, delay_s: float = 3.0) -> None:
        """Retry on startup: a healthy RabbitMQ container can still report
        its healthcheck before the AMQP listener is fully accepting
        connections, and docker-compose `depends_on: condition:
        service_healthy` doesn't guard against that race.
        """
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                self._connection = await aio_pika.connect_robust(self._url)
                break
            except Exception as exc:  # noqa: BLE001 - retry any connection failure
                last_error = exc
                await asyncio.sleep(delay_s)
        else:
            raise ConnectionError(f"could not connect to RabbitMQ after {retries} attempts") from last_error

        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=20)
        self._exchange = await self._channel.declare_exchange(
            EXCHANGE_NAME, aio_pika.ExchangeType.TOPIC, durable=True
        )

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()

    async def publish(self, event: BaseModel) -> None:
        if self._exchange is None:
            raise RuntimeError("EventBus.connect() must be awaited before publish()")
        routing_key = event.routing_key  # type: ignore[attr-defined]
        body = event.model_dump_json().encode("utf-8")
        await self._exchange.publish(
            aio_pika.Message(
                body=body,
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=routing_key,
        )

    async def subscribe(
        self,
        routing_keys: list[str],
        schema: type[T],
        handler: Handler[T],
        queue_name: str | None = None,
    ) -> None:
        """Bind a durable queue (named after the service + routing keys) and
        consume forever, deserializing each message into `schema` before
        calling `handler`. Messages that fail to parse are nacked without
        requeue rather than poison-looping the queue.
        """
        if self._channel is None or self._exchange is None:
            raise RuntimeError("EventBus.connect() must be awaited before subscribe()")

        name = queue_name or f"{self._service_name}.{'.'.join(routing_keys)}"
        queue = await self._channel.declare_queue(name, durable=True)
        for key in routing_keys:
            await queue.bind(self._exchange, routing_key=key)

        log = ServiceLoggerAdapter(logging.getLogger(self._service_name), {"service": self._service_name})

        async def _on_message(message: AbstractIncomingMessage) -> None:
            async with message.process(ignore_processed=True):
                try:
                    payload = json.loads(message.body)
                    event = schema.model_validate(payload)
                except Exception:
                    log.exception(
                        "dropping unparseable message",
                        extra={"trace": {"routing_key": message.routing_key, "queue": name}},
                    )
                    await message.reject(requeue=False)
                    return
                try:
                    await handler(event)
                except Exception:
                    # Surface handler bugs through the same structured JSON
                    # logs every other decision is traced through, instead
                    # of letting them fall through to a raw stderr
                    # traceback that never reaches log aggregation. Dropped
                    # (not requeued) to avoid poison-looping a bad message.
                    log.exception(
                        "handler failed, message dropped",
                        extra={"trace": {"routing_key": message.routing_key, "queue": name}},
                    )
                    await message.reject(requeue=False)

        await queue.consume(_on_message)

    async def run_forever(self) -> None:
        await asyncio.Event().wait()
