import json
import logging
import sys
import time


class JsonFormatter(logging.Formatter):
    """Structured JSON logs so every service's output is machine-traceable.

    Every decision the system makes must be traceable back to specific
    evidence (design principle in the architecture docs) -- structured
    fields (event, agent_id, rule, evidence) let us grep/aggregate logs
    across services before a real observability stack exists.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "service": getattr(record, "service", "unknown"),
            "message": record.getMessage(),
        }
        extra = getattr(record, "trace", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ServiceLoggerAdapter(logging.LoggerAdapter):
    """logging.LoggerAdapter.process() unconditionally overwrites
    kwargs["extra"] with self.extra, silently discarding any extra=...
    passed at the call site -- every log.info(..., extra={"trace": {...}})
    call in this codebase was losing its trace payload because of this.
    Override process() to merge instead of replace.
    """

    def process(self, msg, kwargs):
        kwargs["extra"] = {**self.extra, **kwargs.get("extra", {})}
        return msg, kwargs


def get_logger(service_name: str, level: str = "INFO") -> ServiceLoggerAdapter:
    logger = logging.getLogger(service_name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return ServiceLoggerAdapter(logger, {"service": service_name})
