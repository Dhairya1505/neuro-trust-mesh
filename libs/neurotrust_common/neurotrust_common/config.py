import os
from functools import lru_cache

from pydantic import BaseModel


class Settings(BaseModel):
    """Common settings every service reads from the environment.

    Kept as a plain pydantic model (not BaseSettings) so services with
    extra fields can subclass without pulling pydantic-settings as a dep.
    """

    service_name: str
    rabbitmq_url: str = "amqp://guest:guest@rabbitmq:5672/"
    postgres_dsn: str = "postgresql+asyncpg://neurotrust:neurotrust@postgres:5432/neurotrust"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, service_name: str) -> "Settings":
        return cls(
            service_name=service_name,
            rabbitmq_url=os.environ.get("RABBITMQ_URL", cls.model_fields["rabbitmq_url"].default),
            postgres_dsn=os.environ.get("POSTGRES_DSN", cls.model_fields["postgres_dsn"].default),
            log_level=os.environ.get("LOG_LEVEL", cls.model_fields["log_level"].default),
        )


@lru_cache
def get_settings(service_name: str) -> Settings:
    return Settings.from_env(service_name)
