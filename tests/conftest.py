from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

from .app.database import Base, engine, session
from .app.main import app


@pytest.fixture(autouse=True)
def run_migrations():
    Base.metadata.create_all(engine)

    yield

    Base.metadata.drop_all(engine)


@asynccontextmanager
async def get_client(headers=None):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers=headers,
    ) as client:
        yield client


# see: https://anyio.readthedocs.io/en/stable/testing.html#specifying-the-backends-to-run-on
@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    async with get_client() as client:
        yield client


@pytest.fixture
def db():
    with session() as sess:
        yield sess
