import pytest
import sqlalchemy as sa
from fastapi.applications import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.params import Depends
from httpx import AsyncClient
from sqlalchemy.orm import Session

from tidelog import LogRecord

from .app.database import OperationLog


@pytest.mark.anyio
class TestApp:
    async def test_a(self, client: AsyncClient, db: Session):
        resp = await client.get("/a")
        assert resp.is_error
        stmt = sa.select(OperationLog).filter_by(id=2)
        result = db.scalar(stmt)
        assert result is not None
        assert result.data == "result"

    async def test_b(self, client: AsyncClient, db: Session):
        resp = await client.get("/b", params={"name": 1})
        assert resp.is_success
        stmt = sa.select(OperationLog).filter_by(id=1)
        result = db.scalar(stmt)
        assert result is not None
        assert result.data == "成功了"
