import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase as DB
from sqlalchemy.orm import Mapped as M
from sqlalchemy.orm import mapped_column as m
from sqlalchemy.orm import sessionmaker


class Base(DB):
    id: M[int] = m(autoincrement=True, primary_key=True)


class OperationLog(Base):
    __tablename__ = "operation_logs"
    data: M[str]


engine = sa.create_engine("sqlite+pysqlite:///file.sqlite3", echo=True)

session = sessionmaker(engine, expire_on_commit=False, autoflush=False)
