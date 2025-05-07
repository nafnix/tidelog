import string
from contextlib import asynccontextmanager

from fastapi.applications import FastAPI
from fastapi.params import Depends

from tidelog import LogRecord

from . import dependencies, functions, service, database


database.Base.metadata.drop_all(database.engine)
database.Base.metadata.create_all(database.engine)


app = FastAPI()


log_record = LogRecord(
    success="成功了",
    success_handlers=[service.success_handler],
    failure_handlers=[service.failure_handler],
    functions=[functions.get],
)


@app.get(
    "/a",
)
@log_record(failure=string.Template("$get"))
def exc_endpoint(
    exc=Depends(dependencies.bad),
): ...


@app.get(
    "/b",
)
@log_record
def ok_endpoint(
    a=Depends(dependencies.A),
): ...
