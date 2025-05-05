from fastapi import FastAPI, Depends, HTTPException, Query

from tidelog import LogRecord

app = FastAPI()


def exc_dep():
    raise HTTPException(400, detail="lalala")
    raise ValueError("2333")


class A:
    def __init__(self, name: int) -> None:
        self.name = name


from devtools import debug


log_record = LogRecord(
    success="lalalala",
    success_handlers=[lambda detail: print(detail)],
    failure_handlers=[lambda detail: debug(detail.duration)],
    # dependencies=[Depends(exc_dep)],
)


@app.get(
    "/a",
)
@log_record
def aaaa(b=Depends(exc_dep), a=Depends(A)):
    return id(object())


@app.get(
    "/b",
)
@log_record
def bbbb(c: str = Query()):
    return id(object()), c
