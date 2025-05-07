from tidelog import FailureDetail, SuccessDetail

from .database import OperationLog, session


def failure_handler(detail: FailureDetail):
    with session.begin() as db:
        db.add(OperationLog(id=2, data=detail.message))


def success_handler(detail: SuccessDetail):
    with session.begin() as db:
        db.add(OperationLog(id=1, data=detail.message))
