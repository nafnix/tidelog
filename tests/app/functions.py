from tidelog import FailureDetail, SuccessDetail
from devtools import debug


def get(*args, __: SuccessDetail | FailureDetail, **kwds):
    debug(__, "lalala")
    return "result"
