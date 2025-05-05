from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Generic, Literal

from ._types import ContextT, EndpointT, ExceptionT, OptionalExceptionT, SuccessT, T


def _now():
    return datetime.now(UTC)


@dataclass(kw_only=True, frozen=True)
class ResultSummary(Generic[SuccessT, OptionalExceptionT, T]):
    success: SuccessT
    result: T
    exception: OptionalExceptionT
    args: tuple
    kwds: dict[str, Any]
    start: datetime
    end: datetime = field(default_factory=_now, init=False)

    @property
    def duration(self):
        return self.end - self.start


@dataclass(kw_only=True, frozen=True)
class SuccessSummary(ResultSummary[Literal[True], None, T], Generic[T]):
    success: Literal[True] = field(default=True, init=False)
    exception: None = field(default=None, init=False)


@dataclass(frozen=True)
class SuccessDetail(SuccessSummary[T], Generic[T, ContextT, EndpointT]):
    message: str
    context: ContextT
    endpoint: EndpointT


@dataclass(kw_only=True, frozen=True)
class FailureSummary(
    ResultSummary[Literal[False], ExceptionT, None], Generic[ExceptionT]
):
    success: Literal[False] = field(default=False, init=False)
    result: None = field(default=None, init=False)


@dataclass(frozen=True)
class FailureDetail(
    FailureSummary[ExceptionT], Generic[ExceptionT, ContextT, EndpointT]
):
    message: str
    context: ContextT
    endpoint: EndpointT
