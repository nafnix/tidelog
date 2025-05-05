from typing import Generic

from ._types import ContextT, ExceptionT, P, T, EndpointT
from .context import AsyncLogRecordContext, LogRecordContext
from .interfaces import AbstractAsyncLogRecord, AbstractLogRecord
from .models import FailureDetail, FailureSummary, SuccessDetail, SuccessSummary


class LogRecord(
    AbstractLogRecord[
        SuccessDetail[T, ContextT | None, EndpointT],
        FailureDetail[ExceptionT, ContextT | None, EndpointT],
        P,
        T,
        ExceptionT,
        LogRecordContext[ContextT],
        ContextT,
        EndpointT,
    ],
    Generic[P, T, ExceptionT, ContextT, EndpointT],
):
    def get_success_detail(
        self,
        *,
        summary: SuccessSummary[T],
        context: ContextT | None,
        message: str,
        endpoint: EndpointT,
    ) -> SuccessDetail[T, ContextT | None, EndpointT]:
        return SuccessDetail(
            message=message,
            context=context,
            result=summary.result,
            args=summary.args,
            kwds=summary.kwds,
            start=summary.start,
            endpoint=endpoint,
        )

    def get_failure_detail(
        self,
        *,
        summary: FailureSummary[ExceptionT],
        context: ContextT | None,
        message: str,
        endpoint: EndpointT,
    ) -> FailureDetail[ExceptionT, ContextT | None, EndpointT]:
        return FailureDetail(
            message=message,
            context=context,
            exception=summary.exception,
            args=summary.args,
            kwds=summary.kwds,
            start=summary.start,
            endpoint=endpoint,
        )


class AsyncLogRecord(
    AbstractAsyncLogRecord[
        SuccessDetail[T, ContextT | None, EndpointT],
        FailureDetail[ExceptionT, ContextT | None, EndpointT],
        P,
        T,
        ExceptionT,
        AsyncLogRecordContext[ContextT],
        ContextT,
        EndpointT,
    ],
    Generic[P, T, ExceptionT, ContextT, EndpointT],
):
    async def get_success_detail(
        self,
        *,
        summary: SuccessSummary[T],
        context: ContextT | None,
        message: str,
        endpoint: EndpointT,
    ) -> SuccessDetail[T, ContextT | None, EndpointT]:
        return SuccessDetail(
            message=message,
            context=context,
            result=summary.result,
            args=summary.args,
            kwds=summary.kwds,
            start=summary.start,
            endpoint=endpoint,
        )

    async def get_failure_detail(
        self,
        *,
        summary: FailureSummary[ExceptionT],
        context: ContextT | None,
        message: str,
        endpoint: EndpointT,
    ) -> FailureDetail[ExceptionT, ContextT | None, EndpointT]:
        return FailureDetail(
            message=message,
            context=context,
            exception=summary.exception,
            args=summary.args,
            kwds=summary.kwds,
            start=summary.start,
            endpoint=endpoint,
        )
