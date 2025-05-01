import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Awaitable
from functools import wraps
from string import Template
from typing import Any, Generic, TypeVar, overload, cast

from fastapi.params import Depends

from ._types import ContextT, ExceptionT, P, T, MessageTemplate, EndpointT
from .context import LogRecordContextT, AsyncLogRecordContextT, AnyLogRecordContextT
from .models import FailureDetail, FailureSummary, SuccessDetail, SuccessSummary
from .utils import (
    add_parameter,
    is_awaitable,
    is_failure,
    is_success,
    list_parameters,
    sync_execute,
    update_signature,
    async_execute,
)


_SuccessDetailT = TypeVar("_SuccessDetailT", bound=SuccessDetail)
_FailureDetailT = TypeVar("_FailureDetailT", bound=FailureDetail)


class Handler(Generic[_SuccessDetailT, _FailureDetailT, P]):
    def before(self, Callable, *args: P.args, **kwds: P.kwargs): ...
    def after(
        self,
        detail: _SuccessDetailT | _FailureDetailT,
        *args: P.args,
        **kwds: P.kwargs,
    ): ...

    def success(
        self,
        detail: _SuccessDetailT,
        *args: P.args,
        **kwds: P.kwargs,
    ): ...
    def failure(
        self,
        detail: _FailureDetailT,
        *args: P.args,
        **kwds: P.kwargs,
    ): ...


class AsyncHandler(Generic[_SuccessDetailT, _FailureDetailT, P]):
    async def before(self, *args: P.args, **kwds: P.kwargs): ...
    async def after(
        self,
        detail: _SuccessDetailT | _FailureDetailT,
        *args: P.args,
        **kwds: P.kwargs,
    ): ...

    async def success(
        self,
        detail: _SuccessDetailT,
        *args: P.args,
        **kwds: P.kwargs,
    ): ...
    async def failure(
        self,
        detail: _FailureDetailT,
        *args: P.args,
        **kwds: P.kwargs,
    ): ...


_HandlerT = TypeVar("_HandlerT", bound=Handler | AsyncHandler)

_SuccessHandlerT = TypeVar("_SuccessHandlerT", bound=Callable)
_FailureHandlerT = TypeVar("_FailureHandlerT", bound=Callable)


class _AbstractLogRecord(
    ABC,
    Generic[
        _HandlerT, _SuccessHandlerT, _FailureHandlerT, AnyLogRecordContextT, EndpointT
    ],
):
    _log_record_deps_name = "extra"
    _endpoint_deps_name = "endpoint_deps"

    def __init__(
        self,
        *,
        success: MessageTemplate | None = None,
        failure: MessageTemplate | None = None,
        utils: list[Callable] | dict[str, Callable] | None = None,
        dependencies: list[Depends] | dict[str, Depends] | None = None,
        context_factory: Callable[[], AnyLogRecordContextT] | None = None,
        handlers: list[_HandlerT] | None = None,
        success_handlers: list[_SuccessHandlerT] | None = None,
        failure_handlers: list[_FailureHandlerT] | None = None,
    ) -> None:
        self.success = success or ""
        self.failure = failure or ""

        self.dependencies: dict[str, Depends] = {}

        self.context_factory = context_factory

        self.functions: dict[str, Callable] = {}

        self.handlers = handlers or []
        self.success_handlers = success_handlers or []
        self.failure_handlers = failure_handlers or []

        # 用于判断当前装饰的是哪个端点
        self._endpoints: dict[Callable, EndpointT] = {}

        if dependencies:
            if isinstance(dependencies, dict):
                for name, dep in dependencies.items():
                    self.add_dependency(dep, name)
            else:
                for dep in dependencies:
                    self.add_dependency(dep)

        if utils:
            if isinstance(utils, dict):
                for name, fn in utils.items():
                    self.register_function(fn, name)
            else:
                for fn in utils:
                    self.register_function(fn)

    @overload
    def register_function(self, fn: Callable): ...
    @overload
    def register_function(self, fn: Callable, name: str): ...
    def register_function(self, fn: Callable, name: str | None = None):
        self.functions[name or fn.__name__] = fn

    def description(self) -> str | None: ...

    @overload
    def add_dependency(self, dependency: Depends): ...
    @overload
    def add_dependency(self, dependency: Depends, name: str): ...
    def add_dependency(self, dependency: Depends, name: str | None = None):
        name = name or (dependency.dependency and dependency.dependency.__name__)
        assert name, "The dependency must be a callable function"

        if name in self.dependencies:
            raise ValueError(f"The dependency name {name} is already in use")

        self.dependencies[name] = dependency

    def add_handler(self, handler: _HandlerT, /):
        self.handlers.append(handler)

    def add_success_handler(self, handler: _SuccessHandlerT, /):
        self.success_handlers.append(handler)

    def add_failure_handler(self, handler: _FailureHandlerT, /):
        self.failure_handlers.append(handler)

    def _log_record_deps(self):
        if not self.dependencies:
            return None

        def log_record_dependencies(**kwargs):
            return kwargs

        parameters = [
            inspect.Parameter(
                name=name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=dep,
            )
            for name, dep in self.dependencies.items()
        ]

        update_signature(log_record_dependencies, parameters=parameters)

        return log_record_dependencies

    def _endpoint_deps(self, fn: Callable) -> Callable | None:
        if parameters := list_parameters(fn):

            def endpoint_deps(*args, **kwargs):
                return args, kwargs

            update_signature(endpoint_deps, parameters=parameters)
            return endpoint_deps

        return None

    @abstractmethod
    def _log_function(self, fn) -> Callable: ...

    def __call__(self, fn: EndpointT):
        ofn = fn

        new_fn = None
        # 日志记录器本身所需的依赖
        log_record_deps = self._log_record_deps()
        if callable(log_record_deps):
            new_fn = add_parameter(
                fn,
                name=self._log_record_deps_name,
                default=Depends(self._log_function(log_record_deps)),
            )

        # 端点的依赖
        endpoint_deps = self._endpoint_deps(fn)
        if callable(endpoint_deps):
            new_fn = add_parameter(
                new_fn if callable(new_fn) else fn,
                name=self._endpoint_deps_name,
                default=Depends(self._log_function(endpoint_deps)),
            )

        wrapped = new_fn or fn
        self._endpoints[wrapped] = ofn
        return self._log_function(wrapped)


class AbstractLogRecord(
    _AbstractLogRecord[
        Handler[_SuccessDetailT, _FailureDetailT, P],
        Callable[[_SuccessDetailT], None],
        Callable[[_FailureDetailT], None],
        LogRecordContextT,
        EndpointT,
    ],
    ABC,
    Generic[
        _SuccessDetailT,
        _FailureDetailT,
        P,
        T,
        ExceptionT,
        LogRecordContextT,
        ContextT,
        EndpointT,
    ],
):
    @overload
    def format_message(
        self,
        summary: SuccessSummary[T],
        extra: dict[str, Any] | None = None,
        context: ContextT | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> str: ...
    @overload
    def format_message(
        self,
        summary: FailureSummary[ExceptionT],
        extra: dict[str, Any] | None = None,
        context: ContextT | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> str: ...
    def format_message(
        self,
        summary: SuccessSummary[T] | FailureSummary[ExceptionT],
        extra: dict[str, Any] | None = None,
        context: ContextT | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> str:
        kwargs["$"] = {
            "summary": summary,
            self._log_record_deps_name: extra,
            "context": context,
        }

        message = self.success if summary.success else self.failure

        result_ = ""

        if isinstance(message, str):
            result_ += message.format(*args, **kwargs)

        elif isinstance(message, Template):
            identifiers = message.get_identifiers()
            values = {}
            for i in identifiers:
                fn = self.functions.get(i)
                if fn:
                    values[i] = fn(*args, **kwargs)

            result_ += message.safe_substitute(
                **values,
                **kwargs,
            ).format(*args, **kwargs)

        return result_

    @abstractmethod
    def get_success_detail(
        self,
        *,
        summary: SuccessSummary[T],
        message: str,
        context: ContextT | None,
        endpoint: EndpointT,
    ) -> _SuccessDetailT:
        raise NotImplementedError

    @abstractmethod
    def get_failure_detail(
        self,
        *,
        summary: FailureSummary,
        message: str,
        context: ContextT | None,
        endpoint: EndpointT | None,
    ) -> _FailureDetailT:
        raise NotImplementedError

    def _log_function(self, fn: Callable):
        @wraps(fn)
        def decorator(*args, **kwds):
            is_endpoint_fn = fn in self._endpoints
            endpoint = None

            for i in self.handlers:
                i.before(*args, **kwds)

            log_record_deps = None
            context: ContextT | None = None

            if is_endpoint_fn:
                endpoint = self._endpoints[fn]
                log_record_deps = kwds.pop(self._log_record_deps_name, None)

                kwds.setdefault(self._endpoint_deps_name, None)
                parameters: tuple[tuple, dict] = kwds.pop(self._endpoint_deps_name) or (
                    (),
                    {},
                )
                args, kwds = parameters

                if self.context_factory:
                    with self.context_factory() as ctx:
                        summary = sync_execute(fn, *args, **kwds)
                        context = ctx.info
                else:
                    summary = sync_execute(fn, *args, **kwds)

            else:
                summary = sync_execute(fn, *args, **kwds)

            if is_endpoint_fn and is_success(summary):
                message = self.format_message(
                    summary,
                    log_record_deps,
                    context,
                    *args,
                    **kwds,
                )
                detail = self.get_success_detail(
                    summary=summary,
                    context=context,
                    message=message,
                    endpoint=cast(EndpointT, endpoint),
                )

                for i in self.success_handlers:
                    i(detail)

                for i in self.handlers:
                    i.success(detail, *args, **kwds)
                    i.after(detail, *args, **kwds)

                return summary.result

            elif is_failure(summary):
                # 失败时, 依赖的上下文有可能是空的(例如如果是依赖项异常, 那么上下文是空的)
                # 如果是端点本身的异常, 则可能有值(具体看端点有没有触发上下文操作)
                message = self.format_message(
                    summary,
                    log_record_deps,
                    context,
                    *args,
                    **kwds,
                )
                detail = self.get_failure_detail(
                    summary=summary,
                    context=context,
                    message=message,
                    endpoint=endpoint,
                )

                for i in self.failure_handlers:
                    i(detail)

                for i in self.handlers:
                    i.failure(detail, *args, **kwds)
                    i.after(detail, *args, **kwds)

                raise summary.exception

        return decorator


class AbstractAsyncLogRecord(
    _AbstractLogRecord[
        AsyncHandler[_SuccessDetailT, _FailureDetailT, P],
        Callable[[_SuccessDetailT], None | Awaitable[None]],
        Callable[[_FailureDetailT], None | Awaitable[None]],
        AsyncLogRecordContextT,
        EndpointT,
    ],
    ABC,
    Generic[
        _SuccessDetailT,
        _FailureDetailT,
        P,
        T,
        ExceptionT,
        AsyncLogRecordContextT,
        ContextT,
        EndpointT,
    ],
):
    @overload
    async def format_message(
        self,
        summary: SuccessSummary[T],
        extra: dict[str, Any] | None = None,
        context: ContextT | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> str: ...
    @overload
    async def format_message(
        self,
        summary: FailureSummary[ExceptionT],
        extra: dict[str, Any] | None = None,
        context: ContextT | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> str: ...
    async def format_message(
        self,
        summary: SuccessSummary[T] | FailureSummary[ExceptionT],
        extra: dict[str, Any] | None = None,
        context: ContextT | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> str:
        kwargs["$"] = {
            "summary": summary,
            self._log_record_deps_name: extra,
            "context": context,
        }

        message = self.success if summary.success else self.failure

        result_ = ""

        if isinstance(message, str):
            result_ += message.format(*args, **kwargs)

        elif isinstance(message, Template):
            identifiers = message.get_identifiers()
            values = {}
            for i in identifiers:
                fn = self.functions.get(i)
                if fn:
                    fn_result = fn(*args, **kwargs)
                    if is_awaitable(fn_result):
                        fn_result = await fn_result
                    values[i] = fn_result

            result_ += message.safe_substitute(
                **values,
                **kwargs,
            ).format(*args, **kwargs)

        return result_

    @abstractmethod
    async def get_success_detail(
        self,
        *,
        summary: SuccessSummary[T],
        message: str,
        context: ContextT | None,
        endpoint: EndpointT,
    ) -> _SuccessDetailT:
        raise NotImplementedError

    @abstractmethod
    async def get_failure_detail(
        self,
        *,
        summary: FailureSummary,
        message: str,
        context: ContextT | None,
        endpoint: EndpointT | None,
    ) -> _FailureDetailT:
        raise NotImplementedError

    def _log_function(self, fn: Callable):
        @wraps(fn)
        async def decorator(*args, **kwds):
            is_endpoint_fn = fn in self._endpoints
            endpoint = None

            for i in self.handlers:
                await i.before(*args, **kwds)

            log_record_deps = None
            context: ContextT | None = None

            if is_endpoint_fn:
                endpoint = self._endpoints[fn]
                log_record_deps = kwds.pop(self._log_record_deps_name, None)
                args, kwds = kwds.pop(self._endpoint_deps_name, ((), {}))

                if self.context_factory:
                    async with self.context_factory() as ctx:
                        summary = await async_execute(fn, *args, **kwds)
                        context = ctx.info
                else:
                    summary = await async_execute(fn, *args, **kwds)

            else:
                summary = await async_execute(fn, *args, **kwds)

            if is_endpoint_fn and is_success(summary):
                message = await self.format_message(
                    summary,
                    log_record_deps,
                    context,
                    *args,
                    **kwds,
                )
                detail = await self.get_success_detail(
                    summary=summary,
                    context=context,
                    message=message,
                    endpoint=cast(EndpointT, endpoint),
                )

                for i in self.success_handlers:
                    i_result = i(detail)
                    if is_awaitable(i_result):
                        await i_result

                for i in self.handlers:
                    await i.success(detail, *args, **kwds)
                    await i.after(detail, *args, **kwds)

                return summary.result

            elif is_failure(summary):
                # 失败时, 依赖的上下文有可能是空的(例如如果是依赖项异常, 那么上下文是空的)
                # 如果是端点本身的异常, 则可能有值(具体看端点有没有触发上下文操作)
                message = await self.format_message(
                    summary,
                    log_record_deps,
                    context,
                    *args,
                    **kwds,
                )
                detail = await self.get_failure_detail(
                    summary=summary,
                    context=context,
                    message=message,
                    endpoint=endpoint,
                )

                for i in self.failure_handlers:
                    i_result = i(detail)
                    if is_awaitable(i_result):
                        await i_result

                for i in self.handlers:
                    await i.failure(detail, *args, **kwds)
                    await i.after(detail, *args, **kwds)

                raise summary.exception

        return decorator
