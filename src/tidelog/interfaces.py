import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Awaitable
from functools import wraps, partial
from string import Template
from typing import Any, Generic, TypeVar, overload, Annotated, Self

from fastapi.params import Depends

from ._types import ContextT, ExceptionT, P, T, MessageTemplate, EndpointT
from .context import LogRecordContextT, AsyncLogRecordContextT, AnyLogRecordContextT
from .models import FailureDetail, FailureSummary, SuccessDetail, SuccessSummary
from .utils import (
    is_annotation,
    is_awaitable,
    is_failure,
    get_annotation_metadata,
    get_annotation_type,
    is_success,
    list_parameters,
    sync_execute,
    update_signature,
    async_execute,
)


_SuccessDetailT = TypeVar("_SuccessDetailT", bound=SuccessDetail)
_FailureDetailT = TypeVar("_FailureDetailT", bound=FailureDetail)


class Handler(Generic[_SuccessDetailT, _FailureDetailT, P]):
    def before(self, *args: P.args, **kwds: P.kwargs): ...
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
_UtilFunctionT = TypeVar("_UtilFunctionT", bound=Callable)


class _AbstractLogRecord(
    ABC,
    Generic[
        _HandlerT,
        _SuccessHandlerT,
        _FailureHandlerT,
        AnyLogRecordContextT,
        EndpointT,
        _UtilFunctionT,
    ],
):
    _log_record_deps_name = "extra"
    _endpoint_deps_name = "endpoint_deps"

    def __init__(
        self,
        *,
        success: MessageTemplate | None = None,
        failure: MessageTemplate | None = None,
        functions: list[_UtilFunctionT] | dict[str, _UtilFunctionT] | None = None,
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

        self.functions: dict[str, _UtilFunctionT] = {}

        self.handlers = handlers or []
        self.success_handlers = success_handlers or []
        self.failure_handlers = failure_handlers or []

        # 用于判断当前装饰的是哪个端点
        self._endpoints: dict[Callable, EndpointT] = {}
        self._bind = {}

        if dependencies:
            if isinstance(dependencies, dict):
                for name, dep in dependencies.items():
                    self.add_dependency(dep, name)
            else:
                for dep in dependencies:
                    self.add_dependency(dep)

        if functions:
            if isinstance(functions, dict):
                for name, fn in functions.items():
                    self.register_function(fn, name)
            else:
                for fn in functions:
                    self.register_function(fn)

    @overload
    def register_function(self, fn: _UtilFunctionT): ...
    @overload
    def register_function(self, fn: _UtilFunctionT, name: str): ...
    def register_function(self, fn: _UtilFunctionT, name: str | None = None):
        self.functions[name or fn.__name__] = fn

    def description(self) -> str | None: ...

    @overload
    def add_dependency(self, dependency: Depends): ...
    @overload
    def add_dependency(self, dependency: Depends, name: str): ...
    def add_dependency(self, dependency: Depends, name: str | None = None):
        assert callable(dependency.dependency), (
            "The dependency must be a callable function"
        )
        name = name or (dependency.dependency and dependency.dependency.__name__)

        if name in self.dependencies:
            raise ValueError(f"The dependency name {name} is already in use")

        self.dependencies.setdefault(name, dependency)

    def add_handler(self, handler: _HandlerT, /):
        self.handlers.append(handler)

    def add_success_handler(self, handler: _SuccessHandlerT, /):
        self.success_handlers.append(handler)

    def add_failure_handler(self, handler: _FailureHandlerT, /):
        self.failure_handlers.append(handler)

    def _bind_fn(self, endpoint: EndpointT, fn: Callable):
        self._bind.setdefault(fn, endpoint)

    def _log_record_deps(self, endpoint: EndpointT):
        if not self.dependencies:
            return None

        def log_record_dependencies(**kwargs):
            return kwargs

        parameters = []
        for name, dep in self.dependencies.items():
            assert dep.dependency is not None

            parameters.append(
                inspect.Parameter(
                    name=name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=Depends(
                        self._log_function(dep.dependency, endpoint),
                        use_cache=dep.use_cache,
                    ),
                )
            )

        update_signature(log_record_dependencies, parameters=parameters)

        return log_record_dependencies

    def _wrap_dependency(self, parameter: inspect.Parameter, endpoint: EndpointT):
        default = parameter.default
        annotation = parameter.annotation
        # e.g.
        # 1. def endpoint(value=Depends(dependency_function)): ...
        # 2. >>>>>>
        #    class Value:
        #        def __init__(self, demo: int):
        #            self.demo = demo
        #
        #    def endpoint(value: Value = Depends()): ...
        #    <<<<<<
        if isinstance(default, Depends):
            # handle 1
            if default.dependency:
                new_dep = Depends(
                    self._log_function(default.dependency, endpoint),
                    use_cache=default.use_cache,
                )
                self._bind.setdefault(default.dependency, endpoint)
                return parameter.replace(default=new_dep)
            # handle 2
            elif inspect.isclass(annotation):
                self._bind.setdefault(default.dependency, endpoint)
                return parameter.replace(
                    annotation=self._log_function(annotation, endpoint)
                )

        # e.g.
        # 1. >>>>>>
        #    class Value:
        #        def __init__(self, demo: int):
        #            self.demo = demo
        #
        #    def endpoint(value: Annotated[Value, Depends()]): ...
        #    <<<<<<
        #
        # 2. >>>>>>
        #    def endpoint(value: Annotated[Value, Depends(dependency_function)]): ...
        #    <<<<<<
        elif is_annotation(annotation):
            typ = get_annotation_type(annotation)
            metadata = []
            cls_dep = True
            for i in get_annotation_metadata(annotation):
                if isinstance(i, Depends):
                    if i.dependency:
                        cls_dep = False
                        new_dep = Depends(
                            self._log_function(i.dependency, endpoint),
                            use_cache=default.use_cache,
                        )
                        metadata.append(new_dep)
                    else:
                        metadata.append(i)
            if cls_dep and inspect.isclass(typ):
                typ = self._log_function(typ, endpoint)

            return parameter.replace(annotation=Annotated[typ, *metadata])

        return parameter

    def _endpoint_deps(self, endpoint: EndpointT) -> Callable | None:
        if parameters := list_parameters(endpoint):

            def endpoint_deps(*args, **kwargs):
                return args, kwargs

            update_signature(
                endpoint_deps,
                parameters=[self._wrap_dependency(p, endpoint) for p in parameters],
            )
            return endpoint_deps

        return None

    @abstractmethod
    def _log_function(self, fn: Callable, endpoint: EndpointT) -> Callable: ...

    @classmethod
    def new(
        cls,
        *,
        success: MessageTemplate | None = None,
        failure: MessageTemplate | None = None,
        functions: list[_UtilFunctionT] | dict[str, _UtilFunctionT] | None = None,
        dependencies: list[Depends] | dict[str, Depends] | None = None,
        context_factory: Callable[[], AnyLogRecordContextT] | None = None,
        handlers: list[_HandlerT] | None = None,
        success_handlers: list[_SuccessHandlerT] | None = None,
        failure_handlers: list[_FailureHandlerT] | None = None,
    ) -> Self:
        return cls(
            success=success,
            failure=failure,
            functions=functions,
            dependencies=dependencies,
            context_factory=context_factory,
            handlers=handlers,
            success_handlers=success_handlers,
            failure_handlers=failure_handlers,
        )

    @overload
    def __call__(self, endpoint: EndpointT) -> EndpointT: ...

    @overload
    def __call__(
        self,
        *,
        success: MessageTemplate | None = None,
        failure: MessageTemplate | None = None,
        functions: list[_UtilFunctionT] | dict[str, _UtilFunctionT] | None = None,
        dependencies: list[Depends] | dict[str, Depends] | None = None,
        context_factory: Callable[[], AnyLogRecordContextT] | None = None,
        handlers: list[_HandlerT] | None = None,
        success_handlers: list[_SuccessHandlerT] | None = None,
        failure_handlers: list[_FailureHandlerT] | None = None,
    ) -> Callable[[EndpointT], EndpointT]: ...

    def __call__(
        self,
        endpoint: EndpointT | None = None,
        *,
        success: MessageTemplate | None = None,
        failure: MessageTemplate | None = None,
        functions: list[_UtilFunctionT] | dict[str, _UtilFunctionT] | None = None,
        dependencies: list[Depends] | dict[str, Depends] | None = None,
        context_factory: Callable[[], AnyLogRecordContextT] | None = None,
        handlers: list[_HandlerT] | None = None,
        success_handlers: list[_SuccessHandlerT] | None = None,
        failure_handlers: list[_FailureHandlerT] | None = None,
    ):
        def none_or(new, old) -> Any:
            return old if new is None else new

        if endpoint is None:
            return self.new(
                success=none_or(success, self.success),
                failure=none_or(failure, self.failure),
                functions=none_or(functions, self.functions),
                dependencies=none_or(dependencies, self.dependencies),
                context_factory=none_or(context_factory, self.context_factory),
                handlers=none_or(handlers, self.handlers),
                success_handlers=none_or(success_handlers, self.success_handlers),
                failure_handlers=none_or(failure_handlers, self.failure_handlers),
            )

        ofn = endpoint

        # 日志记录器本身所需的依赖
        log_record_deps = self._log_record_deps(endpoint)
        parameters = []
        if callable(log_record_deps):
            parameters.append(
                inspect.Parameter(
                    name=self._log_record_deps_name,
                    default=Depends(log_record_deps),
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            )

        # 端点的依赖
        endpoint_deps = self._endpoint_deps(endpoint)
        if callable(endpoint_deps):
            parameters.append(
                inspect.Parameter(
                    name=self._endpoint_deps_name,
                    default=Depends(endpoint_deps),
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            )

        new_fn = partial(endpoint)
        update_signature(new_fn, parameters=parameters)
        wraps(endpoint)(new_fn)

        self._endpoints[new_fn] = ofn

        return self._log_function(new_fn, endpoint)


class AbstractLogRecord(
    _AbstractLogRecord[
        Handler[_SuccessDetailT, _FailureDetailT, P],
        Callable[[_SuccessDetailT], None],
        Callable[[_FailureDetailT], None],
        LogRecordContextT,
        EndpointT,
        Callable[P, Any],
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
        kwargs["__"] = {
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
        endpoint: EndpointT,
    ) -> _FailureDetailT:
        raise NotImplementedError

    def _log_function(self, fn: Callable, endpoint: EndpointT):
        @wraps(fn)
        def decorator(*args, **kwds):
            is_endpoint_fn = fn in self._endpoints

            log_record_deps = None
            context: ContextT | None = None

            if is_endpoint_fn:
                for i in self.handlers:
                    i.before(*args, **kwds)

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
                    endpoint=endpoint,
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

            return summary.result

        return decorator


class AbstractAsyncLogRecord(
    _AbstractLogRecord[
        AsyncHandler[_SuccessDetailT, _FailureDetailT, P],
        Callable[[_SuccessDetailT], None | Awaitable[None]],
        Callable[[_FailureDetailT], None | Awaitable[None]],
        AsyncLogRecordContextT,
        EndpointT,
        Callable[P, Awaitable | Any],
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
        endpoint: EndpointT,
    ) -> _FailureDetailT:
        raise NotImplementedError

    def _log_function(self, fn: Callable, endpoint: EndpointT):
        @wraps(fn)
        async def decorator(*args, **kwds):
            is_endpoint_fn = fn in self._endpoints

            for i in self.handlers:
                await i.before(*args, **kwds)

            log_record_deps = None
            context: ContextT | None = None

            if is_endpoint_fn:
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
                    endpoint=endpoint,
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

            return summary.result

        return decorator
