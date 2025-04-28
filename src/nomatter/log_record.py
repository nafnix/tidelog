import inspect
import secrets
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from functools import wraps
from string import Template
from types import TracebackType
from typing import Any, Generic, ParamSpec, TypeVar, cast

from fastapi.params import Depends

from .utils import (
    ExecFailureResult,
    ExecSuccessResult,
    add_document,
    add_parameter,
    update_signature,
    wrap_result,
)

_P = ParamSpec("_P")
_T = TypeVar("_T")


class InfoMaker(AbstractContextManager):
    def __init__(self) -> None:
        self.info = None

    def __enter__(self):
        self.info = None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.info = None
        return None

    def __call__(self, fn: Callable[_P, _T]):
        @wraps(fn)
        async def decorator(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            with self:
                result = fn(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                return cast(_T, result)

        return decorator


class _AbstractLogRecord(ABC, Generic[_P, _T]):
    def __init__(
        self,
        *,
        operation_type: str | None = None,
        success: str | Template | None = None,
        failure: str | Template | None = None,
        condition: Callable[_P, bool] | None = None,
        dependencies: Sequence[Depends] | dict[str, Depends] | None = None,
        info_maker: InfoMaker | None = None,
        functions: list[Callable] | None = None,
    ) -> None:
        self.__new_dependencies: set[str] = set()

        self.operation_type = operation_type
        self.success = success or ""
        self.failure = failure or ""
        self.condition = condition

        self.dependencies: dict[str, Depends] = {}
        if dependencies:
            if isinstance(dependencies, dict):
                for name, dep in dependencies.items():
                    self.dependencies[name] = dep
            else:
                for dep in dependencies:
                    if dep.dependency:
                        name = dep.dependency.__name__ + "_" + secrets.token_hex(8)
                        self.dependencies[name] = dep

        self.info_maker = info_maker or InfoMaker()

        self._functions: dict[str, Callable] = {}

        if functions:
            for fn in functions:
                self.register_function(fn)

    def register_function(self, fn):
        self._functions[fn.__name__] = fn
        return fn

    def document(self) -> str | None: ...

    def _deps_collector(self):
        def collect_cls_dependencies(**kwargs):
            return kwargs

        parameters = [
            inspect.Parameter(
                name=name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=dep,
            )
            for name, dep in self.dependencies.items()
        ]

        has_cls_deps = bool(parameters)
        if has_cls_deps:
            update_signature(collect_cls_dependencies, parameters=parameters)

        return collect_cls_dependencies

    def _add_dependencies(self, fn: Callable):
        """添加依赖, 会跳过同注解的依赖"""

        dep_collector = self._deps_collector()

        return add_parameter(
            fn,
            name=dep_collector.__name__,
            default=Depends(dep_collector),
        )

    def _update_signature(self, fn: Callable[_P, _T]):
        new_fn = self._add_dependencies(fn)

        document = self.document()
        if document:
            add_document(new_fn, document)

        return new_fn


class AbstractLogRecord(_AbstractLogRecord, ABC, Generic[_P, _T]):
    def format_message(
        self,
        message: str | Template,
        duration: int,
        result: _T | None,
        exception: Exception | None,
        info: Any,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ):
        result_ = ""
        kwargs["result"] = result
        kwargs["duration"] = duration
        kwargs["exception"] = exception
        kwargs["info"] = info
        if isinstance(message, str):
            result_ += message.format(*args, **kwargs)

        elif isinstance(message, Template):
            identifiers = message.get_identifiers()
            values = {}
            for i in identifiers:
                fn = self._functions.get(i)
                if fn:
                    values[i] = fn(*args, **kwargs)

            result_ += message.safe_substitute(
                **values,
                **kwargs,
            ).format(*args, **kwargs)

        return result_

    def log_record(self, fn: Callable[_P, _T]):
        @wraps(fn)
        def decorator(*args: _P.args, **kwds: _P.kwargs) -> _T:
            result = cast(
                ExecFailureResult | ExecSuccessResult,
                wrap_result(fn, *args, **kwds),
            )

            if result.success:
                message = self.format_message(
                    self.success,
                    result.duration,
                    result.result,
                    result.exception,
                    self.info_maker.info,
                    *args,
                    **kwds,
                )
                result = cast(ExecSuccessResult, result)
                self.on_success(
                    message,
                    result.duration,
                    result.result,
                    self.info_maker.info,
                    *args,
                    **kwds,
                )
                return result.result
            else:
                message = self.format_message(
                    self.failure,
                    result.duration,
                    result.result,
                    result.exception,
                    self.info_maker.info,
                    *args,
                    **kwds,
                )
                result = cast(ExecFailureResult, result)

                self.on_failure(
                    message,
                    result.duration,
                    result.exception,
                    self.info_maker.info,
                    *args,
                    **kwds,
                )

                raise cast(ExecFailureResult, result).exception

        return cast(Callable[_P, _T], decorator)

    @abstractmethod
    def on_success(
        self,
        message: str,
        duration: int,
        result: _T,
        info: Any,
        *args: _P.args,
        **kwds: _P.kwargs,
    ): ...

    @abstractmethod
    def on_failure(
        self,
        message: str,
        duration: int,
        exception: Exception,
        info: Any,
        *args: _P.args,
        **kwds: _P.kwargs,
    ): ...

    def __call__(self, fn: Callable[_P, _T]):
        if self.operation_type is None:
            self.operation_type = fn.__name__

        new_route = self._update_signature(fn)

        return self.info_maker(self.log_record(new_route))


class AbstractAsyncLogRecord(_AbstractLogRecord, ABC, Generic[_P, _T]):
    async def format_message(
        self,
        message: str | Template,
        duration: int,
        result: _T | None,
        exception: Exception | None,
        info: Any,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ):
        result_ = ""
        kwargs["result"] = result
        kwargs["duration"] = duration
        kwargs["exception"] = exception
        kwargs["info"] = info
        if isinstance(message, str):
            result_ += message.format(*args, **kwargs)

        elif isinstance(message, Template):
            identifiers = message.get_identifiers()
            values = {}
            for i in identifiers:
                fn = self._functions.get(i)
                if fn:
                    fn_result = fn(*args, **kwargs)
                    values[i] = (
                        (await fn_result)
                        if inspect.isawaitable(fn_result)
                        else fn_result
                    )

            result_ += message.safe_substitute(
                **values,
                **kwargs,
            ).format(*args, **kwargs)

        return result_

    async def log_record(self, fn: Callable[_P, _T]):
        @wraps(fn)
        async def decorator(*args: _P.args, **kwds: _P.kwargs) -> _T:
            result = cast(
                ExecFailureResult | ExecSuccessResult,
                wrap_result(fn, *args, **kwds),
            )

            if result.success:
                message = await self.format_message(
                    self.success,
                    result.duration,
                    result.result,
                    result.exception,
                    self.info_maker.info,
                    *args,
                    **kwds,
                )
                result = cast(ExecSuccessResult, result)
                await self.on_success(
                    message,
                    result.duration,
                    result.result,
                    self.info_maker.info,
                    *args,
                    **kwds,
                )
                return result.result
            else:
                message = await self.format_message(
                    self.failure,
                    result.duration,
                    result.result,
                    result.exception,
                    self.info_maker.info,
                    *args,
                    **kwds,
                )
                result = cast(ExecFailureResult, result)

                await self.on_failure(
                    message,
                    result.duration,
                    result.exception,
                    self.info_maker.info,
                    *args,
                    **kwds,
                )

                raise cast(ExecFailureResult, result).exception

        return cast(Callable[_P, _T], decorator)

    @abstractmethod
    async def on_success(
        self,
        message: str,
        duration: int,
        result: _T,
        info: Any,
        *args: _P.args,
        **kwds: _P.kwargs,
    ): ...

    @abstractmethod
    async def on_failure(
        self,
        message: str,
        duration: int,
        exception: Exception,
        info: Any,
        *args: _P.args,
        **kwds: _P.kwargs,
    ): ...

    async def __call__(self, fn: Callable[_P, _T]):
        if self.operation_type is None:
            self.operation_type = fn.__name__

        new_route = self._update_signature(fn)

        return self.info_maker(await self.log_record(new_route))
