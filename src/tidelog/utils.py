import inspect
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from datetime import UTC, datetime

# from functools import partial, update_wrapper
from typing import (
    Annotated,
    Any,
    NamedTuple,
    ParamSpec,
    TypeGuard,
    TypeVar,
    get_origin,
    # overload,
)

# from fastapi.params import Depends
# from pydantic.fields import FieldInfo

from .models import FailureSummary, ResultSummary, SuccessSummary

_P = ParamSpec("_P")
_T = TypeVar("_T")


class _Undefined:
    def __new__(cls):
        k = "_singleton"
        if not hasattr(cls, k):
            setattr(cls, k, super().__new__(cls))
        return getattr(cls, k)

    @classmethod
    def ne(cls, v):
        return v is not cls()


_undefined = _Undefined()


def is_awaitable(v: _T) -> TypeGuard[Awaitable[_T]]:
    return inspect.isawaitable(v)


def is_coroutine_function(
    v: Callable[_P, _T],
) -> TypeGuard[Callable[_P, Coroutine[Any, Any, _T]]]:
    return inspect.iscoroutinefunction(v)


def add_document(fn: Callable, document: str):
    if fn.__doc__ is None:
        fn.__doc__ = document
    else:
        fn.__doc__ += f"\n\n{document}"


# def new_function(
#     fn: Callable,
#     *,
#     parameters: Sequence[inspect.Parameter] | None | _Undefined = _undefined,
#     return_annotation: type | None | _Undefined = _undefined,
# ):
#     result = update_wrapper(partial(fn), fn)
#     update_signature(result, parameters=parameters, return_annotation=return_annotation)
#     return result


def list_parameters(fn: Callable, /) -> list[inspect.Parameter]:
    signature = inspect.signature(fn)
    return list(signature.parameters.values())


class WithParameterResult(NamedTuple):
    parameters: list[inspect.Parameter]
    parameter: inspect.Parameter
    parameter_index: int


# @overload
# def with_parameter(
#     fn: Callable,
#     *,
#     name: str,
#     annotation: type | Annotated,
# ) -> WithParameterResult: ...
# @overload
# def with_parameter(
#     fn: Callable,
#     *,
#     name: str,
#     default: Any,
# ) -> WithParameterResult: ...
# @overload
# def with_parameter(
#     fn: Callable,
#     *,
#     name: str,
#     annotation: type | Annotated,
#     default: Any,
# ) -> WithParameterResult: ...


# def with_parameter(
#     fn: Callable,
#     *,
#     name: str,
#     annotation: type | Annotated | _Undefined = _undefined,
#     default: Any = _undefined,
# ) -> WithParameterResult:
#     kwargs = {}
#     if annotation is not _undefined:
#         kwargs["annotation"] = annotation
#     if default is not _undefined:
#         kwargs["default"] = default

#     parameters = list_parameters(fn)
#     parameter = inspect.Parameter(
#         name=name,
#         kind=inspect.Parameter.KEYWORD_ONLY,
#         **kwargs,
#     )
#     index = -1
#     if parameters and parameters[index].kind == inspect.Parameter.VAR_KEYWORD:
#         parameters.insert(index, parameter)
#         index = -2
#     else:
#         parameters.append(parameter)

#     return WithParameterResult(parameters, parameter, index)


def update_signature(
    fn: Callable,
    *,
    parameters: Sequence[inspect.Parameter] | None | _Undefined = _undefined,
    return_annotation: type | None | _Undefined = _undefined,
):
    signature = inspect.signature(fn)

    if not isinstance(parameters, _Undefined):
        signature = signature.replace(parameters=parameters)

    if not isinstance(return_annotation, _Undefined):
        signature = signature.replace(return_annotation=return_annotation)

    setattr(fn, "__signature__", signature)


# def add_parameter(
#     fn: Callable,
#     *,
#     name: str,
#     annotation: type | Annotated | _Undefined = _undefined,
#     default: Any | _Undefined = _undefined,
# ):
#     """添加参数, 会将添加参数后的新函数返回"""

#     p = with_parameter(
#         fn,
#         name=name,
#         annotation=annotation,
#         default=default,
#     )

#     new_fn = update_wrapper(partial(fn), fn)
#     if p.parameters:
#         update_signature(new_fn, parameters=p.parameters)

#     return new_fn


# def is_dependency(value):
#     types = Depends | FieldInfo

#     if isinstance(value, types) or (
#         get_origin(value) is Annotated and isinstance(value.__metadata__[-1], types)
#     ):
#         return True

#     return False


def is_annotation(value) -> TypeGuard[Annotated]:
    return get_origin(value) is Annotated


def get_annotation_type(value: Annotated) -> type:
    return value.__origin__


def get_annotation_metadata(value: Annotated) -> tuple:
    return value.__metadata__


# def get_dependency(value) -> Depends | FieldInfo | None:
#     types = (Depends, FieldInfo)

#     if isinstance(value, types):
#         return value

#     if is_annotation(value) and isinstance(value.__metadata__[-1], types):
#         return value.__metadata__[-1]

#     return None


def is_success(summary: ResultSummary) -> TypeGuard[SuccessSummary]:
    return summary.success


def is_failure(summary: ResultSummary) -> TypeGuard[FailureSummary]:
    return not summary.success


async def async_execute(
    fn: Callable[_P, _T],
    *args: _P.args,
    **kwds: _P.kwargs,
) -> SuccessSummary[_T] | FailureSummary:
    start = datetime.now(UTC)

    try:
        result = fn(*args, **kwds)
        if is_awaitable(result):
            result = await result

        return SuccessSummary(start=start, result=result, args=args, kwds=kwds)
    except Exception as exception:
        return FailureSummary(start=start, exception=exception, args=args, kwds=kwds)


def sync_execute(
    fn: Callable[_P, _T], *args: _P.args, **kwds: _P.kwargs
) -> SuccessSummary[_T] | FailureSummary:
    start = datetime.now(UTC)

    try:
        result = fn(*args, **kwds)
        return SuccessSummary(start=start, result=result, args=args, kwds=kwds)
    except Exception as exception:
        return FailureSummary(start=start, exception=exception, args=args, kwds=kwds)
