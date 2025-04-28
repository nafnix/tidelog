import inspect
from collections.abc import Callable, Sequence, Awaitable
from functools import partial, update_wrapper
from time import perf_counter
from typing import Annotated, Any, NamedTuple, get_origin, overload, Literal

from fastapi.params import Depends
from pydantic.fields import FieldInfo


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


def add_document(fn: Callable, document: str):
    if fn.__doc__ is None:
        fn.__doc__ = document
    else:
        fn.__doc__ += f"\n\n{document}"


def list_parameters(fn: Callable, /) -> list[inspect.Parameter]:
    signature = inspect.signature(fn)
    return list(signature.parameters.values())


class WithParameterResult(NamedTuple):
    parameters: list[inspect.Parameter]
    parameter: inspect.Parameter
    parameter_index: int


@overload
def with_parameter(
    fn: Callable, *, name: str, annotation: type | Annotated
) -> WithParameterResult: ...
@overload
def with_parameter(fn: Callable, *, name: str, default: Any) -> WithParameterResult: ...
@overload
def with_parameter(
    fn: Callable, *, name: str, annotation: type | Annotated, default: Any
) -> WithParameterResult: ...


def with_parameter(
    fn: Callable,
    *,
    name: str,
    annotation: type | Annotated | _Undefined = _undefined,
    default: Any = _undefined,
) -> WithParameterResult:
    kwargs = {}
    if annotation is not _undefined:
        kwargs["annotation"] = annotation
    if default is not _undefined:
        kwargs["default"] = default

    parameters = list_parameters(fn)
    parameter = inspect.Parameter(
        name=name,
        kind=inspect.Parameter.KEYWORD_ONLY,
        **kwargs,
    )
    index = -1
    if parameters and parameters[index].kind == inspect.Parameter.VAR_KEYWORD:
        parameters.insert(index, parameter)
        index = -2
    else:
        parameters.append(parameter)

    return WithParameterResult(parameters, parameter, index)


def update_signature(
    fn: Callable,
    *,
    parameters: Sequence[inspect.Parameter] | None = _Undefined,  # type: ignore
    return_annotation: type | None = _Undefined,
):
    signature = inspect.signature(fn)
    if parameters is not _Undefined:
        signature = signature.replace(parameters=parameters)
    if return_annotation is not _Undefined:
        signature = signature.replace(return_annotation=return_annotation)

    setattr(fn, "__signature__", signature)


def add_parameter(
    fn: Callable,
    *,
    name: str,
    annotation: type | Annotated | _Undefined = _undefined,
    default: Any | _Undefined = _undefined,
):
    """添加参数, 会将添加参数后的新函数返回"""

    p = with_parameter(
        fn,
        name=name,
        annotation=annotation,
        default=default,
    )

    new_fn = update_wrapper(partial(fn), fn)
    if p.parameters:
        update_signature(new_fn, *p.parameters)
    return new_fn


def is_dependency(value):
    if isinstance(value, Depends):
        return True

    if get_origin(value) is Annotated and isinstance(
        value.__metadata__[-1],
        (Depends, FieldInfo),
    ):
        return True

    return False


class ExecSuccessResult(NamedTuple):
    success: Literal[True]
    duration: int
    result: Any
    exception: None


class ExecFailureResult(NamedTuple):
    success: Literal[False]
    duration: int
    result: Any
    exception: Exception


def wrap_result(
    fn,
) -> Callable[
    ...,
    ExecSuccessResult
    | ExecFailureResult
    | Awaitable[ExecSuccessResult | ExecFailureResult],
]:
    def duration(start: float):
        return int((perf_counter() - start) * 1000)

    if inspect.iscoroutinefunction(fn):

        async def async_decorator(*args, **kwds):
            start = perf_counter()

            result = None

            try:
                result = await fn(*args, **kwds)
                return ExecSuccessResult(True, duration(start), result, None)
            except Exception as error:
                return ExecFailureResult(False, duration(start), result, error)

        return async_decorator

    def decorator(*args, **kwds):
        start = perf_counter()

        result = None

        try:
            result = fn(*args, **kwds)
            return ExecSuccessResult(True, duration(start), result, None)
        except Exception as error:
            return ExecFailureResult(False, duration(start), result, error)

    return decorator
