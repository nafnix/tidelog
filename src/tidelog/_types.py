from string import Template
from typing import ParamSpec, TypeVar
from collections.abc import Callable

T = TypeVar("T")
P = ParamSpec("P")

ContextT = TypeVar("ContextT")

SuccessT = TypeVar("SuccessT", bound=bool)
OptionalExceptionT = TypeVar("OptionalExceptionT", bound=Exception | None)

ExceptionT = TypeVar("ExceptionT", bound=Exception)

MessageTemplate = str | Template

EndpointT = TypeVar("EndpointT", bound=Callable)
