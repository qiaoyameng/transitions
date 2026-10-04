from typing import Any, Callable, Dict, IO, Optional, Tuple, Union


class SCXMLParseError(ValueError): ...


class SCXMLAdapter:
    unsupported_elements: frozenset
    callbacks: Dict[str, Callable[..., Any]]
    conditions: Dict[str, Callable[..., Any]]
    separator: str
    state_map: Dict[str, str]
    def __init__(self, callbacks: Optional[Dict[str, Callable[..., Any]]] = ...,
                 conditions: Optional[Dict[str, Callable[..., Any]]] = ...,
                 separator: Optional[str] = ...) -> None: ...
    def parse(self, source: Union[str, bytes, IO[Any]]) -> Dict[str, Any]: ...
    def resolve(self, markup: Dict[str, Any]) -> Dict[str, Any]: ...


def parse_scxml(source: Union[str, bytes, IO[Any]],
                callbacks: Optional[Dict[str, Callable[..., Any]]] = ...,
                conditions: Optional[Dict[str, Callable[..., Any]]] = ...,
                separator: Optional[str] = ...) -> Tuple[Dict[str, Any], Dict[str, str]]: ...
