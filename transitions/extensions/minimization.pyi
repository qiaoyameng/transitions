from ..core import Machine
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, OrderedDict, Tuple, Union

REJECT: str

StateName = Optional[str]
EventWord = List[str]
Observation = Tuple[Tuple[Hashable, ...], Any]
ObservableSequence = List[Any]
LabelsArg = Union[Mapping[str, Hashable], Callable[[str], Hashable]]
ActionKey = Callable[[Any], Hashable]

class MinimizationError(ValueError): ...

class MinimizationResult:
    machine: Machine
    partition: OrderedDict[str, Tuple[str, ...]]
    mapping: OrderedDict[str, str]
    events: List[str]
    labels: Dict[str, Any]
    def observable_trace(self, state: StateName, word: Iterable[str]) -> ObservableSequence: ...
    def quotient_trace(self, state: StateName, word: Iterable[str]) -> ObservableSequence: ...
    def check_observations(self, max_length: int) -> List[Tuple[StateName, EventWord,
                                                                ObservableSequence,
                                                                ObservableSequence]]: ...
    def distinguish(self, state_a: StateName, state_b: StateName) -> Optional[EventWord]: ...

def minimize_machine(machine: Machine,
                     events: Union[str, Iterable[str]],
                     labels: Optional[LabelsArg] = ...,
                     action_key: Optional[ActionKey] = ...,
                     reject_state: Union[str, bool, None] = ...,
                     include_reject_state: Optional[bool] = ...) -> MinimizationResult: ...
