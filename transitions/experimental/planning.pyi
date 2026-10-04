from numbers import Real
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

from transitions.core import Machine, Transition

ACTION: str
RESET: str

StepKind = str
TransitionIdentity = Tuple[str, str, str, int]
CostKey = Union[int, Transition, Tuple[str, ...], str]
CostMap = Union[Dict[Any, Real], Dict[Any, float], Dict[Any, int]]
ResetCallable = Callable[[], Any]

class RawTransition(NamedTuple):
    trigger: str
    source: str
    dest: Optional[str]
    transition: Transition
    scope: List[str]
    index: int

class ResetOperation(NamedTuple):
    trigger: Optional[str]
    cost: Optional[Real]
    sources: List[str]
    dest: Optional[str]
    operation: Optional[ResetCallable]
    identity: Optional[str]

class Edge(NamedTuple):
    identity: TransitionIdentity
    trigger: str
    source: str
    dest: str
    transition: Optional[Transition]
    cost: Real
    kind: StepKind
    operation: Optional[ResetOperation]

class CoverageGap(NamedTuple):
    kind: str
    trigger: Optional[str]
    source: Optional[str]
    dest: Optional[str]
    message: str

class PlanStep(NamedTuple):
    kind: StepKind
    trigger: str
    identity: Union[TransitionIdentity, str]
    source: str
    dest: str
    cost: Real
    operation: Optional[ResetOperation]
    transition: Optional[Transition]

class TestPlan(NamedTuple):
    steps: List[PlanStep]
    events: List[str]
    cost: Real
    required_pairs: Set[Tuple[TransitionIdentity, TransitionIdentity]]
    coverage_gaps: List[CoverageGap]
    initial_configuration: Optional[str]
    optimal: bool

class PlanExecution(NamedTuple):
    plan: TestPlan
    steps: List[PlanStep]
    transition_identities: List[Union[TransitionIdentity, str]]
    adjacent_identities: List[Tuple[Union[TransitionIdentity, str], Union[TransitionIdentity, str]]]
    covered_pairs: Set[Tuple[TransitionIdentity, TransitionIdentity]]
    missing_pairs: Set[Tuple[TransitionIdentity, TransitionIdentity]]
    extra_pairs: Set[Tuple[TransitionIdentity, TransitionIdentity]]
    initial_configuration: Optional[str]
    final_configuration: Any
    successful: bool

ResetInput = Union[
    str,
    ResetCallable,
    ResetOperation,
    Dict[str, Any],
    Tuple[Any, ...],
    Sequence[Union[str, ResetCallable, ResetOperation, Dict[str, Any], Tuple[Any, ...]]]
]

def generate_transition_pair_test_plans(
    machine: Machine,
    costs: Optional[CostMap] = ...,
    reset: Optional[ResetInput] = ...,
    include_auto_transitions: bool = ...
) -> List[TestPlan]: ...

def execute_test_plan(plan: TestPlan, machine: Optional[Machine] = ...,
                      model: Optional[Any] = ...) -> PlanExecution: ...

def generate_and_execute_transition_pair_test_plan(
    machine: Machine,
    costs: Optional[CostMap] = ...,
    reset: Optional[ResetInput] = ...,
    include_auto_transitions: bool = ...
) -> List[PlanExecution]: ...
