from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence, Set, Tuple, Union

from ..core import Machine, Transition


ActiveConfig = Union[str, Tuple[Any, ...]]
CostKey = Union[Transition, Tuple[str, str, Optional[str]], str]
Costs = Union[Mapping[object, Union[int, float]], Callable[[str, Transition], Union[int, float]]]
ResetSpec = Optional[Union[str, Tuple[str, Union[int, float]]]]


class Step(NamedTuple):
    trigger: str
    transition: Transition
    cost: Union[int, float]
    reset: bool


class Edge(NamedTuple):
    source: ActiveConfig
    target: ActiveConfig
    trigger: str
    transition: Transition
    cost: Union[int, float]
    reset: bool


class TransitionPair(NamedTuple):
    source_transition: Transition
    target_transition: Transition


class CoverageGap(NamedTuple):
    kind: str
    reason: str
    config: Optional[ActiveConfig]
    trigger: Optional[str]
    transition: Optional[Transition]


class TestPlan(NamedTuple):
    steps: Tuple[Step, ...]
    cost: Union[int, float]


class TestPlanSet(NamedTuple):
    plans: Tuple[TestPlan, ...]
    pairs: Tuple[TransitionPair, ...]
    gaps: Tuple[CoverageGap, ...]
    optimal: bool


class PlanExecution(NamedTuple):
    plan: TestPlan
    transitions: Tuple[Transition, ...]
    pairs: Tuple[TransitionPair, ...]
    steps: Tuple[Step, ...]


def generate_transition_pair_plans(
    machine: Machine,
    costs: Optional[Costs] = ...,
    reset: ResetSpec = ...,
    reset_cost: Union[int, float] = ...,
    default_cost: Union[int, float] = ...,
    include_auto_transitions: bool = ...,
) -> TestPlanSet: ...


def execute_plan(machine: Machine, plan: TestPlan, model: Optional[object] = ...) -> PlanExecution: ...


def execute_plans(machine: Machine, plans: Sequence[TestPlan],
                   model: Optional[object] = ...) -> Tuple[PlanExecution, ...]: ...


def verify_coverage(executions: Sequence[PlanExecution],
                    required_pairs: Sequence[TransitionPair]) -> Sequence[TransitionPair],
                                                                       Set[TransitionPair]]: ...

