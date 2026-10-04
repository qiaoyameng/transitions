"""Deterministic finite state machine minimization (experimental).

This module merges states of a deterministic :class:`~transitions.core.Machine`
that are indistinguishable from the outside. It works on a descriptive view of
the machine: callbacks are never executed, they participate in comparisons
only via stable identifiers supplied by the caller.

Observable semantics
---------------------
A machine is treated as a Moore/Mealy hybrid over a fixed event alphabet:

* the *state output* of a state is the tuple
  ``(observable label, final flag, enter action ids)`` and is observed whenever
  the state is entered, including the initial state;
* a taken transition emits its exit action ids, its transition action ids
  (``prepare``/``before``/``after`` in that order) and the state output of the
  reached state;
* an internal transition (``dest=None``) emits only its transition action ids;
* an event without a transition emits the uniform :data:`REJECTED` token and
  leaves the state unchanged. It is therefore never confused with a self loop.

Restrictions of this first version
-----------------------------------
* only unconditional transitions are accepted -- a transition carrying
  ``conditions`` or ``unless`` makes the machine statically non-deterministic
  and the analysis is refused;
* each (state, event) pair must have at most one candidate transition;
* the event alphabet is fixed (explicitly provided or taken from the machine);
* observable state labels are provided explicitly;
* the rejection policy (``ignore_invalid_triggers``) must be uniform.
"""

from collections import OrderedDict
from typing import (
    Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, NamedTuple,
    Optional, Sequence, Tuple,
)

from transitions.core import Machine, State


__all__ = ["REJECTED", "MinimizationError", "MinimizationResult",
           "minimize_machine", "simulate", "default_action_key"]


class MinimizationError(ValueError):
    """Raised when a machine does not fit the supported deterministic subset."""


# Observable token emitted for an event without a transition. It is a distinct
# sentinel so a rejected event can never look like a self loop.
REJECTED = object()

StateName = str
EventName = str
ActionKey = Any
LabelFunc = Callable[[State], Any]
ActionKeyFunc = Callable[[Any], ActionKey]

# State output observed upon entering a state: (label, final, enter action ids).
StateObservation = Tuple[Any, bool, Tuple[ActionKey, ...]]

# One observable step after feeding an event:
#   rejected -> ("rejected",)
#   internal -> ("internal", transition action ids)
#   taken    -> ("taken", exit action ids, transition action ids, state output)
ObservedStep = Tuple[Any, ...]
ObservedTrace = Tuple[StateObservation, List[Tuple[EventName, ObservedStep]]]
_Pair = Tuple[StateName, StateName]
_ParentInfo = Tuple[Optional[_Pair], Optional[EventName]]


def default_action_key(callback: Any) -> ActionKey:
    """Return the stable identifier of a callback.

    String callback names are used as-is. Arbitrary callables cannot be
    compared across machines by identity, so callers must provide their own
    ``action_key`` function for such callbacks; callbacks are never invoked.
    """
    if isinstance(callback, str):
        return callback
    raise MinimizationError(
        "Callback {!r} is not a stable identifier. Pass an 'action_key' "
        "callable mapping every callback to a hashable identifier; callbacks "
        "are never executed during minimization.".format(callback))


def _as_label_func(labels: Any) -> LabelFunc:
    if callable(labels):
        return labels
    if isinstance(labels, Mapping):
        mapping = dict(labels)

        def label_from_mapping(state: State) -> Any:
            try:
                return mapping[state.name]
            except KeyError as err:
                raise MinimizationError(
                    "Missing observable label for state {!r}.".format(state.name)) from err
        return label_from_mapping
    raise MinimizationError(
        "'labels' must map state names to labels or be a callable taking a State")


class _Edge(NamedTuple):
    # Normalized outgoing transition of a (state, event) pair.
    rejected: bool
    internal: bool
    dest: Optional[StateName]          # None for rejected or internal edges
    exit_actions: Tuple[ActionKey, ...]
    trans_actions: Tuple[ActionKey, ...]


class _Model(object):
    """Validated, purely descriptive view of a deterministic machine."""

    def __init__(self, machine: Machine, alphabet: Sequence[EventName],
                 label_func: LabelFunc, action_key: ActionKeyFunc) -> None:
        self.machine = machine
        self.states = list(machine.states.keys())  # type: List[StateName]
        if machine.initial is None:
            raise MinimizationError("Machine has no initial state.")
        self.initial = machine.initial  # type: StateName
        self.alphabet = list(alphabet)
        self._validate_plain_states()
        self.enter = self._state_actions("on_enter", action_key)
        self.exit_actions = self._state_actions("on_exit", action_key)
        self.labels = self._build_labels(label_func)
        self.observations = self._build_observations()
        self.ignore = self._uniform_ignore_flag()
        self.edges = self._build_edges(action_key)

    def _validate_plain_states(self) -> None:
        for state in self.machine.states.values():
            if type(state) is not State:
                raise MinimizationError(
                    "Only plain (non-nested) State instances are supported; "
                    "state {!r} is of type {}.".format(
                        state.name, type(state).__name__))

    def _state_actions(self, attr: str, action_key: ActionKeyFunc
                       ) -> Dict[StateName, Tuple[ActionKey, ...]]:
        result = {}
        for name, state in self.machine.states.items():
            result[name] = tuple(action_key(cb) for cb in getattr(state, attr))
        return result

    def _build_labels(self, label_func: LabelFunc) -> Dict[StateName, Any]:
        labels = {}
        for name, state in self.machine.states.items():
            label = label_func(state)
            try:
                hash(label)
            except TypeError as err:
                raise MinimizationError(
                    "Observable label of state {!r} is not hashable.".format(name)) from err
            labels[name] = label
        return labels

    def _build_observations(self) -> Dict[StateName, StateObservation]:
        return {
            name: (self.labels[name], bool(self.machine.states[name].final), self.enter[name])
            for name in self.states}

    def _uniform_ignore_flag(self) -> Optional[bool]:
        flags = set()
        for state in self.machine.states.values():
            flag = state.ignore_invalid_triggers
            if flag is None:
                flag = self.machine.ignore_invalid_triggers
            flags.add(flag)
        if len(flags) > 1:
            raise MinimizationError(
                "States disagree on 'ignore_invalid_triggers'; a single "
                "uniform rejection semantics for missing transitions is required.")
        return next(iter(flags))

    def _build_edges(self, action_key: ActionKeyFunc
                     ) -> Dict[StateName, Dict[EventName, _Edge]]:
        edges = {name: OrderedDict() for name in self.states}  # type: Dict[StateName, Dict[EventName, _Edge]]
        for event_name in self.alphabet:
            event = self.machine.events.get(event_name)
            for state_name in self.states:
                candidates = list(event.transitions.get(state_name, [])) if event is not None else []
                if not candidates:
                    edges[state_name][event_name] = _Edge(True, False, None, (), ())
                    continue
                if len(candidates) > 1:
                    raise MinimizationError(
                        "State {!r} has {} candidate transitions for event {!r}; "
                        "at most one deterministic candidate is allowed.".format(
                            state_name, len(candidates), event_name))
                edges[state_name][event_name] = self._normalize(candidates[0], action_key)
        return edges

    def _normalize(self, transition: Any, action_key: ActionKeyFunc) -> _Edge:
        if transition.conditions:
            raise MinimizationError(
                "Transition from {!r} to {!r} carries conditions; conditional "
                "branches are not supported.".format(transition.source, transition.dest))
        trans_actions = tuple(action_key(cb) for cb in
                              list(transition.prepare) + list(transition.before) + list(transition.after))
        if transition.dest is None:
            return _Edge(False, True, None, (), trans_actions)
        dest = transition.dest  # type: StateName
        if dest not in self.machine.states:
            raise MinimizationError(
                "Transition from {!r} points to unknown state {!r}.".format(
                    transition.source, dest))
        # enter actions are observed as part of the reached state's observation
        return _Edge(False, False, dest, self.exit_actions[transition.source], trans_actions)


def _observed_step(model: _Model, current: StateName, edge: _Edge) -> ObservedStep:
    if edge.rejected:
        return ("rejected",)
    if edge.internal:
        return ("internal", edge.trans_actions)
    return ("taken", edge.exit_actions, edge.trans_actions,
            model.observations[edge.dest])


def _edge_signature(edge: _Edge, block_of: Callable[[StateName], int]) -> Tuple[Any, ...]:
    if edge.rejected:
        return ("rejected",)
    if edge.internal:
        return ("internal", edge.trans_actions)
    return ("taken", edge.exit_actions, edge.trans_actions, block_of(edge.dest))


def _refine(model: _Model) -> Dict[StateName, int]:
    # Initial partition distinguishes the Moore output of the states.
    blocks = OrderedDict()  # type: OrderedDict[StateObservation, List[StateName]]
    for name in model.states:
        blocks.setdefault(model.observations[name], []).append(name)
    block_ids = {}
    for index, members in enumerate(blocks.values()):
        for member in members:
            block_ids[member] = index

    while True:
        new_blocks = OrderedDict()  # type: OrderedDict[Tuple[Any, ...], List[StateName]]
        for name in model.states:
            edge_sig = tuple((event_name,
                              _edge_signature(model.edges[name][event_name], block_ids.__getitem__))
                             for event_name in model.alphabet)
            new_blocks.setdefault((block_ids[name], edge_sig), []).append(name)
        new_block_ids = {}
        for index, members in enumerate(new_blocks.values()):
            for member in members:
                new_block_ids[member] = index
        if new_block_ids == block_ids:
            return new_block_ids
        block_ids = new_block_ids


def _representatives(model: _Model, block_ids: Dict[StateName, int]
                     ) -> Tuple[Dict[StateName, StateName], List[StateName]]:
    chosen = OrderedDict()  # type: OrderedDict[int, StateName]
    representative = {}
    for name in model.states:  # earliest declared state of a block wins
        block = block_ids[name]
        if block not in chosen:
            chosen[block] = name
        representative[name] = chosen[block]
    return representative, list(chosen.values())


def _build_quotient(model: _Model, representatives: Mapping[StateName, StateName],
                    representative_states: List[StateName],
                    machine_cls: Callable[..., Machine]) -> Machine:
    quotient_states = []
    for name in representative_states:
        state = model.machine.states[name]
        quotient_states.append(State(
            name=name,
            on_enter=list(state.on_enter),
            on_exit=list(state.on_exit),
            ignore_invalid_triggers=state.ignore_invalid_triggers,
            final=state.final))
    quotient = machine_cls(
        model=None,
        states=quotient_states,
        initial=representatives[model.initial],
        auto_transitions=False,
        ignore_invalid_triggers=model.ignore)
    for event_name in model.alphabet:
        for source in representative_states:
            edge = model.edges[source][event_name]
            if edge.rejected:
                continue
            transition = None  # type: Optional[Any]
            if edge.internal:
                dest = None
                for candidate in model.machine.events[event_name].transitions[source]:
                    if candidate.dest is None:
                        transition = candidate
                        break
            else:
                target = edge.dest  # type: StateName
                dest = representatives[target]
                for candidate in model.machine.events[event_name].transitions[source]:
                    if candidate.dest == target:
                        transition = candidate
                        break
            quotient.add_transition(
                trigger=event_name,
                source=source,
                dest=dest,
                prepare=list(transition.prepare) if transition is not None else [],
                before=list(transition.before) if transition is not None else [],
                after=list(transition.after) if transition is not None else [])
    return quotient


def simulate(model: _Model, initial: StateName,
             events: Iterable[EventName]) -> ObservedTrace:
    """Replay an event string on a normalized view without executing anything.

    Returns the state output observed at the starting state together with one
    observable step per event. A missing transition yields the uniform
    ``("rejected",)`` token and the current state does not change.
    """
    current = initial
    steps = []  # type: List[Tuple[EventName, ObservedStep]]
    for event_name in events:
        edge = model.edges[current][event_name]
        steps.append((event_name, _observed_step(model, current, edge)))
        if not edge.rejected and not edge.internal:
            current = edge.dest
    return model.observations[current], steps


class MinimizationResult(object):
    """Quotient machine, representative mapping and analysis helpers."""

    def __init__(self, model: _Model, quotient: Machine,
                 representatives: Dict[StateName, StateName],
                 block_ids: Dict[StateName, int]) -> None:
        self.original = model.machine
        self.quotient = quotient
        self.mapping = dict(representatives)
        groups = {}  # type: Dict[int, List[StateName]]
        for name, block in block_ids.items():
            groups.setdefault(block, []).append(name)
        self.partition = {}  # type: Dict[StateName, FrozenSet[StateName]]
        for members in groups.values():
            frozen = frozenset(members)
            for member in members:
                self.partition[member] = frozen
        self._model = model
        self._block_ids = block_ids
        quotient_labels = {name: model.labels[name] for name in representatives.values()}
        self._quotient_view = _Model(
            quotient, model.alphabet,
            lambda state: quotient_labels[state.name], lambda value: value)

    def equivalent(self, state_a: StateName, state_b: StateName) -> bool:
        """Return whether the two original states were merged."""
        return self._block_ids[state_a] == self._block_ids[state_b]

    def witness(self, state_a: StateName, state_b: StateName
                ) -> Optional[List[EventName]]:
        """Return a shortest event string whose observations differ.

        Returns ``None`` when the states are equivalent. The returned string is
        a concrete counterexample rather than a mere partition statement:
        replaying it from the two states yields different observable sequences,
        starting from the initial state output. A rejected event contributes
        the distinct ``("rejected",)`` token.
        """
        if self._block_ids[state_a] == self._block_ids[state_b]:
            return None
        return _shortest_distinguishing_string(self._model, state_a, state_b)

    def verify_quotient(self, max_length: Optional[int] = None
                        ) -> Optional[Tuple[List[EventName], ObservedTrace, ObservedTrace]]:
        """Enumerate short event strings and compare original vs quotient traces.

        Returns ``None`` when every event string up to ``max_length``
        (defaulting to the number of original states) produces the same
        observable sequence in the original and in the quotient. On mismatch a
        tuple ``(events, original_trace, quotient_trace)`` is returned.
        """
        limit = len(self._model.states) if max_length is None else max_length
        if limit < 0:
            raise MinimizationError("'max_length' must not be negative.")
        quotient_initial = self.mapping[self._model.initial]
        for events in _event_strings(self._model.alphabet, limit):
            original_trace = simulate(self._model, self._model.initial, events)
            quotient_trace = simulate(self._quotient_view, quotient_initial, events)
            if original_trace != quotient_trace:
                return list(events), original_trace, quotient_trace
        return None


def _event_strings(alphabet: Sequence[EventName], max_length: int
                   ) -> Iterable[Tuple[EventName, ...]]:
    yield ()
    if max_length == 0 or not alphabet:
        return
    frontier = [()]  # type: List[Tuple[EventName, ...]]
    for _ in range(max_length):
        next_frontier = []
        for prefix in frontier:
            for event_name in alphabet:
                candidate = prefix + (event_name,)
                yield candidate
                next_frontier.append(candidate)
        frontier = next_frontier


def _shortest_distinguishing_string(model: _Model, start_a: StateName,
                                    start_b: StateName) -> List[EventName]:
    # Breadth-first search over unordered state pairs with parent pointers so
    # the first observable difference found yields a shortest event string.
    def canon(pair: Tuple[StateName, StateName]) -> Tuple[StateName, StateName]:
        return pair if pair <= (pair[1], pair[0]) else (pair[1], pair[0])

    start = canon((start_a, start_b))
    if model.observations[start[0]] != model.observations[start[1]]:
        return []
    # pair -> (predecessor pair, event used to reach it)
    parents = {start: (None, None)}  # type: Dict[_Pair, _ParentInfo]
    queue = [start]
    while queue:
        next_queue = []
        for pair in queue:
            left, right = pair
            for event_name in model.alphabet:
                edge_a = model.edges[left][event_name]
                edge_b = model.edges[right][event_name]
                observed_a = _observed_step(model, left, edge_a)
                observed_b = _observed_step(model, right, edge_b)
                if observed_a != observed_b:
                    return _reconstruct(parents, pair, event_name)
                if edge_a.rejected or edge_b.rejected or edge_a.internal or edge_b.internal:
                    # Equal observations and the state is unchanged, so no new
                    # successor pair can emerge.
                    continue
                next_a = edge_a.dest  # type: StateName
                next_b = edge_b.dest  # type: StateName
                child = canon((next_a, next_b))
                if child[0] == child[1] or child in parents:
                    continue
                parents[child] = (pair, event_name)
                if model.observations[child[0]] != model.observations[child[1]]:
                    return _reconstruct(parents, child, None)
                next_queue.append(child)
        queue = next_queue
    raise MinimizationError(  # pragma: no cover - defensive
        "States {!r} and {!r} are in different partitions but no "
        "distinguishing event string exists.".format(start_a, start_b))


def _reconstruct(parents: Mapping[_Pair, _ParentInfo], pair: _Pair,
                 last_event: Optional[EventName]) -> List[EventName]:
    events = []  # type: List[EventName]
    if last_event is not None:
        events.append(last_event)
    current = pair  # type: Optional[_Pair]
    while current is not None:
        parent, event_name = parents[current]
        if event_name is not None:
            events.append(event_name)
        current = parent
    events.reverse()
    return events


def minimize_machine(machine: Machine,
                     labels: Any,
                     alphabet: Optional[Sequence[EventName]] = None,
                     action_key: ActionKeyFunc = default_action_key,
                     machine_cls: Optional[Callable[..., Machine]] = None
                     ) -> MinimizationResult:
    """Minimize a deterministic :class:`~transitions.core.Machine`.

    Args:
        machine: The machine to analyze. Only plain states and unconditional,
            uniquely resolved transitions are supported.
        labels: Either a mapping of state name to a hashable observable label,
            or a callable taking a :class:`~transitions.core.State` and
            returning one. States with different labels are never merged.
        alphabet: Fixed event alphabet. Defaults to every event registered on
            the machine (in declaration order).
        action_key: Maps each enter/exit/transition callback to a stable,
            hashable identifier. Callbacks are never called; the default
            accepts string callback names and rejects arbitrary callables.
        machine_cls: Class used to build the quotient machine. Defaults to the
            analyzed machine's class. It is constructed without a model and
            with ``auto_transitions=False``.

    Returns:
        MinimizationResult with the quotient machine, the original state to
        representative state mapping, the equivalence partition, a shortest
        witness finder and a bounded quotient verification helper.

    Raises:
        MinimizationError: If the machine contains conditional branches,
            multiple candidate transitions for one (state, event) pair, nested
            states, ambiguous rejection semantics or unstable callbacks.
    """
    if alphabet is None:
        event_names = list(machine.events.keys())
    else:
        event_names = list(alphabet)
        unknown = [name for name in event_names if name not in machine.events]
        if unknown:
            raise MinimizationError(
                "Alphabet contains events not registered on the machine: {}.".format(unknown))
    if len(set(event_names)) != len(event_names):
        raise MinimizationError("The event alphabet must not contain duplicates.")

    label_func = _as_label_func(labels)
    model = _Model(machine, event_names, label_func, action_key)
    block_ids = _refine(model)
    representatives, representative_states = _representatives(model, block_ids)
    quotient = _build_quotient(
        model, representatives, representative_states,
        machine_cls if machine_cls is not None else type(machine))
    return MinimizationResult(model, quotient, representatives, block_ids)
