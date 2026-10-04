from __future__ import absolute_import

import copy
import heapq
import inspect
import math
from enum import Enum
from collections import deque, namedtuple

from six import string_types

from ..core import Machine
from ..extensions.nesting import HierarchicalMachine

try:
    from ..extensions.asyncio import AsyncMachine
except (ImportError, SyntaxError):  # pragma: no cover
    class AsyncMachine(Machine):
        pass


Step = namedtuple('Step', ['trigger', 'transition', 'cost', 'reset'])
Edge = namedtuple('Edge', ['source', 'target', 'trigger', 'transition', 'cost', 'reset'])
TransitionPair = namedtuple('TransitionPair', ['source_transition', 'target_transition'])
CoverageGap = namedtuple('CoverageGap', ['kind', 'reason', 'config', 'trigger', 'transition'])
TestPlan = namedtuple('TestPlan', ['steps', 'cost'])
TestPlanSet = namedtuple('TestPlanSet', ['plans', 'pairs', 'gaps', 'optimal'])
PlanExecution = namedtuple('PlanExecution', ['plan', 'transitions', 'pairs', 'steps'])


def _walk_config(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            for nested in _walk_config(item):
                yield nested
    else:
        yield value


def _freeze_config(value):
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config(item) for item in value)
    return value


def _flatten_config(value):
    return tuple(_walk_config(value))


def _nested_leaves(value):
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_nested_leaves(item))
        return tuple(result)
    return value,


def _is_auto_trigger(machine, trigger):
    if not trigger.startswith('to_') or trigger == 'to':
        return False
    target = trigger[3:]
    if machine.model_attribute != 'state':
        suffix = '_' + machine.model_attribute
        if target.endswith(suffix):
            target = target[:-len(suffix)]
    try:
        machine.get_state(target)
    except (KeyError, ValueError):
        return False
    return True


def _iter_transitions(machine):
    for event in machine.events.values():
        for transitions in event.transitions.values():
            for transition in transitions:
                yield event.name, transition


def _core_candidates(machine, config, trigger):
    event = machine.events.get(trigger)
    if event is None:
        return ()
    return tuple(event.transitions.get(config, ()))


def _nested_candidates(machine, config, trigger):
    candidates = []
    seen = set()
    for leaf in _nested_leaves(config):
        for transition in machine.get_transitions(trigger=trigger, source=leaf, delegate=True):
            if id(transition) not in seen:
                seen.add(id(transition))
                candidates.append(transition)
    return tuple(candidates)


def _candidates(machine, config, trigger):
    if isinstance(machine, HierarchicalMachine):
        return _nested_candidates(machine, config, trigger)
    return _core_candidates(machine, _flatten_config(config)[0], trigger)


def _probe_model(machine):
    if machine.models:
        source = machine.models[0]
        try:
            return copy.copy(source)
        except TypeError:
            probe = type('PlanningProbe', (object,), {})()
            probe.__dict__.update(getattr(source, '__dict__', {}))
            return probe
    probe = type('PlanningProbe', (object,), {})()
    setattr(probe, machine.model_attribute, None)
    return probe


def _set_probe_state(machine, probe, config):
    if isinstance(machine, HierarchicalMachine):
        machine.set_state(config, probe)
    else:
        setattr(probe, machine.model_attribute, config)


def _probe_state(machine, probe):
    return _freeze_config(getattr(probe, machine.model_attribute))


def _trigger(machine, probe, trigger):
    if isinstance(machine, HierarchicalMachine):
        return machine.trigger_event(probe, trigger)
    return machine.events[trigger].trigger(probe)


def _record_trigger(machine, probe, trigger):
    recorded = []
    originals = {}

    def recorder(transition):
        def callback(*args, **kwargs):
            recorded.append(transition)
        return callback

    for _, transition in _iter_transitions(machine):
        originals[id(transition)] = (transition, list(transition.after))
        transition.after.append(recorder(transition))
    try:
        result = _trigger(machine, probe, trigger)
    finally:
        for transition, callbacks in originals.values():
            transition.after = callbacks
    if result is False:
        return None
    return tuple(recorded)


def _gap(kind, reason, config=None, trigger=None, transition=None):
    return CoverageGap(kind, reason, config, trigger, transition)


def _transition_cost(costs, trigger, transition, default_cost):
    if callable(costs):
        value = costs(trigger, transition)
    else:
        value = costs.get(transition)
        if value is None:
            value = costs.get((trigger, transition.source, transition.dest))
        if value is None:
            value = costs.get(trigger)
    if value is None:
        return default_cost
    if (not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0 or
            not math.isfinite(value)):
        raise ValueError('transition costs must be finite non-negative numbers')
    return value


def _reset_spec(reset, reset_cost):
    if reset is None:
        return None, reset_cost
    if isinstance(reset, string_types):
        return reset, reset_cost
    if isinstance(reset, (list, tuple)) and len(reset) == 2:
        trigger, cost = reset
        if (not isinstance(trigger, string_types) or not isinstance(cost, (int, float)) or
                isinstance(cost, bool) or cost < 0 or not math.isfinite(cost)):
            raise ValueError('reset must be an event name and a finite non-negative cost')
        return trigger, cost
    raise ValueError('reset must be None, an event name, or a (trigger, cost) pair')


def _initial_config(machine):
    if not isinstance(machine, HierarchicalMachine):
        return _freeze_config(machine.initial)
    initial = machine.initial
    if isinstance(initial, string_types):
        initial = machine._resolve_initial(_probe_model(machine), initial.split(machine.state_cls.separator))
    return _freeze_config(initial)


def _config_triggers(machine, config, include_auto_transitions):
    if isinstance(machine, HierarchicalMachine):
        triggers = set()
        with machine():
            for leaf in _nested_leaves(config):
                path = leaf.split(machine.state_cls.separator)
                while path:
                    triggers.update(machine.get_nested_triggers(path))
                    path.pop()
    else:
        triggers = set(machine.get_triggers(_flatten_config(config)[0]))
    if not include_auto_transitions:
        triggers = {trigger for trigger in triggers if not _is_auto_trigger(machine, trigger)}
    return triggers


def _explore(machine, initial, costs, default_cost, reset_trigger, reset_cost,
             include_auto_transitions):
    gaps = []
    gap_keys = set()
    edges = []
    edge_keys = {}
    pending = deque([initial])
    configs = {initial}

    def add_gap(kind, reason, config, trigger=None, transition=None):
        key = (kind, config, trigger, transition)
        if key not in gap_keys:
            gap_keys.add(key)
            gaps.append(_gap(kind, reason, config, trigger, transition))

    while pending:
        config = pending.popleft()
        for trigger in _config_triggers(machine, config, include_auto_transitions):
            candidates = _candidates(machine, config, trigger)
            deterministic = [transition for transition in candidates if not transition.conditions]
            for transition in candidates:
                if transition.conditions:
                    add_gap('conditional', 'Conditional transitions cannot be covered offline',
                            config, trigger, transition)
            if not deterministic:
                continue
            probe = _probe_model(machine)
            _set_probe_state(machine, probe, config)
            try:
                executed = _record_trigger(machine, probe, trigger)
            except Exception as error:
                add_gap('unsupported', str(error), config, trigger)
                continue
            if executed is None:
                add_gap('unsupported', 'Trigger did not execute a transition', config, trigger)
                continue
            if len(executed) != 1:
                add_gap('unsupported', 'Trigger executes multiple simultaneous transitions', config, trigger)
                continue
            transition = executed[0]
            if transition not in deterministic:
                add_gap('unsupported', 'Executed transition was not statically identifiable',
                        config, trigger, transition)
                continue
            target = _probe_state(machine, probe)
            reset = trigger == reset_trigger
            cost = reset_cost if reset else _transition_cost(costs, trigger, transition, default_cost)
            edge = Edge(config, target, trigger, transition, cost, reset)
            key = (config, target, trigger, transition)
            if key not in edge_keys:
                edge_keys[key] = len(edges)
                edges.append(edge)
            if target not in configs:
                configs.add(target)
                pending.append(target)
    return tuple(edges), tuple(gaps)


def _adjacency(edges):
    result = {}
    for index, edge in enumerate(edges):
        result.setdefault(edge.source, []).append(index)
    return result


def _route(edges, adjacency, source, last_index, pair):
    queue = [(0, 0, source, last_index, ())]
    best = {}
    tie_breaker = 0
    while queue:
        cost, _, config, previous_index, path = heapq.heappop(queue)
        state = (config, previous_index)
        if best.get(state) is not None and cost > best[state]:
            continue
        best[state] = cost
        for edge_index in adjacency.get(config, ()):
            edge = edges[edge_index]
            next_path = path + (edge_index,)
            next_cost = cost + edge.cost
            if previous_index is not None and not edges[previous_index].reset and \
                    edges[previous_index].transition is pair.source_transition and \
                    edge.transition is pair.target_transition and not edge.reset:
                return next_path, next_cost
            next_state = (edge.target, edge_index)
            if best.get(next_state) is None or next_cost < best[next_state]:
                tie_breaker += 1
                heapq.heappush(queue, (next_cost, tie_breaker, edge.target, edge_index, next_path))
    return None, None


def _pairs(edges):
    normal = [edge for edge in edges if not edge.reset]
    outgoing = {}
    for edge in normal:
        outgoing.setdefault(edge.source, []).append(edge)
    pairs = set()
    for first in normal:
        for second in outgoing.get(first.target, ()):
            pairs.add(TransitionPair(first.transition, second.transition))
    return tuple(sorted(pairs, key=lambda pair: (id(pair.source_transition), id(pair.target_transition))))


def _steps(path):
    return tuple(Step(edge.trigger, edge.transition, edge.cost, edge.reset) for edge in path)


def generate_transition_pair_plans(machine, costs=None, reset='reset', reset_cost=0,
                                   default_cost=1, include_auto_transitions=False):
    """Generate reachable event sequences covering adjacent unconditional transition pairs.

    Args:
        machine: A synchronous deterministic state machine.
        costs: Optional non-negative costs keyed by trigger, ``(trigger, source, dest)``,
            transition, or a callable receiving trigger and transition.
        reset: Event name of a real reset transition, a ``(event, cost)`` pair, or ``None``.
        reset_cost: Cost used when ``reset`` is provided as an event name.
        default_cost: Cost assigned to transitions without an explicit cost.
        include_auto_transitions: Include generated ``to_<state>`` events in planning.

    Returns:
        A test-plan set containing plans, required pairs, and coverage gaps. Reset steps remain
        executable transitions and are explicitly marked. Conditional and otherwise unsupported
        behavior is reported instead of being guessed offline. The search is deterministic and
        heuristic; ``optimal`` is always ``False`` because it does not guarantee a minimum result.
    """
    costs = {} if costs is None else costs
    reset_trigger, fixed_reset_cost = _reset_spec(reset, reset_cost)
    if (not isinstance(default_cost, (int, float)) or isinstance(default_cost, bool) or
            default_cost < 0 or not math.isfinite(default_cost)):
        raise ValueError('default_cost must be a finite non-negative number')
    if (not isinstance(fixed_reset_cost, (int, float)) or isinstance(fixed_reset_cost, bool) or
            fixed_reset_cost < 0 or not math.isfinite(fixed_reset_cost)):
        raise ValueError('reset cost must be a finite non-negative number')
    if not isinstance(machine, Machine):
        raise ValueError('machine must be a transitions Machine')

    gaps = []
    if isinstance(machine, AsyncMachine) or inspect.iscoroutinefunction(machine._process):
        gaps.append(_gap('unsupported', 'Asynchronous machines are not supported by the offline planner'))
        return TestPlanSet((), (), tuple(gaps), False)
    if getattr(machine, 'has_queue', False):
        gaps.append(_gap('unsupported', 'Queued machines are not supported by the offline planner'))
        return TestPlanSet((), (), tuple(gaps), False)

    initial = _initial_config(machine)
    edges, explore_gaps = _explore(machine, initial, costs, default_cost, reset_trigger,
                                   fixed_reset_cost, include_auto_transitions)
    gaps.extend(explore_gaps)
    required = _pairs(edges)
    adjacency = _adjacency(edges)
    plans = []
    uncovered = set(required)

    while uncovered:
        current = initial
        last_index = None
        path = ()
        while uncovered:
            route = None
            route_cost = None
            for pair in uncovered:
                candidate, candidate_cost = _route(edges, adjacency, current, last_index, pair)
                if candidate is not None and (route is None or candidate_cost < route_cost):
                    route = candidate
                    route_cost = candidate_cost
            if route is None:
                if not path:
                    break
                route_cost = None
                for pair in uncovered:
                    candidate, candidate_cost = _route(edges, adjacency, initial, None, pair)
                    if candidate is not None and (route is None or candidate_cost < route_cost):
                        route = candidate
                        route_cost = candidate_cost
                break
            for edge_index in route:
                if last_index is not None:
                    uncovered.discard(TransitionPair(edges[last_index].transition, edges[edge_index].transition))
                path += (edge_index,)
                current = edges[edge_index].target
                last_index = edge_index
        if not path:
            break
        plan_edges = tuple(edges[index] for index in path)
        plans.append(TestPlan(_steps(plan_edges), sum(edge.cost for edge in plan_edges)))
        covered_by_new_plan = {TransitionPair(plan_edges[index].transition, plan_edges[index + 1].transition)
                               for index in range(len(plan_edges) - 1)
                               if not plan_edges[index].reset and not plan_edges[index + 1].reset}
        uncovered.difference_update(covered_by_new_plan)

    return TestPlanSet(tuple(plans), required, tuple(gaps), False)


def execute_plan(machine, plan, model=None):
    """Execute a generated plan on a real machine and record adjacent transition identities.

    Every planned event must execute the transition selected by the planner. Pairs adjacent to a
    reset step are intentionally omitted because the reset separates normal test actions.
    """
    if isinstance(machine, AsyncMachine) or inspect.iscoroutinefunction(machine._process):
        raise ValueError('execute_plan does not support asynchronous machines')
    model = machine.models[0] if model is None and machine.models else model
    if model is None:
        raise ValueError('a model is required')
    machine.set_state(_initial_config(machine), model)
    recorded = []
    originals = {}

    def recorder(transition):
        def callback(*args, **kwargs):
            recorded.append(transition)
        return callback

    for _, transition in _iter_transitions(machine):
        originals[id(transition)] = (transition, list(transition.after))
        transition.after.append(recorder(transition))
    executed_steps = []
    try:
        for step in plan.steps:
            before = len(recorded)
            result = _trigger(machine, model, step.trigger)
            if result is not True or len(recorded) != before + 1 or recorded[-1] is not step.transition:
                raise AssertionError('planned transition %r was not executed by trigger %r'
                                     % (step.transition, step.trigger))
            executed_steps.append(step)
    finally:
        for transition, callbacks in originals.values():
            transition.after = callbacks
    pairs = tuple(TransitionPair(recorded[index], recorded[index + 1])
                  for index in range(len(recorded) - 1)
                  if not executed_steps[index].reset and not executed_steps[index + 1].reset)
    return PlanExecution(plan, tuple(recorded), pairs, tuple(executed_steps))


def execute_plans(machine, plans, model=None):
    return tuple(execute_plan(machine, plan, model) for plan in plans)


def verify_coverage(executions, required_pairs):
    covered = {pair for execution in executions for pair in execution.pairs}
    return required_pairs, set(required_pairs) - covered
