"""Offline test-plan generation for finite deterministic state machines.

The planner reasons from the machine's static graph. A continuation is either a
transition that can be triggered from the current active configuration or an
explicitly declared reset. It never starts again from an arbitrary state. The
router is a greedy heuristic and does not guarantee a minimum-cost plan.
"""

from collections import namedtuple
import inspect
from numbers import Real

from six import string_types


ACTION = "action"
RESET = "reset"


RawTransition = namedtuple("RawTransition", ["trigger", "source", "dest", "transition", "scope",
                                             "index", "wildcard"])
Edge = namedtuple("Edge", ["identity", "trigger", "source", "dest", "transition", "cost", "kind", "operation"])
CoverageGap = namedtuple("CoverageGap", ["kind", "trigger", "source", "dest", "message"])
PlanStep = namedtuple("PlanStep", ["kind", "trigger", "identity", "source", "dest", "cost",
                                   "operation", "transition"])
TestPlan = namedtuple("TestPlan", ["steps", "events", "cost", "required_pairs", "coverage_gaps",
                                   "initial_configuration", "optimal"])
PlanExecution = namedtuple("PlanExecution", ["plan", "steps", "transition_identities",
                                             "adjacent_identities", "covered_pairs", "missing_pairs",
                                             "extra_pairs", "initial_configuration", "final_configuration",
                                             "successful"])
ResetOperation = namedtuple("ResetOperation", ["trigger", "cost", "sources", "dest", "operation", "identity"])


def _is_nested(machine):
    return hasattr(machine, "build_state_tree")


def _is_async(machine):
    return inspect.iscoroutinefunction(getattr(machine, "_process_async", None))


def _separator(machine):
    state_cls = getattr(machine, "state_cls", None)
    return getattr(state_cls, "separator", "_") if _is_nested(machine) else "_"


def _state_name(state):
    value = getattr(state, "value", None)
    return value.name if hasattr(value, "name") else str(value)


def _state_exists(machine, path):
    states = machine.states
    for index, name in enumerate(path):
        state = states.get(name)
        if state is None:
            return False
        if index == len(path) - 1:
            return True
        child_states = getattr(state, "states", None)
        if child_states is None or not child_states:
            return False
        states = child_states
    return True


def _initial_names(state):
    initial = getattr(state, "initial", None)
    if initial is None:
        return []
    if isinstance(initial, (list, tuple)):
        return [name.name if hasattr(name, "name") else name for name in initial]
    return [initial.name if hasattr(initial, "name") else initial]


def _has_supported_initial(state, separator):
    if not getattr(state, "states", None):
        return True
    names = _initial_names(state)
    if len(names) != 1:
        return True
    return names[0] in state.states


def _close_path(machine, path, separator, gaps=None):
    states = machine.states
    resolved = []
    for index, name in enumerate(path):
        state = states.get(name)
        if state is None:
            return None
        resolved.append(name)
        child_states = getattr(state, "states", None)
        if index == len(path) - 1:
            while child_states:
                names = _initial_names(state)
                if len(names) > 1:
                    if gaps is not None:
                        _add_gap(gaps, CoverageGap(
                            "parallel_configuration", None, separator.join(resolved), None,
                            "Parallel active configurations are not supported by the finite planner."))
                    return None
                if not names:
                    break
                initial_name = names[0]
                if initial_name in child_states:
                    state = child_states[initial_name]
                    resolved.append(initial_name)
                    child_states = getattr(state, "states", None)
                else:
                    if gaps is not None:
                        _add_gap(gaps, CoverageGap(
                            "unsupported_initial", None, separator.join(resolved), None,
                            "Nested initial references to deeper descendants are not supported."))
                    return None
            return separator.join(resolved)
        if not child_states:
            return None
        states = child_states
    return None


def _add_gap(gaps, gap):
    key = (gap.kind, gap.trigger, gap.source, gap.dest, gap.message)
    if all(key != (item.kind, item.trigger, item.source, item.dest, item.message) for item in gaps):
        gaps.append(gap)


def _collect_configurations(machine, separator):
    gaps = []
    configurations = []

    def walk(states, scope):
        for state in states.values():
            local_name = _state_name(state)
            path = scope + [local_name]
            config = separator.join(path)
            child_states = getattr(state, "states", None)
            if child_states:
                if len(_initial_names(state)) > 1:
                    _add_gap(gaps, CoverageGap(
                        "parallel_configuration", None, config, None,
                        "Parallel active configurations are not supported by the finite planner."))
                    continue
                if not _has_supported_initial(state, separator):
                    _add_gap(gaps, CoverageGap(
                        "unsupported_initial", None, config, None,
                        "Nested initial references to deeper descendants are not supported."))
                    continue
                configurations.append(config)
                walk(child_states, path)
            else:
                configurations.append(config)

    if _is_nested(machine):
        walk(machine.states, [])
        initial = getattr(machine, "_initial", machine.initial)
        if isinstance(initial, (list, tuple)):
            if len(initial) > 1:
                _add_gap(gaps, CoverageGap(
                    "parallel_configuration", None,
                    separator.join([_state_name(state) for state in initial]), None,
                    "Parallel initial configurations are not supported by the finite planner."))
                initial_config = None
            else:
                initial_config = _close_path(machine, [_state_name(initial[0])], separator, gaps)
        else:
            name = initial.name if hasattr(initial, "name") else initial
            initial_config = _close_path(machine, [name], separator, gaps)
    else:
        configurations.extend(machine.states.keys())
        initial = getattr(machine, "_initial", machine.initial)
        initial_config = initial.name if hasattr(initial, "name") else initial

    return sorted(set(configurations)), initial_config, gaps


def _iter_nested_raw(machine, separator, include_auto_transitions):
    for event in machine.events.values():
        if not include_auto_transitions and event.name.startswith("to_"):
            continue
        for source, transitions in event.transitions.items():
            for transition in transitions:
                yield RawTransition(event.name, source, transition.dest, transition, [], -1,
                                    source == machine.wildcard_all)

    def walk(states, scope):
        for state in states.values():
            local_name = _state_name(state)
            path = scope + [local_name]
            child_states = getattr(state, "states", {})
            wildcard_states = [_state_name(child) for child in child_states.values()]
            for index, event in enumerate(state.events.values()):
                for source, transitions in event.transitions.items():
                    sources = wildcard_states if source == machine.wildcard_all else [source]
                    for source_name in sources:
                        for transition in transitions:
                            raw_source = separator.join(path + source_name.split(separator))
                            yield RawTransition(event.name, raw_source, transition.dest, transition, path,
                                                -1, source == machine.wildcard_all)
            walk(child_states, path)

    for raw in walk(machine.states, []):
        if include_auto_transitions or not raw.trigger.startswith("to_"):
            yield raw


def _collect_raw(machine, separator, include_auto_transitions):
    if _is_nested(machine):
        return list(_iter_nested_raw(machine, separator, include_auto_transitions))

    raw = []
    index = 0
    for event in machine.events.values():
        if not include_auto_transitions and event.name.startswith("to_"):
            continue
        for source, transitions in event.transitions.items():
            for transition in transitions:
                raw.append(RawTransition(event.name, source, transition.dest, transition, [], index,
                                         source == machine.wildcard_all))
                index += 1
    return raw


def _resolve_destination(machine, raw, source_config, separator, gaps):
    if raw.dest is None:
        return source_config

    destination = raw.dest.name if hasattr(raw.dest, "name") else raw.dest
    destination_path = destination.split(separator) if _is_nested(machine) else [destination]
    candidates = []
    if _is_nested(machine):
        candidates.append(raw.scope + destination_path)
        if not raw.scope and _state_exists(machine, destination_path):
            candidates.append(destination_path)
    else:
        candidates.append(destination_path)

    for path in candidates:
        if _state_exists(machine, path):
            resolved = _close_path(machine, path, separator, gaps)
            if resolved is not None:
                return resolved

    _add_gap(gaps, CoverageGap(
        "unresolved_destination", raw.trigger, raw.source, destination,
        "Transition destination cannot be resolved in a supported finite configuration."))
    return None


def _numeric_cost(value, message):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(message)
    if value < 0:
        raise ValueError(message)
    return value


def _transition_cost(transition, raw, costs):
    declared_dest = raw.dest if raw.dest is not None else raw.source
    keys = (
        id(transition),
        transition,
        (raw.trigger, raw.source, declared_dest),
        (raw.trigger, raw.source),
        raw.trigger,
    )
    for key in keys:
        if key in costs:
            return _numeric_cost(costs[key], "Transition costs must be non-negative numbers.")
    cost = getattr(transition, "cost", None)
    if cost is not None:
        return _numeric_cost(cost, "Transition costs must be non-negative numbers.")
    return 0


def _identity(raw):
    dest = raw.dest if raw.dest is not None else raw.source
    dest = dest.name if hasattr(dest, "name") else dest
    return raw.trigger, raw.source, dest, id(raw.transition)


def _ground_edges(machine, raw_transitions, configurations, initial_config, separator, costs,
                  reset_operations, gaps):
    reset_triggers = set(operation.trigger for operation in reset_operations if operation.trigger)
    raw_by_trigger = {}
    for index, raw in enumerate(raw_transitions):
        raw = raw._replace(index=index)
        raw_by_trigger.setdefault(raw.trigger, []).append(raw)
        if raw.transition.conditions:
            _add_gap(gaps, CoverageGap(
                "conditional_transition", raw.trigger, raw.source,
                raw.dest.name if hasattr(raw.dest, "name") else raw.dest,
                "Conditional transitions cannot be planned without executing model predicates."))

    edges = []
    seen = set()
    config_paths = dict((config, config.split(separator)) for config in configurations)

    def add_edge(raw, source_config, dest_config, cost, kind, operation):
        identity = _identity(raw)
        operation_key = operation.identity if operation is not None else None
        key = (identity, source_config, dest_config, kind, operation_key)
        if key in seen:
            return
        seen.add(key)
        edges.append(Edge(identity, raw.trigger, source_config, dest_config,
                          raw.transition, cost, kind, operation))

    for config in configurations:
        config_path = config_paths[config]
        for trigger, raws in raw_by_trigger.items():
            operations = [op for op in reset_operations if op.trigger == trigger]
            is_reset = bool(operations)
            if not is_reset and trigger in reset_triggers:
                continue

            candidates = []
            for raw in raws:
                raw_source_path = raw.source.split(separator)
                exact_match = config_path == raw_source_path
                wildcard_match = raw.wildcard and len(config_path) == len(raw_source_path)
                delegation_match = config_path[:len(raw_source_path)] == raw_source_path
                if not exact_match and not wildcard_match and not delegation_match:
                    continue
                if not _state_exists(machine, raw_source_path):
                    continue
                dest_config = _resolve_destination(machine, raw, config, separator, gaps)
                if dest_config is None:
                    continue
                candidates.append((raw, dest_config))

            candidates.sort(key=lambda item: (-len(item[0].source.split(separator)), item[0].index))
            if candidates:
                depth = len(candidates[0][0].source.split(separator))
                same_depth = [item for item in candidates if len(item[0].source.split(separator)) == depth]
                unconditional = [item for item in same_depth if not item[0].transition.conditions]
                if not unconditional:
                    continue
                for position, item in enumerate(unconditional):
                    raw, dest_config = item
                    if position > 0:
                        _add_gap(gaps, CoverageGap(
                            "ambiguous_transition", raw.trigger, raw.source,
                            raw.dest.name if hasattr(raw.dest, "name") else raw.dest,
                            "Multiple unconditional transitions compete for the same trigger and configuration."))
                        continue
                    operation = operations[0] if operations else None
                    if is_reset and operation.sources and not _source_matches(operation, config, raw):
                        continue
                    if is_reset:
                        cost = operation.cost if operation.cost is not None else \
                            _transition_cost(raw.transition, raw, costs)
                    else:
                        cost = _transition_cost(raw.transition, raw, costs)
                    add_edge(raw, config, dest_config, cost, RESET if is_reset else ACTION, operation)

        for operation in reset_operations:
            if operation.trigger is not None:
                continue
            if operation.sources and not _external_source_matches(operation, config):
                continue
            requested_dest = operation.dest if operation.dest is not None else initial_config
            if requested_dest is None:
                dest_config = None
            elif _is_nested(machine):
                dest_config = _close_path(machine, requested_dest.split(separator), separator, gaps)
            else:
                dest_config = requested_dest
            if dest_config not in configurations:
                _add_gap(gaps, CoverageGap(
                    "unresolved_reset", None, config, dest_config,
                    "Reset destination is not a supported finite configuration."))
                continue
            raw = RawTransition(operation.identity, config, dest_config, None, [], -1, False)
            add_edge(raw, config, dest_config, operation.cost, RESET, operation)

    return _reachable_edges(initial_config, edges), edges


def _source_matches(operation, config, raw):
    sources = operation.sources
    if config in sources or raw.source in sources:
        return True
    for source in sources:
        source_name = source.name if hasattr(source, "name") else source
        if source_name == config:
            return True
        if source_name == raw.source:
            return True
    return False


def _external_source_matches(operation, config):
    for source in operation.sources:
        source_name = source.name if hasattr(source, "name") else source
        if source_name == config:
            return True
    return False


def _reachable_edges(initial_config, edges):
    if initial_config is None:
        return []
    outgoing = {}
    for edge in edges:
        outgoing.setdefault(edge.source, []).append(edge)

    reachable = set()
    queue = [initial_config]
    seen_configs = set()
    while queue:
        config = queue.pop(0)
        if config in seen_configs:
            continue
        seen_configs.add(config)
        for edge in outgoing.get(config, []):
            reachable.add(_edge_key(edge))
            if edge.dest not in seen_configs:
                queue.append(edge.dest)
    return [edge for edge in edges if _edge_key(edge) in reachable]


def _edge_key(edge):
    operation_key = edge.operation.identity if edge.operation is not None else None
    return (edge.identity, edge.source, edge.dest, edge.kind, operation_key)


def _listify(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalize_reset(reset, gaps):
    if reset is None:
        return []

    operations = list(reset) if isinstance(reset, (list, tuple)) else [reset]
    normalized = []
    for index, value in enumerate(operations):
        if isinstance(value, ResetOperation):
            operation = value
        elif isinstance(value, dict):
            operation = ResetOperation(
                value.get("trigger"),
                value.get("cost"),
                _listify(value.get("sources")),
                value.get("dest"),
                value.get("operation") or value.get("callable"),
                value.get("identity"))
        elif isinstance(value, string_types):
            operation = ResetOperation(value, None, [], None, None, value)
        elif callable(value):
            operation = ResetOperation(None, None, [], None, value,
                                       getattr(value, "__name__", "reset_%d" % index))
        elif isinstance(value, (list, tuple)):
            trigger = value[0] if len(value) > 0 else None
            cost = value[1] if len(value) > 1 else None
            operation = ResetOperation(trigger, cost, [], None, None, trigger)
        else:
            raise ValueError("Reset operations must be strings, callables, mappings or ResetOperation values.")

        if operation.trigger is None and operation.operation is None:
            raise ValueError("A reset operation must provide a trigger or callable.")
        cost = 0 if operation.cost is None else operation.cost
        cost = _numeric_cost(cost, "Reset costs must be non-negative numbers.")
        identity = operation.identity or operation.trigger or getattr(operation.operation, "__name__", "reset")
        normalized.append(operation._replace(cost=cost, identity=identity))

    triggers = [operation.trigger for operation in normalized if operation.trigger]
    if len(triggers) != len(set(triggers)):
        raise ValueError("Each reset trigger can only be configured once.")
    return normalized


def _required_pairs(edges):
    outgoing = {}
    for edge in edges:
        outgoing.setdefault(edge.source, set()).add(edge.identity)

    pairs = set()
    for edge in edges:
        for following in outgoing.get(edge.dest, set()):
            pairs.add((edge.identity, following))
    return pairs, outgoing


def _edge_path_key(path):
    return tuple((edge.kind, edge.trigger, edge.source, edge.dest) for edge in path)


def _route_for_concrete_pair(edges, current_config, first, second):
    route = _route_to_edge(edges, current_config, first)
    if route is None:
        return None
    cost, path = route
    if not path or path[-1].dest != second.source:
        return None
    return cost + second.cost, path + (second,)


def _route_to_edge(edges, source, target):
    outgoing = {}
    for edge in edges:
        outgoing.setdefault(edge.source, []).append(edge)

    queue = [(0, (source, None), ())]
    seen = {(source, None): 0}
    while queue:
        queue.sort(key=lambda item: (item[0], str(item[1]), _edge_path_key(item[2])))
        cost, state, path = queue.pop(0)
        config = state[0]
        if path and path[-1] == target:
            return cost, path
        for edge in outgoing.get(config, []):
            next_cost = cost + edge.cost
            next_path = path + (edge,)
            state_key = (edge.dest, _edge_key(edge))
            if next_cost < seen.get(state_key, float("inf")):
                seen[state_key] = next_cost
                queue.append((next_cost, state_key, next_path))
    return None


def _new_pair_count(path, covered_pairs, previous_edge=None):
    pair_keys = _segment_pair_keys(path)
    if previous_edge is not None and path:
        pair_keys.add(_edge_pair_key(previous_edge, path[0]))
    return len(pair_keys - covered_pairs)


def _segment_pair_keys(path):
    return set(_edge_pair_key(edge, path[index + 1]) for index, edge in enumerate(path[:-1]))


def _edge_pair_key(first, second):
    return (_edge_key(first), _edge_key(second))


def _build_plan(edges, initial_config, required_pairs, gaps):
    steps = []
    covered = set()
    remaining = {}
    for first in edges:
        for second in edges:
            if (first.identity, second.identity) in required_pairs and first.dest == second.source:
                remaining[_edge_pair_key(first, second)] = (first, second)
    current_config = initial_config
    previous_edge = None

    while remaining:
        direct = [(first, second) for first, second in remaining.values()
                  if first == previous_edge]
        if direct:
            first, second = min(direct, key=lambda item: (item[1].cost,
                                                          item[1].kind,
                                                          item[1].identity,
                                                          item[1].source))
            path = (second,)
        else:
            routes = []
            for first, second in remaining.values():
                route = _route_for_concrete_pair(edges, current_config, first, second)
                if route is not None:
                    routes.append((route, first, second))

            if not routes:
                ordered = sorted(remaining.values(),
                                 key=lambda item: (item[0].identity, item[1].identity))
                for first, second in ordered:
                    _add_gap(gaps, CoverageGap(
                        "unreachable_transition_pair", None, None, None,
                        "Transition pair cannot be reached without restarting from an unsupported state: "
                        "%s -> %s" % (first.identity, second.identity)))
                break

            routes.sort(key=lambda item: (-_new_pair_count(item[0][1], covered, previous_edge),
                                          item[0][0],
                                          _edge_path_key(item[0][1]),
                                          item[1].identity, item[2].identity))
            route, first, second = routes[0]
            path = route[1]

        for edge in path:
            steps.append(PlanStep(edge.kind, edge.trigger, edge.identity, edge.source, edge.dest,
                                  edge.cost, edge.operation, edge.transition))
            current_config = edge.dest

        new_pairs = [(path[index], path[index + 1]) for index in range(len(path) - 1)]
        if previous_edge is not None and path:
            new_pairs.insert(0, (previous_edge, path[0]))
        covered.update(_edge_pair_key(first, second) for first, second in new_pairs)
        for first, second in new_pairs:
            remaining.pop(_edge_pair_key(first, second), None)
        if path:
            previous_edge = path[-1]

    events = [step.trigger for step in steps]
    cost = sum(step.cost for step in steps)
    return TestPlan(steps, events, cost, required_pairs, list(gaps), initial_config, False)


def generate_transition_pair_test_plans(machine, costs=None, reset=None,
                                        include_auto_transitions=False):
    """Generate one offline plan covering reachable consecutive transition pairs.

    Args:
        machine: A finite, deterministic core or hierarchical Machine.
        costs: Optional costs keyed by trigger, ``(trigger, source)``,
            ``(trigger, source, dest)``, transition object, or transition id.
        reset: Optional trigger, callable, mapping, ResetOperation or list.
        include_auto_transitions: Include generated ``to_*`` convenience events.

    Returns:
        list: A list containing one :class:`TestPlan`. Coverage gaps are reported
        on the plan when conditional, parallel or unsupported behaviour is found.
    """
    costs = costs or {}
    separator = _separator(machine)
    configurations, initial_config, gaps = _collect_configurations(machine, separator)

    if getattr(machine, "_queued", False):
        _add_gap(gaps, CoverageGap(
            "unsupported_machine", None, None, None,
            "Queued machines cannot be planned because trigger results are deferred."))
    if _is_async(machine):
        _add_gap(gaps, CoverageGap(
            "unsupported_machine", None, None, None,
            "Asynchronous machines cannot be planned by the synchronous offline planner."))

    reset_operations = _normalize_reset(reset, gaps)
    raw = [] if getattr(machine, "_queued", False) or _is_async(machine) else \
        _collect_raw(machine, separator, include_auto_transitions)
    reachable_edges, _ = _ground_edges(machine, raw, configurations, initial_config, separator,
                                       costs, reset_operations, gaps)
    grounded_reset_identities = set(edge.operation.identity for edge in reachable_edges
                                    if edge.kind == RESET and edge.operation is not None)
    for operation in reset_operations:
        if operation.identity not in grounded_reset_identities:
            _add_gap(gaps, CoverageGap(
                "unresolved_reset", operation.trigger, None, operation.dest,
                "Reset operation cannot be executed from a reachable configuration."))
    required_pairs, _ = _required_pairs(reachable_edges)
    return [_build_plan(reachable_edges, initial_config, required_pairs, gaps)]


def execute_test_plan(plan, machine=None, model=None):
    """Execute a plan on a real machine and verify required adjacent identities."""
    if machine is None:
        raise ValueError("A Machine instance is required to execute a test plan.")
    if model is None:
        model = machine.model if getattr(machine, "models", None) else machine
    if isinstance(model, list):
        model = model[0]

    machine.set_state(plan.initial_configuration, model)

    recorded = []
    executed_steps = []

    installed = []
    try:
        for edge in plan.steps:
            if edge.operation is not None and edge.kind == RESET and edge.transition is None:
                operation = edge.operation.operation
                operation()
                machine.set_state(edge.dest, model)
                recorded.append(edge.identity)
                executed_steps.append(edge)
                continue

            before = len(recorded)
            transition = edge.transition

            def recorder(transition=transition, edge=edge):
                recorded.append(edge.identity)

            transition.after.append(recorder)
            installed.append((transition, recorder))

            getattr(model, edge.trigger)()
            if len(recorded) != before + 1:
                raise RuntimeError("Planned trigger '%s' did not execute exactly one transition." %
                                   edge.trigger)
            current_config = getattr(model, machine.model_attribute)
            expected = edge.dest
            if current_config != expected:
                raise RuntimeError("Planned trigger '%s' reached %s, expected %s." %
                                   (edge.trigger, current_config, expected))
            executed_steps.append(edge._replace(identity=recorded[-1]))
            transition.after.remove(recorder)
            installed.pop()
    finally:
        for transition, recorder in installed:
            if recorder in transition.after:
                transition.after.remove(recorder)

    adjacent = list(zip(recorded, recorded[1:]))
    covered_pairs = set(
        (recorded[index], recorded[index + 1])
        for index in range(len(recorded) - 1)
    )
    missing_pairs = set(plan.required_pairs) - covered_pairs
    extra_pairs = covered_pairs - set(plan.required_pairs)
    successful = not missing_pairs and not extra_pairs

    return PlanExecution(
        plan=plan,
        steps=executed_steps,
        transition_identities=recorded,
        adjacent_identities=adjacent,
        covered_pairs=covered_pairs,
        missing_pairs=missing_pairs,
        extra_pairs=extra_pairs,
        initial_configuration=plan.initial_configuration,
        final_configuration=getattr(model, machine.model_attribute),
        successful=successful)


def generate_and_execute_transition_pair_test_plan(machine, costs=None, reset=None,
                                                   include_auto_transitions=False):
    plans = generate_transition_pair_test_plans(
        machine, costs=costs, reset=reset, include_auto_transitions=include_auto_transitions)
    return [execute_test_plan(plan, machine=machine) for plan in plans]
