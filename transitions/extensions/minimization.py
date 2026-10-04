"""
    transitions.extensions.minimization
    ------------------------------------

    This module contains a deterministic finite state machine minimizer. It computes a bisimulation
    quotient of a :class:`transitions.Machine`: states that share the same observable state label and
    behave identically for every event of a fixed alphabet are merged into one representative state.

    The first version deliberately limits the accepted input:

    * only unconditional, deterministic transitions are analysed, meaning that every ``(state, event)``
      pair may have at most one candidate transition without ``conditions``/``unless`` predicates;
    * the event alphabet is fixed and passed explicitly by the caller;
    * observable state labels are passed explicitly (a mapping or a callable);
    * enter, exit and transition callbacks are never executed. They participate in the comparison
      through stable identifiers returned by a caller-provided ``action_key`` function.

    A missing transition is *not* treated like a self loop. It has a uniform rejection semantics instead:
    processing an event without a transition leads to an implicit, non-accepting rejection sink that
    rejects every further event. The sink can optionally be materialized as an explicit state of the
    quotient machine.
"""

try:
    from builtins import object
except ImportError:  # pragma: no cover
    # python2
    pass

from collections import OrderedDict, deque

from six import string_types

from ..core import Machine

#: Observation emitted instead of a state label once an event without a transition has been processed.
#: All further observations of a rejected run are equal to this sentinel.
REJECT = '__reject__'


class MinimizationError(ValueError):
    """Raised when a machine cannot be minimized by this deterministic minimizer.

    This happens when the machine contains conditional or otherwise non-deterministic transitions,
    nested states, events outside of the provided alphabet, callbacks without a stable identifier or
    transitions to unknown states.
    """


class _Edge(object):
    """An internal, fully resolved transition edge.

    Attributes:
        kind (str): ``'external'`` for a transition that changes the state (including self loops),
            ``'internal'`` for a transition with ``dest is None`` and ``'missing'`` for an event
            without any transition.
        dest (str or None): Name of the destination state or ``None`` for internal/missing edges.
        actions (tuple): Stable identifiers of the observable action sequence of the edge, namely
            ``prepare``, ``before``, source ``on_exit``, destination ``on_enter`` and ``after`` for
            external edges. Internal edges do not run exit/enter callbacks.
    """

    def __init__(self, kind, dest, actions):
        self.kind = kind
        self.dest = dest
        self.actions = tuple(actions)

    def __repr__(self):
        return "<_Edge('%s', '%s')@%s>" % (self.kind, self.dest, id(self))


def _default_action_key(action):
    """Return the stable identifier of an action that has been declared by name.

    String callbacks reference callables on a model and therefore describe a stable, declaration-level
    behaviour. Raw callables do not carry a comparable identity and force the caller to provide an
    explicit ``action_key`` function.
    """
    if isinstance(action, string_types):
        return action
    raise MinimizationError(
        "Callback %r has no stable identifier. Pass an 'action_key' function that maps every "
        "enter/exit/transition callback to a comparable identifier. Callbacks are never executed "
        "during minimization." % (action,))


def _identifiers(callbacks, action_key):
    """Map a list of callbacks to the tuple of their stable identifiers."""
    return tuple(action_key(callback) for callback in callbacks)


def _resolve_label(labels, state_name):
    """Look up the observable label of a state in a mapping or by calling a callable."""
    if callable(labels):
        return labels(state_name)
    try:
        return labels[state_name]
    except KeyError:
        raise MinimizationError("No observable label provided for state '%s'." % state_name)


class _View(object):
    """An immutable, normalized view of either the original machine or of a quotient.

    The view adds a virtual rejection sink (``None``) so that missing transitions and self loops can
    never be confused.
    """

    def __init__(self, states, initial, events, edges, bases):
        self.states = states
        self.initial = initial
        self.events = events
        self.edges = edges
        self.bases = bases

    @staticmethod
    def _reject_edge():
        return _Edge('missing', None, ())

    def edge(self, state, event):
        """Return the resolved edge for ``(state, event)``; missing edges lead to the sink."""
        if state is None:
            return self._reject_edge()
        edge = self.edges[state].get(event)
        return edge if edge is not None else self._reject_edge()


class MinimizationResult(object):
    """The result of :func:`minimize_machine`.

    Attributes:
        machine (Machine): The minimized (quotient) machine. State names are the names of the
            representative states of the original machine.
        partition (OrderedDict): Maps every representative state name to the tuple of original state
            names it represents, ordered by the original state declaration order.
        mapping (OrderedDict): Maps every original state name to the name of its representative state.
        events (list): The event alphabet the minimization was based on.
        labels (dict): Observable labels of the quotient states (and of the rejection sink if present).
    """

    def __init__(self, machine, partition, mapping, events, labels, view, quotient_view):
        self.machine = machine
        self.partition = partition
        self.mapping = mapping
        self.events = events
        self.labels = labels
        self._view = view
        self._quotient_view = quotient_view

    # -- observation semantics ---------------------------------------------------------------

    def _trace(self, view, state, word):
        """Compute the observable sequence of processing ``word`` starting from ``state``."""
        trace = [REJECT if state is None else view.bases[state][0]]
        for event in word:
            if event not in view.events:
                raise MinimizationError("Event '%s' is not part of the alphabet." % event)
            edge = view.edge(state, event)
            if edge.kind == 'missing':
                state = None
                trace.append((tuple(edge.actions), REJECT))
            else:
                if edge.kind == 'external':
                    state = edge.dest
                label = REJECT if state is None else view.bases[state][0]
                trace.append((tuple(edge.actions), label))
        return trace

    def observable_trace(self, state, word):
        """Return the observable sequence emitted by the *original* machine.

        The first element is the label of the start state. Every subsequent element is a pair of the
        stable action identifiers emitted while processing the event and the label of the reached
        state. Once an event without a transition has been processed, the run stays in the rejection
        sink and every following observation is ``((), REJECT)``.

        Args:
            state (str): Name of the original start state.
            word (list or tuple): Sequence of event names.
        """
        if state is None:
            return self._trace(self._view, None, word)
        if state not in self._view.states:
            raise MinimizationError("Unknown state '%s'." % state)
        return self._trace(self._view, state, list(word))

    def quotient_trace(self, state, word):
        """Return the observable sequence of the quotient machine for the same input."""
        if state is None:
            return self._trace(self._quotient_view, None, word)
        representative = self.mapping[state]
        return self._trace(self._quotient_view, representative, list(word))

    def check_observations(self, max_length):
        """Enumerate every event string up to ``max_length`` from every state.

        Compares the observable sequences of the original machine and of the quotient machine,
        including runs that enter the rejection sink. Note that the number of event strings grows
        exponentially, so this check is intended for small machines only.

        Returns:
            list: A list of ``(start_state, word, original_trace, quotient_trace)`` tuples for every
            pair that differs. An empty list proves observational equivalence within the bound.
        """
        mismatches = []
        words = [()]
        for _ in range(max_length):
            for state in list(self._view.states) + [None]:
                for word in words:
                    original = self._trace(self._view, state, list(word))
                    quotient = self._trace(self._quotient_view,
                                           None if state is None else self.mapping[state], list(word))
                    if original != quotient:
                        mismatches.append((state, list(word), original, quotient))
            words = [word + (event,) for word in words for event in self.events]
        return mismatches

    # -- distinguishing words ---------------------------------------------------------------

    def distinguish(self, state_a, state_b):
        """Return a shortest event string that distinguishes two original states.

        Args:
            state_a (str): Name of the first state; ``None`` denotes the rejection sink.
            state_b (str): Name of the second state; ``None`` denotes the rejection sink.

        Returns:
            list or None: The shortest list of event names for which the two states emit different
            observable sequences, or ``None`` if the states are equivalent (including the empty word
            when their labels already differ).
        """
        if state_a is not None and state_a not in self._view.states:
            raise MinimizationError("Unknown state '%s'." % state_a)
        if state_b is not None and state_b not in self._view.states:
            raise MinimizationError("Unknown state '%s'." % state_b)
        if state_a == state_b:
            return None

        def signature(pair_state):
            if pair_state is None:
                return (REJECT,)
            return self._view.bases[pair_state]

        start = (state_a, state_b)
        if signature(state_a) != signature(state_b):
            return []

        queue = deque([start])
        parents = {start: None}

        def step_observation(pair_state, edge):
            if edge.kind == 'missing':
                return None
            if edge.kind == 'internal':
                target = pair_state
            else:
                target = edge.dest
            return edge.actions, signature(target)

        while queue:
            current = queue.popleft()
            for event in self.events:
                edge_a = self._view.edge(current[0], event)
                edge_b = self._view.edge(current[1], event)
                next_a = None if edge_a.kind == 'missing' else edge_a.dest
                next_b = None if edge_b.kind == 'missing' else edge_b.dest
                if edge_a.kind == 'internal':
                    next_a = current[0]
                if edge_b.kind == 'internal':
                    next_b = current[1]
                observation_a = step_observation(current[0], edge_a)
                observation_b = step_observation(current[1], edge_b)
                if observation_a != observation_b:
                    return self._reconstruct(parents, current) + [event]
                successor = self._normalize_pair(next_a, next_b)
                if successor not in parents:
                    parents[successor] = (current, event)
                    queue.append(successor)
        return None

    @staticmethod
    def _normalize_pair(first, second):
        """Canonicalize an unordered state pair, treating the rejection sink as the smallest element."""
        key = lambda item: (1, item) if item is not None else (0, '')
        return tuple(sorted((first, second), key=key))

    @staticmethod
    def _reconstruct(parents, pair):
        """Rebuild the event list leading from the BFS start pair to ``pair``."""
        word = []
        current = pair
        while parents[current] is not None:
            current, event = parents[current]
            word.append(event)
        word.reverse()
        return word


def minimize_machine(machine, events, labels=None, action_key=None, reject_state='rejected',
                     include_reject_state=None):
    """Minimize a deterministic :class:`transitions.Machine`.

    States are merged when they have the same observable label, the same enter/exit action identifiers
    and the same successor behaviour (including transition action identifiers) for every event of the
    fixed alphabet. Actions are compared by identifier and never executed.

    Args:
        machine (Machine): The machine to minimize. It must be a flat, deterministic machine without
            conditional transitions.
        events (str or list): The fixed event alphabet. A single string is treated as one event.
        labels (dict or callable, optional): Explicit observable state labels. Either a mapping of
            state name to a hashable label or a callable ``state_name -> label``. When omitted, all
            states share the same label and only the action/transition behaviour distinguishes states.
        action_key (callable, optional): Maps every enter/exit/transition callback to a hashable
            stable identifier. Callbacks are never called. By default only string callback names are
            accepted and raw callables cause a :class:`MinimizationError`.
        reject_state (str or bool, optional): Name of the state that materializes the uniform rejection
            sink in the quotient. Set to ``False`` to keep the quotient partial (missing transitions
            still reject during observation). Defaults to ``'rejected'``.
        include_reject_state: Deprecated alias kept for clarity; if explicitly set it overrides
            ``reject_state`` (``False`` disables the explicit reject state).

    Returns:
        MinimizationResult: The quotient machine, the original-to-representative state mapping and
        helpers for observation enumeration and distinguishing words.

    Raises:
        MinimizationError: If the machine is outside the supported fragment.
    """
    if action_key is None:
        action_key = _default_action_key

    if include_reject_state is not None:
        reject_state = 'rejected' if include_reject_state else False

    alphabet = _prepare_alphabet(machine, events)

    if getattr(machine, 'ignore_invalid_triggers', None):
        raise MinimizationError(
            "Machines or states with 'ignore_invalid_triggers' cannot be minimized because a missing "
            "event must keep its rejection semantics instead of being silently ignored.")

    state_order = list(machine.states.keys())
    if machine.initial is None or machine.initial not in machine.states:
        raise MinimizationError("The machine must have an initial state.")

    def label_of(state_name):
        if labels is None:
            return None
        return _resolve_label(labels, state_name)

    state_info = OrderedDict()
    for state_name in state_order:
        state = machine.states[state_name]
        _ensure_flat_state(state, state_name)
        if getattr(state, 'ignore_invalid_triggers', None):
            raise MinimizationError(
                "State '%s' ignores invalid triggers; missing transitions must keep their rejection "
                "semantics." % state_name)
        enter_ids = _identifiers(state.on_enter, action_key)
        exit_ids = _identifiers(state.on_exit, action_key)
        state_info[state_name] = {
            'state': state,
            'label': label_of(state_name),
            'enter': enter_ids,
            'exit': exit_ids,
            'final': bool(getattr(state, 'final', False)),
        }

    edges, raw_transitions = _build_edges(machine, state_order, alphabet, action_key, state_info)
    base_signatures = OrderedDict((name, (state_info[name]['label'],)) for name in state_order)
    view = _View(state_order, machine.initial, alphabet, edges, base_signatures)

    # Partition refinement (Moore-style bisimulation over the fixed alphabet).
    block_of = OrderedDict()
    for state_name in state_order:
        block_of[state_name] = base_signatures[state_name]
    block_ids = _number_blocks(block_of)

    while True:
        new_block_of = OrderedDict()
        for state_name in state_order:
            state_edges = edges[state_name]
            signature = [base_signatures[state_name]]
            for event in alphabet:
                edge = state_edges.get(event)
                if edge is None:
                    signature.append(('missing',))
                else:
                    if edge.kind == 'internal':
                        target_block = block_ids[state_name]
                    else:
                        target_block = block_ids[edge.dest]
                    signature.append((edge.actions, target_block))
            new_block_of[state_name] = tuple(signature)
        new_block_ids = _number_blocks(new_block_of)
        if new_block_ids == block_ids:
            break
        block_of = new_block_of
        block_ids = new_block_ids

    representatives = OrderedDict()
    for state_name in state_order:
        block_id = block_ids[state_name]
        if block_id not in representatives:
            representatives[block_id] = state_name

    mapping = OrderedDict((name, representatives[block_ids[name]]) for name in state_order)
    partition = OrderedDict()
    for state_name in state_order:
        representative = mapping[state_name]
        partition.setdefault(representative, []).append(state_name)
    partition = OrderedDict((representative, tuple(members))
                            for representative, members in partition.items())

    needs_reject = any(edges[state_name].get(event) is None
                      for state_name in state_order for event in alphabet)
    quotient, quotient_labels = _build_quotient(
        machine, mapping, partition, alphabet, edges, state_info, raw_transitions,
        reject_state, needs_reject)

    quotient_view = _build_quotient_view(machine, mapping, partition, alphabet, edges, base_signatures)

    return MinimizationResult(quotient, partition, mapping, alphabet, quotient_labels,
                              view, quotient_view)


def _prepare_alphabet(machine, events):
    """Validate and order the fixed event alphabet."""
    if isinstance(events, string_types):
        alphabet = [events]
    else:
        alphabet = list(events)
    if not alphabet:
        raise MinimizationError("The event alphabet must not be empty.")
    seen = set()
    for event in alphabet:
        if not isinstance(event, string_types):
            raise MinimizationError("Event names must be strings, got %r." % (event,))
        if event in seen:
            raise MinimizationError("Event '%s' is listed more than once in the alphabet." % event)
        seen.add(event)

    auto_prefix = 'to_' if machine.model_attribute == 'state' else 'to_%s_' % machine.model_attribute
    for trigger in machine.events:
        if trigger in seen:
            continue
        if machine.auto_transitions and trigger.startswith(auto_prefix):
            # Auto-generated to_<state> triggers are convenience transitions, not protocol events.
            continue
        raise MinimizationError(
            "Trigger '%s' is not part of the provided event alphabet. Every observable event must be "
            "listed explicitly." % trigger)
    return alphabet


def _ensure_flat_state(state, state_name):
    """Reject states that belong to a hierarchical machine."""
    substates = getattr(state, 'states', None)
    if substates:
        raise MinimizationError(
            "State '%s' is a nested state. Minimization of hierarchical machines is not supported in "
            "the first version." % state_name)
    if getattr(state, 'parent', None) is not None:
        raise MinimizationError(
            "State '%s' has a parent state. Minimization of hierarchical machines is not supported in "
            "the first version." % state_name)


def _number_blocks(block_of):
    """Assign a stable integer id to every distinct partition signature."""
    ids = {}
    result = {}
    counter = 0
    try:
        for state_name, signature in block_of.items():
            if signature not in ids:
                ids[signature] = counter
                counter += 1
            result[state_name] = ids[signature]
    except TypeError:
        raise MinimizationError("Observable labels and action identifiers must be hashable.")
    return result


def _build_edges(machine, state_order, alphabet, action_key, state_info):
    """Resolve the single deterministic edge of every ``(state, event)`` pair.

    Events without a transition are left absent; they are interpreted later as edges into the
    rejection sink and therefore stay distinguishable from internal transitions and self loops.
    """
    edges = OrderedDict((state_name, OrderedDict()) for state_name in state_order)
    raw_transitions = {}
    for state_name in state_order:
        exit_ids = state_info[state_name]['exit']
        for event in alphabet:
            candidates = list(machine.events[event].transitions[state_name]) \
                if event in machine.events else []
            if not candidates:
                continue
            if len(candidates) > 1:
                raise MinimizationError(
                    "State '%s' has %d candidate transitions for event '%s'. Non-deterministic or "
                    "conditional branching cannot be minimized in the first version."
                    % (state_name, len(candidates), event))
            transition = candidates[0]
            if transition.conditions:
                raise MinimizationError(
                    "The transition of event '%s' from state '%s' has conditions. Conditional "
                    "branching cannot be minimized in the first version." % (event, state_name))
            prepare_ids = _identifiers(transition.prepare, action_key)
            before_ids = _identifiers(transition.before, action_key)
            after_ids = _identifiers(transition.after, action_key)
            if transition.dest is None:
                edge = _Edge('internal', None, prepare_ids + before_ids + after_ids)
            else:
                if transition.dest not in machine.states:
                    raise MinimizationError(
                        "The transition of event '%s' from state '%s' points to unknown state '%s'."
                        % (event, state_name, transition.dest))
                enter_ids = state_info[transition.dest]['enter']
                actions = prepare_ids + before_ids + exit_ids + enter_ids + after_ids
                edge = _Edge('external', transition.dest, actions)
            edges[state_name][event] = edge
            raw_transitions[(state_name, event)] = transition
    return edges, raw_transitions


def _build_quotient(machine, mapping, partition, alphabet, edges, state_info, raw_transitions,
                    reject_state, needs_reject):
    """Construct the quotient :class:`Machine` and its label table."""
    if reject_state is True:
        reject_state = 'rejected'

    state_configs = []
    for representative in partition:
        state = state_info[representative]['state']
        config = {
            'name': representative,
            'on_enter': list(state.on_enter),
            'on_exit': list(state.on_exit),
        }
        if state_info[representative]['final']:
            config['final'] = True
        state_configs.append(config)

    quotient = Machine(model=None, states=state_configs, initial=mapping[machine.initial],
                       auto_transitions=False, ignore_invalid_triggers=False)

    for representative, members in partition.items():
        member = members[0]
        for event in alphabet:
            edge = edges[member].get(event)
            if edge is None:
                continue
            transition = raw_transitions[(member, event)]
            if edge.kind == 'internal':
                quotient.add_transition(event, representative, None,
                                        prepare=list(transition.prepare),
                                        before=list(transition.before),
                                        after=list(transition.after))
            else:
                quotient.add_transition(event, representative, mapping[edge.dest],
                                        prepare=list(transition.prepare),
                                        before=list(transition.before),
                                        after=list(transition.after))

    quotient_labels = OrderedDict((representative, state_info[representative]['label'])
                                  for representative in partition)

    if needs_reject and reject_state is not False:
        if reject_state in quotient.states:
            raise MinimizationError(
                "Reject state name '%s' collides with a representative state name." % reject_state)
        quotient.add_state(reject_state)
        for representative, members in partition.items():
            member = members[0]
            for event in alphabet:
                if edges[member].get(event) is None:
                    quotient.add_transition(event, representative, reject_state)
        for event in alphabet:
            quotient.add_transition(event, reject_state, reject_state)
        quotient_labels[reject_state] = REJECT

    return quotient, quotient_labels


def _build_quotient_view(machine, mapping, partition, alphabet, edges, base_signatures):
    """Derive the normalized observation view of the quotient directly from the partition.

    Building it analytically (instead of reading the quotient machine back) guarantees that explicit
    reject-state transitions keep the same uniform rejection semantics as missing edges.
    """
    quotient_edges = OrderedDict((representative, OrderedDict()) for representative in partition)
    for representative, members in partition.items():
        member = members[0]
        for event in alphabet:
            edge = edges[member].get(event)
            if edge is None:
                continue
            if edge.kind == 'internal':
                quotient_edges[representative][event] = _Edge('internal', None, edge.actions)
            else:
                quotient_edges[representative][event] = _Edge(
                    'external', mapping[edge.dest], edge.actions)
    quotient_bases = OrderedDict((representative, base_signatures[representative])
                                 for representative in partition)
    return _View(list(partition), mapping[machine.initial], alphabet, quotient_edges, quotient_bases)
