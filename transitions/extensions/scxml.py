# -*- coding: utf-8 -*-
"""
    transitions.extensions.scxml
    ----------------------------

    A deliberately restricted adapter between SCXML documents and the
    (hierarchical) state dictionaries used by transitions.

    The adapter is meant for vendor-supplied documents that should be
    converted into plain machine configuration. It supports exactly the
    following constructs:

    * ``<scxml>`` with an optional ``initial`` attribute (or a single
      ``<initial>`` child) and no (or an empty) ``datamodel`` attribute
    * ``<state>`` with a mandatory ``id``, an optional ``initial``
      attribute or ``<initial>`` child and nested ``<state>``,
      ``<parallel>`` and ``<final>`` children
    * ``<parallel>`` whose regions are all entered simultaneously
    * ``<final>`` mapped to a state with ``final=True``
    * ``<transition>`` with a mandatory ``event`` and a single ``target``
      as well as an optional ``cond`` attribute
    * ``<onentry>``/``<onexit>`` and executable content of transitions

    Conditions and actions are never evaluated. A ``cond`` attribute must
    contain white-space separated names (optionally negated with a leading
    ``!``) that are looked up in the caller-provided ``conditions``
    mapping. Executable content elements are looked up by tag name in the
    caller-provided ``actions`` mapping. No expression is evaluated and no
    module is imported based on document content.

    Documents containing ``<script>``, ``<history>`` or ``<datamodel>``
    elements, DOCTYPE/entity declarations, unknown datamodels or any other
    unsupported construct are rejected with an :class:`SCXMLError` that
    points at the offending element.

    SCXML ids are mapped to generated, separator-safe state names which
    are unique within the whole document. The mappings are exposed via
    :attr:`SCXMLAdapter.id_to_name`, :attr:`SCXMLAdapter.name_to_id` and
    :attr:`SCXMLAdapter.path_to_id` so callers can translate between the
    XML world and hierarchical state names without ambiguity.
"""

from collections import OrderedDict
import logging
import xml.etree.ElementTree as ElementTree

from six import string_types

from .nesting import HierarchicalMachine, NestedState

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class SCXMLError(ValueError):
    """Raised when a document or machine cannot be handled by the restricted SCXML adapter."""


class _SCXMLNode(object):
    """Bookkeeping record for a state-like SCXML element."""

    def __init__(self, tag, state_id, name, path):
        self.tag = tag
        self.state_id = state_id
        self.name = name
        self.path = path  # generated names from the document root to this node
        self.children = []  # child nodes (state-like elements only)
        self.initial_name = None  # generated name of the initial child of exclusive states
        self.transitions = []  # _SCXMLTransition records
        self.on_enter = []
        self.on_exit = []


class _SCXMLTransition(object):
    """Bookkeeping record for a <transition> element."""

    def __init__(self, events, conditions, unless, target_id, actions):
        self.events = events
        self.conditions = conditions
        self.unless = unless
        self.target_id = target_id
        self.actions = actions


class SCXMLAdapter(object):
    """Converts a restricted subset of SCXML into transitions state dictionaries and back.

    Attributes:
        conditions (dict): Mapping of names allowed in ``cond`` attributes to callables.
        actions (dict): Mapping of executable content element names to callables.
        separator (str): Hierarchical state name separator of the target machine.
    """

    namespace = 'http://www.w3.org/2005/07/scxml'

    def __init__(self, conditions=None, actions=None, separator=NestedState.separator):
        """
        Args:
            conditions (dict): Maps names used in ``cond`` attributes to callables. Names may be
                negated with a leading ``!`` in which case the callable is used as an ``unless``
                condition.
            actions (dict): Maps executable content element names (for instance ``<log/>``) to
                callables. Elements inside ``<onentry>``/``<onexit>`` become ``on_enter``/``on_exit``
                callbacks, elements inside ``<transition>`` become ``before`` callbacks.
            separator (str): Separator used by the target machine to join hierarchical state names.
        """
        self.conditions = dict(conditions) if conditions else {}
        self.actions = dict(actions) if actions else {}
        self.separator = separator
        self._condition_funcs = dict((id(func), name) for name, func in self.conditions.items())
        self._action_funcs = dict((id(func), name) for name, func in self.actions.items())
        self._reset_mappings()

    @property
    def id_to_name(self):
        """dict: SCXML id -> list of generated state names (document order). Vendor documents
        may reuse an id in different scopes, hence every id maps to a list."""
        return OrderedDict((key, list(value)) for key, value in self._id_to_name.items())

    @property
    def name_to_id(self):
        """dict: Generated state name -> SCXML id. Generated names are unique per document."""
        return OrderedDict(self._name_to_id)

    @property
    def path_to_id(self):
        """dict: Hierarchical state path (separator-joined) -> SCXML id."""
        return OrderedDict((self.separator.join(path), state_id)
                           for path, state_id in self._path_to_id.items())

    def parse(self, source):
        """Parses an SCXML document into a configuration dictionary for HierarchicalMachine.

        Args:
            source (str or bytes or file-like): The SCXML document.

        Returns:
            dict: A dictionary with the keys ``states``, ``initial`` and ``transitions`` which can
            be passed to ``HierarchicalMachine`` (or a subclass) as keyword arguments. Conditions
            and actions are embedded as the callables provided to this adapter.

        Raises:
            SCXMLError: If the document uses constructs outside the supported subset.
        """
        self._reset_mappings()
        text = _read_text(source)
        _reject_unsafe_constructs(text)
        try:
            root_element = ElementTree.fromstring(text)
        except ElementTree.ParseError as err:
            raise SCXMLError("Document is not well-formed XML: {0}".format(err))
        if _local_name(root_element.tag) != 'scxml':
            raise SCXMLError("Root element must be <scxml>, found <{0}>.".format(_local_name(root_element.tag)))
        datamodel = root_element.get('datamodel')
        if datamodel:
            raise SCXMLError("Unknown datamodel '{0}' in <scxml>; this adapter does not evaluate "
                             "any datamodel.".format(datamodel))
        tops = []
        initial_id = self._check_optional_id(root_element.get('initial'), root_element, 'initial')
        for element in root_element:
            tag = _local_name(element.tag)
            if tag in ('state', 'parallel', 'final'):
                tops.append(self._parse_state(element, None))
            elif tag == 'initial':
                if initial_id is not None:
                    raise SCXMLError("<scxml> defines 'initial' both as attribute and as element.")
                initial_id = self._parse_initial_element(element, None)
            else:
                self._reject_element(element, None)
        if not tops:
            raise SCXMLError("<scxml> does not contain any states.")
        transitions = []
        for node in self._nodes:
            self._emit_transitions(node, transitions)
        return {'states': [self._emit_state(node) for node in tops],
                'initial': self._resolve_root_initial(initial_id, tops),
                'transitions': transitions}

    def export(self, machine):
        """Exports a HierarchicalMachine to an SCXML document string.

        Only constructs understood by :meth:`parse` are emitted. State ids are recovered from the
        mappings of previously parsed documents when the same adapter instance is used; otherwise
        ids are derived from state names. Callbacks must be part of the ``conditions``/``actions``
        mappings (compared by identity) so they can be written as names.

        Args:
            machine (HierarchicalMachine): The machine to export.

        Returns:
            str: The SCXML document.

        Raises:
            SCXMLError: If the machine contains constructs that cannot be represented.
        """
        if not isinstance(machine, HierarchicalMachine):
            raise SCXMLError("export expects a HierarchicalMachine instance, got {0!r}.".format(type(machine)))
        separator = machine.state_cls.separator
        self._used_ids = set(self._path_to_id.values())
        root_names = set(machine.states.keys())
        elements = OrderedDict()  # path tuple -> (element, state)
        pending = []  # (scope path, event name, source name, transition)

        def visit(scope_path):
            for name, state in list(machine.states.items()):
                path = scope_path + (name,)
                elements[path] = (self._export_state_element(state, path), state)
            for event in list(machine.events.values()):
                if self._is_auto_event(machine, event):
                    continue
                for source, event_transitions in event.transitions.items():
                    for transition in event_transitions:
                        pending.append((scope_path, event.name, source, transition))
            for name, state in list(machine.states.items()):
                if state.states:
                    with machine(name):
                        visit(scope_path + (name,))

        visit(())

        root = ElementTree.Element('scxml')
        root.set('xmlns', self.namespace)
        root.set('version', '1.0')
        initial = getattr(machine.initial, 'name', machine.initial)
        if isinstance(initial, string_types):
            root.set('initial', self._export_id((initial,), initial))
        for path, (element, _state) in elements.items():
            parent = elements.get(path[:-1])
            (parent[0] if parent is not None else root).append(element)
        for scope_path, event_name, source, transition in pending:
            source_path = scope_path + tuple(source.split(separator))
            try:
                source_element = elements[source_path][0]
            except KeyError:
                raise SCXMLError("Cannot export transition of event '{0}'; source '{1}' is not a "
                                 "registered state.".format(event_name, source))
            source_element.append(self._export_transition(machine, scope_path, root_names,
                                                          event_name, transition))
        _indent(root)
        payload = ElementTree.tostring(root, encoding='utf-8')
        return payload if isinstance(payload, string_types) else payload.decode('utf-8')

    def _reset_mappings(self):
        self._id_to_name = OrderedDict()  # scxml id -> [generated names]
        self._name_to_id = OrderedDict()  # generated name -> scxml id
        self._path_to_id = OrderedDict()  # path tuple -> scxml id
        self._id_index = OrderedDict()  # scxml id -> [_SCXMLNode]
        self._nodes = []  # all nodes in document order
        self._used_names = set()
        self._used_ids = set()

    # ------------------------------------------------------------------ parsing

    def _parse_state(self, element, parent_node):
        tag = _local_name(element.tag)
        state_id = element.get('id')
        if not state_id:
            raise SCXMLError("<{0}> without 'id' attribute in {1}.".format(tag, self._location(parent_node)))
        parent_path = parent_node.path if parent_node is not None else []
        name = self._generate_name(state_id)
        node = _SCXMLNode(tag, state_id, name, parent_path + [name])
        self._register(node)
        if tag == 'parallel' and element.get('initial'):
            raise SCXMLError("<parallel id='{0}'> must not define 'initial'; all regions are "
                             "entered.".format(state_id))
        initial_element = None
        for child in element:
            child_tag = _local_name(child.tag)
            if child_tag in ('state', 'parallel', 'final'):
                if tag == 'final':
                    raise SCXMLError("<final id='{0}'> must not contain <{1}>.".format(state_id, child_tag))
                node.children.append(self._parse_state(child, node))
            elif child_tag == 'transition':
                if tag == 'final':
                    raise SCXMLError("<final id='{0}'> must not contain <transition>.".format(state_id))
                node.transitions.append(self._parse_transition(child, node))
            elif child_tag == 'initial':
                if tag != 'state':
                    raise SCXMLError("<initial> is not allowed in <{0} id='{1}'>.".format(tag, state_id))
                if initial_element is not None:
                    raise SCXMLError("<state id='{0}'> contains more than one <initial> "
                                     "element.".format(state_id))
                initial_element = child
            elif child_tag == 'onentry':
                node.on_enter.extend(self._parse_executable(child, node))
            elif child_tag == 'onexit':
                node.on_exit.extend(self._parse_executable(child, node))
            else:
                self._reject_element(child, node)
        if tag == 'state':
            initial_id = self._check_optional_id(element.get('initial'), element, 'initial')
            if initial_id is not None and initial_element is not None:
                raise SCXMLError("<state id='{0}'> defines 'initial' both as attribute and as "
                                 "element.".format(state_id))
            if initial_element is not None:
                initial_id = self._parse_initial_element(initial_element, node)
            if initial_id is not None:
                node.initial_name = self._resolve_child_id(node, initial_id)
            elif node.children:
                node.initial_name = node.children[0].name
        elif tag == 'parallel' and not node.children:
            raise SCXMLError("<parallel id='{0}'> does not contain any states.".format(state_id))
        return node

    def _parse_initial_element(self, element, node):
        transitions = []
        for child in element:
            if _local_name(child.tag) == 'transition':
                transitions.append(child)
            else:
                self._reject_element(child, node)
        if len(transitions) != 1:
            raise SCXMLError("<initial> in {0} must contain exactly one <transition>."
                             "".format(self._location(node)))
        transition = transitions[0]
        if transition.get('event') or transition.get('cond') or len(transition):
            raise SCXMLError("<initial> transition in {0} must not define events, conditions or "
                             "actions.".format(self._location(node)))
        target = transition.get('target')
        if not target:
            raise SCXMLError("<initial> transition in {0} has no 'target'.".format(self._location(node)))
        return self._check_optional_id(target, transition, 'target')

    def _parse_transition(self, element, node):
        event_attribute = element.get('event')
        if not event_attribute or not event_attribute.strip():
            raise SCXMLError("<transition> in {0} has no 'event'; eventless transitions are not "
                             "supported.".format(self._location(node)))
        events = []
        for token in event_attribute.split():
            if '.' in token or '*' in token:
                raise SCXMLError("Event descriptor '{0}' in {1} is not supported; use plain event "
                                 "names.".format(token, self._location(node)))
            events.append(token)
        transition_type = element.get('type')
        if transition_type and transition_type != 'external':
            raise SCXMLError("Transition type '{0}' in {1} is not supported."
                             "".format(transition_type, self._location(node)))
        target_id = self._check_optional_id(element.get('target'), element, 'target')
        if target_id is None:
            raise SCXMLError("<transition> in {0} has no 'target'; internal transitions are not "
                             "supported.".format(self._location(node)))
        conditions, unless = self._map_conditions(element.get('cond'), node)
        actions = self._parse_executable(element, node)
        return _SCXMLTransition(events, conditions, unless, target_id, actions)

    def _parse_executable(self, container, node):
        callbacks = []
        for child in container:
            tag = _local_name(child.tag)
            if tag == 'script':
                raise SCXMLError("<script> in {0} is not supported; this adapter never evaluates "
                                 "document content.".format(self._location(node)))
            func = self.actions.get(tag)
            if func is None:
                raise SCXMLError("No action mapped for element <{0}> in {1}; only caller-provided "
                                 "actions are executed.".format(tag, self._location(node)))
            callbacks.append(func)
        return callbacks

    def _map_conditions(self, cond, node):
        conditions, unless = [], []
        for token in (cond or '').split():
            target = True
            if token.startswith('!'):
                target = False
                token = token[1:]
            func = self.conditions.get(token)
            if func is None:
                raise SCXMLError("cond '{0}' in {1} does not match a provided condition name; "
                                 "conditions are never evaluated.".format(token, self._location(node)))
            (conditions if target else unless).append(func)
        return conditions, unless

    def _reject_element(self, element, node):
        tag = _local_name(element.tag)
        if tag == 'script':
            raise SCXMLError("<script> in {0} is not supported; this adapter never evaluates "
                             "document content.".format(self._location(node)))
        if tag == 'history':
            raise SCXMLError("<history> in {0} is not supported by this adapter version."
                             "".format(self._location(node)))
        if tag == 'datamodel':
            raise SCXMLError("<datamodel> in {0} is not supported; this adapter does not evaluate "
                             "any datamodel.".format(self._location(node)))
        raise SCXMLError("Unsupported element <{0}> in {1}.".format(tag, self._location(node)))

    def _location(self, node):
        if node is None:
            return "<scxml>"
        return "<{0} id='{1}'>".format(node.tag, node.state_id)

    def _check_optional_id(self, value, element, attribute):
        if value is None:
            return None
        ids = value.split()
        if len(ids) != 1:
            raise SCXMLError("Attribute '{0}' of <{1}> must reference exactly one id, found "
                             "'{2}'; multiple targets are not supported."
                             "".format(attribute, _local_name(element.tag), value))
        return ids[0]

    def _generate_name(self, state_id):
        name = state_id.replace(self.separator, '-') if self.separator else state_id
        candidate, counter = name, 2
        while candidate in self._used_names:
            candidate = '{0}-{1}'.format(name, counter)
            counter += 1
        self._used_names.add(candidate)
        return candidate

    def _register(self, node):
        self._id_to_name.setdefault(node.state_id, []).append(node.name)
        self._name_to_id[node.name] = node.state_id
        self._path_to_id[tuple(node.path)] = node.state_id
        self._id_index.setdefault(node.state_id, []).append(node)
        self._nodes.append(node)

    def _resolve_child_id(self, node, target_id):
        matches = [child for child in node.children if child.state_id == target_id]
        if not matches:
            raise SCXMLError("Initial state '{0}' of {1} is not one of its children."
                             "".format(target_id, self._location(node)))
        if len(matches) > 1:
            raise SCXMLError("Initial state '{0}' of {1} is ambiguous."
                             "".format(target_id, self._location(node)))
        return matches[0].name

    def _resolve_root_initial(self, initial_id, tops):
        if initial_id is None:
            return tops[0].name
        matches = [node for node in tops if node.state_id == initial_id]
        if not matches:
            raise SCXMLError("Unknown initial state id '{0}' in <scxml>.".format(initial_id))
        if len(matches) > 1:
            raise SCXMLError("Initial state id '{0}' of <scxml> is ambiguous.".format(initial_id))
        return matches[0].name

    def _resolve_target(self, target_id, node):
        candidates = self._id_index.get(target_id, [])
        if not candidates:
            raise SCXMLError("Unknown target id '{0}' referenced by a transition in {1}."
                             "".format(target_id, self._location(node)))
        if len(candidates) == 1:
            return candidates[0]
        best, chosen = -1, []
        for candidate in candidates:
            common = _common_prefix_length(node.path, candidate.path)
            if common > best:
                best, chosen = common, [candidate]
            elif common == best:
                chosen.append(candidate)
        if len(chosen) > 1:
            matches = ", ".join(sorted("state '{0}'".format(self.separator.join(candidate.path))
                                       for candidate in chosen))
            raise SCXMLError("Target id '{0}' referenced by a transition in {1} is ambiguous; it "
                             "matches {2}.".format(target_id, self._location(node), matches))
        return chosen[0]

    def _emit_state(self, node):
        definition = {'name': node.name}
        if node.tag == 'final':
            definition['final'] = True
        elif node.tag == 'parallel':
            definition['parallel'] = [self._emit_state(child) for child in node.children]
        elif node.children:
            definition['children'] = [self._emit_state(child) for child in node.children]
            definition['initial'] = node.initial_name
        if node.on_enter:
            definition['on_enter'] = list(node.on_enter)
        if node.on_exit:
            definition['on_exit'] = list(node.on_exit)
        return definition

    def _emit_transitions(self, node, transitions):
        for record in node.transitions:
            target = self._resolve_target(record.target_id, node)
            source = self.separator.join(node.path)
            dest = self.separator.join(target.path)
            for event in record.events:
                definition = {'trigger': event, 'source': source, 'dest': dest}
                if record.conditions:
                    definition['conditions'] = list(record.conditions)
                if record.unless:
                    definition['unless'] = list(record.unless)
                if record.actions:
                    definition['before'] = list(record.actions)
                transitions.append(definition)

    # ------------------------------------------------------------------ exporting

    def _export_state_element(self, state, path):
        leaf = path[-1]
        if state.final and state.states:
            raise SCXMLError("State '{0}' is final but has children; this cannot be represented "
                             "in SCXML.".format(self.separator.join(path)))
        if state.final:
            element = ElementTree.Element('final')
        elif isinstance(state.initial, (list, tuple)):
            element = ElementTree.Element('parallel')
        else:
            element = ElementTree.Element('state')
        element.set('id', self._export_id(path, leaf))
        if element.tag == 'state' and state.states:
            initial = getattr(state.initial, 'name', state.initial)
            if isinstance(initial, string_types):
                element.set('initial', self._export_id(path + (initial,), initial))
        if state.on_enter:
            onentry = ElementTree.SubElement(element, 'onentry')
            for callback in state.on_enter:
                ElementTree.SubElement(onentry, self._callback_key(callback, self.actions,
                                                                   self._action_funcs, path))
        if state.on_exit:
            onexit = ElementTree.SubElement(element, 'onexit')
            for callback in state.on_exit:
                ElementTree.SubElement(onexit, self._callback_key(callback, self.actions,
                                                                  self._action_funcs, path))
        return element

    def _export_transition(self, machine, scope_path, root_names, event_name, transition):
        if transition.dest is None:
            raise SCXMLError("Cannot export internal transition (dest=None) of event '{0}'."
                             "".format(event_name))
        if transition.prepare or transition.after:
            raise SCXMLError("Cannot export 'prepare'/'after' callbacks of event '{0}'; only "
                             "conditions and executable content are supported.".format(event_name))
        separator = machine.state_cls.separator
        dest_parts = transition.dest.split(separator)
        # a destination whose first segment is a root state is considered global, otherwise
        # it is resolved relative to the scope the transition is defined in
        if dest_parts[0] in root_names:
            dest_path = tuple(dest_parts)
        else:
            dest_path = scope_path + tuple(dest_parts)
        try:
            machine.get_state(list(dest_path))
        except ValueError:
            raise SCXMLError("Cannot export transition of event '{0}'; dest '{1}' is not a "
                             "registered state.".format(event_name, transition.dest))
        element = ElementTree.Element('transition')
        element.set('event', event_name)
        element.set('target', self._export_id(dest_path, dest_path[-1]))
        tokens = []
        for condition in transition.conditions:
            key = self._callback_key(condition.func, self.conditions, self._condition_funcs, dest_path)
            tokens.append(key if condition.target else '!' + key)
        if tokens:
            element.set('cond', ' '.join(tokens))
        for callback in transition.before:
            ElementTree.SubElement(element, self._callback_key(callback, self.actions,
                                                               self._action_funcs, dest_path))
        return element

    def _export_id(self, path, leaf):
        path = tuple(path)
        state_id = self._path_to_id.get(path)
        if state_id is None:
            state_id, counter = leaf, 2
            while state_id in self._used_ids:
                state_id = '{0}-{1}'.format(leaf, counter)
                counter += 1
            self._path_to_id[path] = state_id
            self._id_to_name.setdefault(state_id, []).append(leaf)
            if leaf not in self._name_to_id:
                self._name_to_id[leaf] = state_id
        self._used_ids.add(state_id)
        return state_id

    def _callback_key(self, callback, mapping, reverse, path):
        if isinstance(callback, string_types):
            if callback in mapping:
                return callback
        else:
            key = reverse.get(id(callback))
            if key is None:
                name = getattr(callback, '__name__', None)
                if name is not None and mapping.get(name) is callback:
                    key = name
            if key is not None:
                return key
        raise SCXMLError("Cannot export callback {0!r} (state '{1}'); add it to the adapter's "
                         "conditions/actions mappings.".format(callback, self.separator.join(path)))

    @staticmethod
    def _is_auto_event(machine, event):
        if not event.name.startswith('to_'):
            return False
        try:
            machine.get_state(event.name[3:])
            return True
        except ValueError:
            return False


def _read_text(source):
    if hasattr(source, 'read'):
        source = source.read()
    if isinstance(source, bytes):
        return source.decode('utf-8')
    return source


def _reject_unsafe_constructs(text):
    for declaration in ('<!DOCTYPE', '<!ENTITY'):
        if declaration in text:
            raise SCXMLError("DOCTYPE and (external) entity declarations are not supported; "
                             "found '{0}'.".format(declaration))


def _local_name(tag):
    if not isinstance(tag, string_types):
        return str(tag)
    return tag.rsplit('}', 1)[-1]


def _common_prefix_length(first, second):
    length = 0
    for one, other in zip(first, second):
        if one != other:
            break
        length += 1
    return length


def _indent(element, level=0):
    padding = '\n' + '  ' * level
    children = list(element)
    if children:
        if not element.text or not element.text.strip():
            element.text = padding + '  '
        for child in children:
            _indent(child, level + 1)
        if not children[-1].tail or not children[-1].tail.strip():
            children[-1].tail = padding
    if level and (not element.tail or not element.tail.strip()):
        element.tail = padding
