"""
    transitions.extensions.scxml
    ----------------------------

    A deliberately limited adapter which converts SCXML documents (https://www.w3.org/TR/scxml/)
    into markup dictionaries suitable for
    :class:`transitions.extensions.markup.HierarchicalMarkupMachine`.

    Supported constructs:
        * ``<scxml>`` (optional ``initial`` attribute or ``<initial>`` child; only the
          ``null`` datamodel is accepted)
        * ``<state>`` with nested states, ``initial`` attribute or ``<initial>`` child
          and ``<onentry>``/``<onexit>`` blocks
        * ``<parallel>`` (all regions are entered; parallel regions may be nested)
        * ``<final>``
        * ``<transition>`` with an ``event`` attribute and a single ``target``

    Conditions (``cond`` attributes) and actions (content of ``<onentry>``, ``<onexit>``
    and ``<transition>``) are resolved exclusively through caller-provided name mappings.
    They are never evaluated and no modules are imported based on document content.
    Everything else -- most notably ``<script>``, data model bindings, ``<history>``,
    DOCTYPE/entity declarations as well as eventless or multi-target transitions -- is
    rejected with an :class:`SCXMLParseError` naming the offending element.
"""

import re
import xml.etree.ElementTree as ET

from six import string_types, iteritems

from .nesting import NestedState


class SCXMLParseError(ValueError):
    """Raised when an SCXML document uses constructs outside the supported subset."""


class SCXMLAdapter(object):
    """Converts a subset of SCXML into ``transitions`` markup dictionaries.

    Args:
        callbacks (dict): Maps action names used in ``<onentry>``, ``<onexit>`` and
            ``<transition>`` bodies to callables.
        conditions (dict): Maps names used in ``cond`` attributes to callables.
        separator (str): Separator used to build hierarchical state names.
            Defaults to ``NestedState.separator``.

    Attributes:
        state_map (dict): Maps SCXML element ids to (possibly escaped) hierarchical
            state names. Populated by :meth:`parse`. XML ids and hierarchical state
            names are tracked separately so that ids containing the separator or
            ids equal to another state's hierarchical name cannot cause conflicts.
    """

    # SCXML elements which are recognized but deliberately not implemented
    unsupported_elements = frozenset([
        'script', 'datamodel', 'data', 'assign', 'raise', 'send', 'log',
        'if', 'elseif', 'else', 'foreach', 'invoke', 'finalize', 'cancel',
        'history', 'content', 'param', 'donedata',
    ])

    # markup keys which contain action references
    _callback_keys = frozenset([
        'on_enter', 'on_exit', 'on_timeout', 'before', 'after', 'prepare',
        'before_state_change', 'after_state_change', 'prepare_event',
        'finalize_event', 'on_exception', 'on_final',
    ])
    # markup keys which contain condition references
    _condition_keys = frozenset(['conditions', 'unless'])

    def __init__(self, callbacks=None, conditions=None, separator=None):
        self.callbacks = dict(callbacks) if callbacks else {}
        self.conditions = dict(conditions) if conditions else {}
        self.separator = separator if separator is not None else NestedState.separator
        self.state_map = {}
        self._full_names = set()
        self._pending_initials = []
        self._pending_transitions = []

    def parse(self, source):
        """Parses an SCXML document into a markup dictionary.

        Args:
            source: The SCXML document as a string, bytes or a file-like object.

        Returns:
            dict: A markup dictionary suitable for ``HierarchicalMarkupMachine(markup=...)``.

        Raises:
            SCXMLParseError: If the document uses constructs outside the supported subset.
        """
        self.state_map = {}
        self._full_names = set()
        self._pending_initials = []
        self._pending_transitions = []
        text = source.read() if hasattr(source, 'read') else source
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        self._reject_entities(text)
        try:
            root = ET.fromstring(text)
        except ET.ParseError as error:
            raise SCXMLParseError("cannot parse SCXML document: {0}".format(error))
        if _local_name(root.tag) != 'scxml':
            raise SCXMLParseError("expected <scxml> root element but found "
                                  "<{0}>".format(_local_name(root.tag)))
        datamodel = root.get('datamodel')
        if datamodel not in (None, '', 'null'):
            raise SCXMLParseError("<scxml> uses unsupported datamodel '{0}'; only the "
                                  "'null' datamodel is supported".format(datamodel))
        states = []
        transitions = []
        initial_holder = {}
        for child in root:
            tag = _local_name(child.tag)
            if tag in ('state', 'parallel', 'final'):
                states.append(self._parse_state(child, [], transitions))
            elif tag == 'initial':
                self._parse_initial(child, initial_holder, context='<scxml>')
            else:
                self._reject_element(child)
        initial_attr = root.get('initial')
        if initial_attr is not None:
            if initial_holder:
                raise SCXMLParseError("<scxml> must not use an 'initial' attribute and "
                                      "an <initial> element at the same time")
            initial_holder['ids'] = initial_attr.split()
        if not states:
            raise SCXMLParseError("<scxml> does not contain any states")
        markup = {'states': states, 'transitions': transitions,
                  'models': [{'class-name': 'self'}]}
        if initial_holder:
            self._pending_initials.append((markup, initial_holder['ids'], None))
        self._resolve_pending()
        if 'initial' not in markup:
            # SCXML default: the first state in document order
            markup['initial'] = states[0]['name']
        return markup

    def resolve(self, markup):
        """Substitutes mapped callables for callback and condition names in a markup dict.

        Exported markup (for instance ``machine.markup``) represents callbacks by name.
        This replaces every name found in the adapter's ``callbacks``/``conditions``
        mappings with the mapped callable so that re-imported machines share the exact
        same callback objects. Names without a mapping are left untouched.

        Args:
            markup (dict): A markup dictionary, e.g. produced by ``machine.markup``.

        Returns:
            dict: A copy of ``markup`` with mapped callables substituted.
        """
        resolved = {}
        for key, value in iteritems(markup):
            if key in ('states', 'children') and isinstance(value, list):
                resolved[key] = [self.resolve(item) for item in value]
            elif key == 'transitions' and isinstance(value, list):
                resolved[key] = [self.resolve(item) for item in value]
            elif key in self._callback_keys:
                resolved[key] = _map_references(value, self.callbacks)
            elif key in self._condition_keys:
                resolved[key] = _map_references(value, self.conditions)
            else:
                resolved[key] = value
        return resolved

    def _parse_state(self, elem, path, transitions):
        tag = _local_name(elem.tag)
        xml_id = elem.get('id')
        if not xml_id:
            raise SCXMLParseError("<{0}> without an 'id' attribute".format(tag))
        if xml_id in self.state_map:
            raise SCXMLParseError("duplicate state id '{0}'".format(xml_id))
        # escape the separator so that ids cannot collide with hierarchical names
        name = xml_id.replace(self.separator, self.separator * 2)
        full_path = path + [name]
        full_name = self.separator.join(full_path)
        if full_name in self._full_names:
            raise SCXMLParseError("id '{0}' results in the ambiguous state name '{1}'; rename the "
                                  "state or choose a different separator".format(xml_id, full_name))
        self._full_names.add(full_name)
        self.state_map[xml_id] = full_name

        state_def = {'name': name}
        children = []
        initial_holder = {}
        for child in elem:
            child_tag = _local_name(child.tag)
            if child_tag in ('state', 'parallel', 'final'):
                if tag == 'final':
                    raise SCXMLParseError("<final id='{0}'> must not contain states".format(xml_id))
                children.append(self._parse_state(child, full_path, transitions))
            elif child_tag == 'transition':
                if tag == 'final':
                    raise SCXMLParseError("<final id='{0}'> must not contain "
                                          "transitions".format(xml_id))
                self._parse_transition(child, full_name, transitions)
            elif child_tag == 'initial':
                if tag != 'state':
                    raise SCXMLParseError("<initial> is not allowed in "
                                          "<{0} id='{1}'>".format(tag, xml_id))
                self._parse_initial(child, initial_holder, context="<state id='{0}'>".format(xml_id))
            elif child_tag == 'onentry':
                state_def['on_enter'] = self._parse_actions(child)
            elif child_tag == 'onexit':
                state_def['on_exit'] = self._parse_actions(child)
            else:
                self._reject_element(child)

        if tag == 'final':
            state_def['final'] = True
            return state_def

        initial_attr = elem.get('initial')
        if initial_attr is not None and initial_holder:
            raise SCXMLParseError("<{0} id='{1}'> must not use an 'initial' attribute and an "
                                  "<initial> element at the same time".format(tag, xml_id))
        if tag == 'parallel':
            if initial_attr is not None or initial_holder:
                raise SCXMLParseError("<parallel id='{0}'> must not define an initial state; "
                                      "all regions are entered".format(xml_id))
            if not children:
                raise SCXMLParseError("<parallel id='{0}'> does not contain any states".format(xml_id))
            state_def['children'] = children
            state_def['initial'] = [child['name'] for child in children]
        elif children:
            state_def['children'] = children
            if initial_holder:
                self._pending_initials.append((state_def, initial_holder['ids'], full_path))
            elif initial_attr is not None:
                self._pending_initials.append((state_def, initial_attr.split(), full_path))
            else:
                # SCXML default: the first child in document order
                state_def['initial'] = children[0]['name']
        elif initial_attr is not None or initial_holder:
            raise SCXMLParseError("<{0} id='{1}'> defines an initial state but has no "
                                  "child states".format(tag, xml_id))
        return state_def

    def _parse_initial(self, elem, holder, context):
        if holder:
            raise SCXMLParseError("multiple <initial> elements in {0}".format(context))
        transitions = [child for child in elem if _local_name(child.tag) == 'transition']
        for child in elem:
            if _local_name(child.tag) != 'transition':
                self._reject_element(child)
        if len(transitions) != 1:
            raise SCXMLParseError("<initial> in {0} must contain exactly one "
                                  "<transition>".format(context))
        transition = transitions[0]
        if transition.get('event') is not None or transition.get('cond') is not None \
                or len(transition):
            raise SCXMLParseError("<initial> transitions in {0} only support the 'target' "
                                  "attribute".format(context))
        target = transition.get('target')
        if not target:
            raise SCXMLParseError("<initial> transition in {0} has no 'target'".format(context))
        holder['ids'] = target.split()

    def _parse_transition(self, elem, source, transitions):
        event_attr = elem.get('event')
        if not event_attr:
            raise SCXMLParseError("<transition> in state '{0}' has no 'event' attribute; "
                                  "eventless transitions are not supported".format(source))
        target_attr = elem.get('target')
        if not target_attr:
            raise SCXMLParseError("<transition event='{0}'> in state '{1}' has no "
                                  "'target'".format(event_attr, source))
        targets = target_attr.split()
        if len(targets) != 1:
            raise SCXMLParseError("<transition event='{0}'> in state '{1}' targets multiple states; "
                                  "only a single target is supported".format(event_attr, source))
        conditions = []
        cond = elem.get('cond')
        if cond is not None:
            cond = cond.strip()
            if cond not in self.conditions:
                raise SCXMLParseError("condition '{0}' of <transition event='{1}'> is not part of the "
                                      "provided condition mapping; conditions are never "
                                      "evaluated".format(cond, event_attr))
            conditions.append(self.conditions[cond])
        before = self._parse_actions(elem)
        for event in event_attr.split():
            self._pending_transitions.append((event, source, targets[0],
                                              conditions, before, transitions))

    def _parse_actions(self, elem):
        actions = []
        for child in elem:
            tag = _local_name(child.tag)
            if tag in self.unsupported_elements:
                self._reject_element(child)
            if tag not in self.callbacks:
                raise SCXMLParseError("no callback mapped for action <{0}> in <{1}>; actions are only "
                                      "resolved through the provided callback "
                                      "mapping".format(tag, _local_name(elem.tag)))
            actions.append(self.callbacks[tag])
        return actions

    def _resolve_pending(self):
        for holder, ids, parent_path in self._pending_initials:
            names = []
            for xml_id in ids:
                if xml_id not in self.state_map:
                    raise SCXMLParseError("initial target '{0}' does not match any "
                                          "state id".format(xml_id))
                name = self.state_map[xml_id]
                if parent_path is not None:
                    prefix = self.separator.join(parent_path) + self.separator
                    if not name.startswith(prefix) or self.separator in name[len(prefix):]:
                        raise SCXMLParseError("initial target '{0}' is not a direct child of "
                                              "state '{1}'".format(xml_id, self.separator.join(parent_path)))
                    name = name[len(prefix):]
                names.append(name)
            if parent_path is not None and len(names) != 1:
                raise SCXMLParseError("the initial state of '{0}' must reference exactly one "
                                      "child".format(self.separator.join(parent_path)))
            holder['initial'] = names[0] if len(names) == 1 else names
        for event, source, target_id, conditions, before, transitions in self._pending_transitions:
            if target_id not in self.state_map:
                raise SCXMLParseError("target '{0}' of <transition event='{1}'> does not match any "
                                      "state id".format(target_id, event))
            transition = {'trigger': event, 'source': source, 'dest': self.state_map[target_id]}
            if conditions:
                transition['conditions'] = list(conditions)
            if before:
                transition['before'] = list(before)
            transitions.append(transition)

    def _reject_element(self, elem):
        tag = _local_name(elem.tag)
        if tag in self.unsupported_elements:
            raise SCXMLParseError("<{0}> is not supported by this SCXML adapter".format(tag))
        raise SCXMLParseError("unknown element <{0}> in SCXML document".format(tag))

    @staticmethod
    def _reject_entities(text):
        match = re.search(r'<!\s*(DOCTYPE|ENTITY)', text, re.IGNORECASE)
        if match:
            raise SCXMLParseError("'{0}' declarations are not supported; external entities "
                                  "are rejected".format(match.group(0)))


def parse_scxml(source, callbacks=None, conditions=None, separator=None):
    """Parses an SCXML document into a markup dictionary (see :class:`SCXMLAdapter`).

    Returns:
        tuple: A ``(markup, state_map)`` tuple where ``markup`` is suitable for
            ``HierarchicalMarkupMachine(markup=...)`` and ``state_map`` maps SCXML
            element ids to hierarchical state names.
    """
    adapter = SCXMLAdapter(callbacks=callbacks, conditions=conditions, separator=separator)
    return adapter.parse(source), adapter.state_map


def _local_name(tag):
    return tag.rsplit('}', 1)[-1] if isinstance(tag, string_types) else tag


def _map_references(value, mapping):
    if isinstance(value, string_types):
        return mapping.get(value, value)
    if isinstance(value, (list, tuple)):
        return [_map_references(item, mapping) for item in value]
    return value
