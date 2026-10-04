try:
    from builtins import object
except ImportError:
    pass

from unittest import TestCase

from transitions.extensions.nesting import HierarchicalMachine
from transitions.extensions.scxml import SCXMLAdapter, SCXMLError


# A fixed vendor document with nested parallel regions. Note that the id 'work_zone'
# contains the state name separator and that the ids 'lane', 'open' and 'shut' are
# reused in different regions on purpose.
DOCUMENT = """<?xml version="1.0" encoding="UTF-8"?>
<scxml xmlns="http://www.w3.org/2005/07/scxml" version="1.0" initial="idle">
  <state id="idle">
    <onentry><notify/></onentry>
    <onexit><log_exit/></onexit>
    <transition event="start" target="work_zone"/>
  </state>
  <parallel id="work_zone">
    <onentry><notify/></onentry>
    <state id="north">
      <initial><transition target="lane"/></initial>
      <state id="lane">
        <transition event="advance" target="lane_done" cond="ready">
          <notify/>
        </transition>
      </state>
      <final id="lane_done"/>
    </state>
    <state id="south">
      <parallel id="lanes">
        <state id="lane">
          <initial><transition target="open"/></initial>
          <state id="open">
            <transition event="advance" target="shut" cond="!blocked"/>
          </state>
          <final id="shut"/>
        </state>
        <state id="overflow" initial="open">
          <state id="open">
            <transition event="advance" target="shut"/>
          </state>
          <final id="shut"/>
        </state>
      </parallel>
    </state>
    <transition event="shutdown" target="terminated"/>
  </parallel>
  <final id="terminated"/>
</scxml>
"""

EVENTS = ('start', 'advance', 'shutdown')


class Trace(object):
    """Records callback invocations; doubles as the condition/action provider."""

    def __init__(self):
        self.entries = []

    def notify(self):
        self.entries.append('notify')

    def log_exit(self):
        self.entries.append('log_exit')

    def ready(self):
        self.entries.append('cond:ready')
        return True

    def blocked(self):
        self.entries.append('cond:blocked')
        return False

    def on_final(self):
        self.entries.append('final')


def conditions(trace):
    return {'ready': trace.ready, 'blocked': trace.blocked}


def actions(trace):
    return {'notify': trace.notify, 'log_exit': trace.log_exit}


def handwritten_config(trace):
    """The state dictionary a developer would write by hand for DOCUMENT. Generated
    state names must not contain the separator, hence 'work-zone' rather than
    'work_zone'; reused ids are suffixed in document order."""
    return {
        'states': [
            {'name': 'idle', 'on_enter': [trace.notify], 'on_exit': [trace.log_exit]},
            {'name': 'work-zone', 'on_enter': [trace.notify], 'parallel': [
                {'name': 'north', 'initial': 'lane', 'children': [
                    {'name': 'lane'},
                    {'name': 'lane-done', 'final': True}]},
                {'name': 'south', 'initial': 'lanes', 'children': [
                    {'name': 'lanes', 'parallel': [
                        {'name': 'lane-2', 'initial': 'open', 'children': [
                            {'name': 'open'},
                            {'name': 'shut', 'final': True}]},
                        {'name': 'overflow', 'initial': 'open-2', 'children': [
                            {'name': 'open-2'},
                            {'name': 'shut-2', 'final': True}]}]}]}]},
            {'name': 'terminated', 'final': True}],
        'initial': 'idle',
        'transitions': [
            {'trigger': 'start', 'source': 'idle', 'dest': 'work-zone'},
            {'trigger': 'shutdown', 'source': 'work-zone', 'dest': 'terminated'},
            {'trigger': 'advance', 'source': 'work-zone_north_lane',
             'dest': 'work-zone_north_lane-done',
             'conditions': [trace.ready], 'before': [trace.notify]},
            {'trigger': 'advance', 'source': 'work-zone_south_lanes_lane-2_open',
             'dest': 'work-zone_south_lanes_lane-2_shut', 'unless': [trace.blocked]},
            {'trigger': 'advance', 'source': 'work-zone_south_lanes_overflow_open-2',
             'dest': 'work-zone_south_lanes_overflow_shut-2'}]}


def run_events(machine, events=EVENTS):
    states = [machine.state]
    for event in events:
        getattr(machine, event)()
        states.append(machine.state)
    return states


class TestSCXMLParse(TestCase):

    def setUp(self):
        self.trace = Trace()
        self.conditions = conditions(self.trace)
        self.actions = actions(self.trace)
        self.adapter = SCXMLAdapter(conditions=self.conditions, actions=self.actions)
        self.config = self.adapter.parse(DOCUMENT)

    def test_parse_matches_handwritten_config(self):
        self.assertEqual(handwritten_config(self.trace), self.config)

    def test_ids_with_separator_are_mapped_to_safe_names(self):
        self.assertEqual(['work-zone'], self.adapter.id_to_name['work_zone'])
        self.assertEqual('work_zone', self.adapter.name_to_id['work-zone'])
        self.assertEqual('lane_done', self.adapter.name_to_id['lane-done'])

    def test_reused_ids_are_mapped_separately(self):
        self.assertEqual(['lane', 'lane-2'], self.adapter.id_to_name['lane'])
        self.assertEqual(['open', 'open-2'], self.adapter.id_to_name['open'])
        self.assertEqual(['shut', 'shut-2'], self.adapter.id_to_name['shut'])
        self.assertEqual('lane', self.adapter.path_to_id['work-zone_north_lane'])
        self.assertEqual('lane', self.adapter.path_to_id['work-zone_south_lanes_lane-2'])
        self.assertEqual('shut', self.adapter.path_to_id['work-zone_south_lanes_overflow_shut-2'])

    def test_machine_can_be_built_from_parse_result(self):
        machine = HierarchicalMachine(**self.config)
        self.assertEqual('idle', machine.state)
        self.assertIn('work-zone', machine.states)
        self.assertTrue(machine.get_state('terminated').final)
        self.assertTrue(machine.get_state('work-zone_north_lane-done').final)
        self.assertFalse(machine.get_state('work-zone_north_lane').final)

    def test_parse_result_embeds_mapped_callables(self):
        advance = [t for t in self.config['transitions'] if t['trigger'] == 'advance']
        self.assertEqual(3, len(advance))
        self.assertIs(self.conditions['ready'], advance[0]['conditions'][0])
        self.assertIs(self.actions['notify'], advance[0]['before'][0])
        self.assertIs(self.conditions['blocked'], advance[1]['unless'][0])

    def test_initial_defaults_to_first_state(self):
        config = SCXMLAdapter().parse('<scxml><state id="a"/><state id="b"/></scxml>')
        self.assertEqual('a', config['initial'])

    def test_initial_element_at_root(self):
        config = SCXMLAdapter().parse(
            '<scxml><initial><transition target="b"/></initial><state id="a"/><state id="b"/></scxml>')
        self.assertEqual('b', config['initial'])

    def test_namespaced_and_plain_documents(self):
        plain = SCXMLAdapter().parse('<scxml initial="a"><state id="a"/></scxml>')
        self.assertEqual('a', plain['initial'])

    def test_parse_file_like_source(self):
        try:
            from io import StringIO
        except ImportError:
            from StringIO import StringIO
        config = SCXMLAdapter().parse(StringIO('<scxml initial="a"><state id="a"/></scxml>'))
        self.assertEqual('a', config['initial'])


class TestSCXMLRejections(TestCase):

    def assert_rejected(self, document, *keywords):
        with self.assertRaises(SCXMLError) as context:
            SCXMLAdapter().parse(document)
        for keyword in keywords:
            self.assertIn(keyword, str(context.exception))

    def test_reject_doctype_and_entities(self):
        self.assert_rejected(
            '<?xml version="1.0"?>\n'
            '<!DOCTYPE scxml [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>\n'
            '<scxml initial="a"><state id="a"/></scxml>', 'DOCTYPE', 'entity')

    def test_reject_script_element(self):
        self.assert_rejected(
            '<scxml initial="a"><state id="a"><script>evil()</script></state></scxml>',
            'script', "id='a'")

    def test_reject_script_in_executable_content(self):
        self.assert_rejected(
            '<scxml initial="a"><state id="a"><onentry><script>e()</script></onentry></state></scxml>',
            'script', "id='a'")

    def test_reject_history_element(self):
        self.assert_rejected(
            '<scxml initial="a"><state id="a"><history id="h"/></state></scxml>',
            'history', "id='a'")

    def test_reject_unknown_datamodel(self):
        self.assert_rejected(
            '<scxml datamodel="ecmascript" initial="a"><state id="a"/></scxml>',
            'datamodel', 'ecmascript')

    def test_reject_datamodel_element(self):
        self.assert_rejected(
            '<scxml initial="a"><datamodel><data id="x"/></datamodel><state id="a"/></scxml>',
            'datamodel')

    def test_reject_unknown_element(self):
        self.assert_rejected(
            '<scxml initial="a"><state id="a"><send event="x"/></state></scxml>',
            'send', "id='a'")

    def test_reject_unknown_condition_name(self):
        self.assert_rejected(
            '<scxml initial="a"><state id="a">'
            '<transition event="e" target="a" cond="missing"/></state></scxml>',
            'missing', "id='a'")

    def test_cond_is_never_evaluated(self):
        self.assert_rejected(
            '<scxml initial="a"><state id="a">'
            '<transition event="e" target="a" cond="__import__(\'os\').getcwd()"/></state></scxml>',
            'never evaluated')

    def test_reject_unmapped_action(self):
        self.assert_rejected(
            '<scxml initial="a"><state id="a"><onentry><explode/></onentry></state></scxml>',
            'explode', "id='a'")

    def test_reject_eventless_transition(self):
        self.assert_rejected(
            '<scxml initial="a"><state id="a"><transition target="a"/></state></scxml>',
            'event', "id='a'")

    def test_reject_targetless_transition(self):
        self.assert_rejected(
            '<scxml initial="a"><state id="a"><transition event="e"/></state></scxml>',
            'target', "id='a'")

    def test_reject_multiple_targets(self):
        self.assert_rejected(
            '<scxml initial="a"><state id="a"><transition event="e" target="a b"/></state>'
            '<state id="b"/></scxml>', 'multiple targets')

    def test_reject_event_descriptors(self):
        self.assert_rejected(
            '<scxml initial="a"><state id="a"><transition event="error.*" target="a"/></state></scxml>',
            'error.*')

    def test_reject_internal_transition_type(self):
        self.assert_rejected(
            '<scxml initial="a"><state id="a">'
            '<transition event="e" target="a" type="internal"/></state></scxml>', 'internal')

    def test_reject_unknown_target(self):
        self.assert_rejected(
            '<scxml initial="a"><state id="a"><transition event="e" target="nowhere"/></state></scxml>',
            'nowhere', "id='a'")

    def test_reject_ambiguous_target(self):
        self.assert_rejected(
            '<scxml initial="s"><state id="s"><transition event="go" target="dup"/></state>'
            '<parallel id="p"><state id="r1"><state id="dup"/></state>'
            '<state id="r2"><state id="dup"/></state></parallel></scxml>',
            'ambiguous', 'dup')

    def test_reject_missing_id(self):
        self.assert_rejected(
            '<scxml initial="a"><state><transition event="e" target="a"/></state></scxml>', 'id')

    def test_reject_non_scxml_root(self):
        self.assert_rejected('<state id="a"/>', 'scxml')

    def test_reject_malformed_xml(self):
        self.assert_rejected('<scxml>', 'well-formed')

    def test_reject_empty_document(self):
        self.assert_rejected('<scxml/>', 'does not contain any states')

    def test_reject_initial_outside_children(self):
        self.assert_rejected(
            '<scxml initial="a"><state id="a"><state id="b"/>'
            '<initial><transition target="c"/></initial></state></scxml>',
            'Initial', "'c'")


class TestSCXMLTraceComparison(TestCase):

    def test_handwritten_and_imported_traces_match(self):
        trace_manual, trace_imported = Trace(), Trace()
        manual = HierarchicalMachine(on_final=trace_manual.on_final,
                                     **handwritten_config(trace_manual))
        adapter = SCXMLAdapter(conditions=conditions(trace_imported), actions=actions(trace_imported))
        imported = HierarchicalMachine(on_final=trace_imported.on_final, **adapter.parse(DOCUMENT))

        states_manual = run_events(manual)
        states_imported = run_events(imported)

        self.assertEqual(states_manual, states_imported)
        self.assertEqual(trace_manual.entries, trace_imported.entries)
        # landmarks: all parallel regions reached their finals before the machine did
        self.assertEqual('terminated', imported.state)
        self.assertEqual(2, trace_imported.entries.count('final'))
        self.assertIn('cond:ready', trace_imported.entries)
        self.assertIn('cond:blocked', trace_imported.entries)
        self.assertIn('notify', trace_imported.entries)

    def test_conditions_can_block_transitions(self):
        class Blocking(Trace):
            def blocked(self):
                self.entries.append('cond:blocked')
                return True

        for config in (handwritten_config, None):
            trace = Blocking()
            if config is not None:
                machine = HierarchicalMachine(on_final=trace.on_final, **config(trace))
            else:
                adapter = SCXMLAdapter(conditions=conditions(trace), actions=actions(trace))
                machine = HierarchicalMachine(on_final=trace.on_final, **adapter.parse(DOCUMENT))
            machine.start()
            machine.advance()
            # the south region is stuck in 'open' since 'blocked' now returns True
            self.assertEqual(['work-zone_north_lane-done',
                              ['work-zone_south_lanes_lane-2_open',
                               'work-zone_south_lanes_overflow_shut-2']], machine.state)


class TestSCXMLRoundtrip(TestCase):

    def setUp(self):
        self.trace = Trace()
        self.conditions = conditions(self.trace)
        self.actions = actions(self.trace)
        self.adapter = SCXMLAdapter(conditions=self.conditions, actions=self.actions)
        self.machine = HierarchicalMachine(on_final=self.trace.on_final,
                                           **self.adapter.parse(DOCUMENT))
        self.document = self.adapter.export(self.machine)

    def test_export_contains_original_ids_and_structure(self):
        self.assertIn('xmlns="http://www.w3.org/2005/07/scxml"', self.document)
        self.assertIn('initial="idle"', self.document)
        self.assertIn('<parallel id="work_zone">', self.document)
        self.assertIn('<parallel id="lanes">', self.document)
        self.assertIn('<final id="terminated"', self.document)
        self.assertEqual(2, self.document.count('id="shut"'))
        self.assertIn('cond="ready"', self.document)
        self.assertIn('cond="!blocked"', self.document)
        self.assertIn('event="advance"', self.document)

    def test_reimport_produces_same_mappings(self):
        adapter = SCXMLAdapter(conditions=self.conditions, actions=self.actions)
        adapter.parse(self.document)
        self.assertEqual(self.adapter.id_to_name, adapter.id_to_name)
        self.assertEqual(self.adapter.name_to_id, adapter.name_to_id)
        self.assertEqual(self.adapter.path_to_id, adapter.path_to_id)

    def test_reimport_preserves_callback_identity(self):
        adapter = SCXMLAdapter(conditions=self.conditions, actions=self.actions)
        machine = HierarchicalMachine(**adapter.parse(self.document))
        north = machine.events['advance'].transitions['work-zone_north_lane'][0]
        self.assertIs(self.conditions['ready'], north.conditions[0].func)
        self.assertTrue(north.conditions[0].target)
        self.assertIs(self.actions['notify'], north.before[0])
        south = machine.events['advance'].transitions['work-zone_south_lanes_lane-2_open'][0]
        self.assertIs(self.conditions['blocked'], south.conditions[0].func)
        self.assertFalse(south.conditions[0].target)
        self.assertIs(self.actions['notify'], machine.get_state('idle').on_enter[0])
        self.assertIs(self.actions['log_exit'], machine.get_state('idle').on_exit[0])

    def test_reimport_preserves_final_semantics(self):
        adapter = SCXMLAdapter(conditions=self.conditions, actions=self.actions)
        machine = HierarchicalMachine(on_final=self.trace.on_final, **adapter.parse(self.document))
        self.assertTrue(machine.get_state('terminated').final)
        self.assertTrue(machine.get_state('work-zone_north_lane-done').final)
        self.assertTrue(machine.get_state('work-zone_south_lanes_lane-2_shut').final)
        # both machines share the very same callbacks, hence also the same trace
        states_original = run_events(self.machine)
        shared = len(self.trace.entries)
        states_reimported = run_events(machine)
        self.assertEqual(states_original, states_reimported)
        self.assertEqual(self.trace.entries[:shared], self.trace.entries[shared:])
        self.assertEqual('terminated', machine.state)
        self.assertEqual(2, self.trace.entries[shared:].count('final'))

    def test_export_scoped_transitions(self):
        adapter = SCXMLAdapter()
        states = ['idle', {'name': 'work', 'initial': 'a', 'children': ['a', 'b'],
                           'transitions': [{'trigger': 'go', 'source': 'a', 'dest': 'b'}]}]
        machine = HierarchicalMachine(states=states, initial='idle')
        document = adapter.export(machine)
        self.assertIn('<transition event="go" target="b"', document)
        reimported = HierarchicalMachine(**adapter.parse(document))
        reimported.to_work()
        reimported.go()
        self.assertEqual('work_b', reimported.state)

    def test_export_rejects_unmapped_callbacks(self):
        def stranger():
            pass

        machine = HierarchicalMachine(states=['a', 'b'], initial='a',
                                      transitions=[{'trigger': 'go', 'source': 'a', 'dest': 'b',
                                                    'conditions': [stranger]}])
        with self.assertRaises(SCXMLError) as context:
            SCXMLAdapter().export(machine)
        self.assertIn('stranger', str(context.exception))

    def test_export_rejects_non_hierarchical_machines(self):
        from transitions import Machine
        with self.assertRaises(SCXMLError):
            SCXMLAdapter().export(Machine(states=['a'], initial='a'))
