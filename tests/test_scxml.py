try:
    from builtins import object
except ImportError:
    pass

from unittest import TestCase

from transitions.extensions.markup import HierarchicalMarkupMachine
from transitions.extensions.scxml import SCXMLAdapter, SCXMLParseError, parse_scxml


# A fixed SCXML document as delivered by the device supplier. It contains nested
# parallel regions (console inside operational), a condition, custom actions and
# a final state.
DOCUMENT = """<scxml xmlns="http://www.w3.org/2005/07/scxml" version="1.0" datamodel="null" initial="idle">
  <state id="idle">
    <onentry><notify/></onentry>
    <transition event="start" target="operational"/>
  </state>
  <parallel id="operational">
    <state id="motion" initial="stopped">
      <state id="stopped">
        <transition event="go" target="moving"/>
      </state>
      <state id="moving">
        <onexit><honk/></onexit>
        <transition event="stop" target="stopped"><beep/></transition>
      </state>
    </state>
    <parallel id="console">
      <state id="lights" initial="off">
        <state id="off">
          <transition event="toggle" target="on"/>
        </state>
        <state id="on">
          <transition event="toggle" target="off"/>
        </state>
      </state>
      <state id="gear" initial="neutral">
        <state id="neutral">
          <transition event="shift" target="engaged" cond="clutch_pressed"/>
        </state>
        <state id="engaged">
          <transition event="shift" target="neutral"/>
        </state>
      </state>
    </parallel>
    <transition event="reset" target="idle"/>
    <transition event="finish" target="done"/>
  </parallel>
  <final id="done"/>
</scxml>
"""


def flatten(state):
    """Flattens (nested) parallel state lists into a sorted list of state names."""
    if isinstance(state, (list, tuple)):
        result = []
        for item in state:
            result.extend(flatten(item))
        return sorted(result)
    return [state]


class Tracker(object):

    def __init__(self):
        self.calls = []
        self.clutch = False


def make_mappings(tracker):
    """Creates name mappings for the document's conditions and actions. The function
    names match the mapping keys so that exported markup can be resolved back."""
    def notify():
        tracker.calls.append('notify')

    def beep():
        tracker.calls.append('beep')

    def honk():
        tracker.calls.append('honk')

    def clutch_pressed():
        return tracker.clutch

    callbacks = {'notify': notify, 'beep': beep, 'honk': honk}
    conditions = {'clutch_pressed': clutch_pressed}
    return callbacks, conditions


def build_handwritten(callbacks, conditions):
    """Builds the machine manually, mirroring DOCUMENT with transitions state dicts."""
    states = [
        {'name': 'idle', 'on_enter': [callbacks['notify']]},
        {'name': 'operational', 'parallel': [
            {'name': 'motion', 'initial': 'stopped', 'children': [
                {'name': 'stopped'},
                {'name': 'moving', 'on_exit': [callbacks['honk']]},
            ]},
            {'name': 'console', 'parallel': [
                {'name': 'lights', 'initial': 'off', 'children': [
                    {'name': 'off'},
                    {'name': 'on'},
                ]},
                {'name': 'gear', 'initial': 'neutral', 'children': [
                    {'name': 'neutral'},
                    {'name': 'engaged'},
                ]},
            ]},
        ]},
        {'name': 'done', 'final': True},
    ]
    transitions = [
        {'trigger': 'start', 'source': 'idle', 'dest': 'operational'},
        {'trigger': 'go', 'source': 'operational_motion_stopped',
         'dest': 'operational_motion_moving'},
        {'trigger': 'stop', 'source': 'operational_motion_moving',
         'dest': 'operational_motion_stopped', 'before': [callbacks['beep']]},
        {'trigger': 'toggle', 'source': 'operational_console_lights_off',
         'dest': 'operational_console_lights_on'},
        {'trigger': 'toggle', 'source': 'operational_console_lights_on',
         'dest': 'operational_console_lights_off'},
        {'trigger': 'shift', 'source': 'operational_console_gear_neutral',
         'dest': 'operational_console_gear_engaged', 'conditions': [conditions['clutch_pressed']]},
        {'trigger': 'shift', 'source': 'operational_console_gear_engaged',
         'dest': 'operational_console_gear_neutral'},
        {'trigger': 'reset', 'source': 'operational', 'dest': 'idle'},
        {'trigger': 'finish', 'source': 'operational', 'dest': 'done'},
    ]
    return HierarchicalMarkupMachine(states=states, transitions=transitions, initial='idle')


class TestSCXMLAdapter(TestCase):

    def setUp(self):
        self.tracker = Tracker()
        self.callbacks, self.conditions = make_mappings(self.tracker)
        self.adapter = SCXMLAdapter(callbacks=self.callbacks, conditions=self.conditions)
        self.markup = self.adapter.parse(DOCUMENT)

    def test_state_map(self):
        self.assertEqual({
            'idle': 'idle',
            'operational': 'operational',
            'motion': 'operational_motion',
            'stopped': 'operational_motion_stopped',
            'moving': 'operational_motion_moving',
            'console': 'operational_console',
            'lights': 'operational_console_lights',
            'off': 'operational_console_lights_off',
            'on': 'operational_console_lights_on',
            'gear': 'operational_console_gear',
            'neutral': 'operational_console_gear_neutral',
            'engaged': 'operational_console_gear_engaged',
            'done': 'done',
        }, self.adapter.state_map)

    def test_markup_structure(self):
        self.assertEqual('idle', self.markup['initial'])
        states = {state['name']: state for state in self.markup['states']}
        self.assertEqual(['done', 'idle', 'operational'], sorted(states))
        self.assertEqual(['motion', 'console'], states['operational']['initial'])
        console = states['operational']['children'][1]
        self.assertEqual(['lights', 'gear'], console['initial'])
        self.assertTrue(states['done']['final'])
        self.assertEqual([self.callbacks['notify']], states['idle']['on_enter'])
        moving = states['operational']['children'][0]['children'][1]
        self.assertEqual([self.callbacks['honk']], moving['on_exit'])
        self.assertEqual(9, len(self.markup['transitions']))
        shift = [t for t in self.markup['transitions']
                 if t['trigger'] == 'shift' and 'conditions' in t]
        self.assertEqual(1, len(shift))
        self.assertEqual([self.conditions['clutch_pressed']], shift[0]['conditions'])
        stop = [t for t in self.markup['transitions'] if t['trigger'] == 'stop']
        self.assertEqual([self.callbacks['beep']], stop[0]['before'])

    def test_trajectory_matches_handwritten(self):
        imported = HierarchicalMarkupMachine(markup=self.markup)
        hw_tracker = Tracker()
        hw_callbacks, hw_conditions = make_mappings(hw_tracker)
        handwritten = build_handwritten(hw_callbacks, hw_conditions)
        for machine in (imported, handwritten):
            machine.on_final.append(lambda *args, **kwargs: None)
        self.assertEqual(flatten(handwritten.state), flatten(imported.state))

        # the first 'shift' is blocked by the unmet condition, the second one passes
        script = [('start', False), ('go', False), ('toggle', False), ('shift', False),
                  ('stop', False), ('reset', False), ('start', False), ('shift', True),
                  ('finish', False)]
        for event, clutch in script:
            self.tracker.clutch = hw_tracker.clutch = clutch
            imported.trigger(event)
            handwritten.trigger(event)
            self.assertEqual(flatten(handwritten.state), flatten(imported.state),
                             "state mismatch after event '{0}'".format(event))
        self.assertEqual(self.tracker.calls, hw_tracker.calls)
        self.assertEqual(['beep', 'honk', 'notify'], self.tracker.calls)
        self.assertTrue(imported.is_done())
        self.assertTrue(handwritten.is_done())

    def test_condition_blocks_transition(self):
        machine = HierarchicalMarkupMachine(markup=self.adapter.parse(DOCUMENT))
        machine.start()
        machine.shift()
        self.assertNotIn('operational_console_gear_engaged', flatten(machine.state))
        self.tracker.clutch = True
        machine.shift()
        self.assertIn('operational_console_gear_engaged', flatten(machine.state))

    def test_markup_roundtrip(self):
        machine = HierarchicalMarkupMachine(markup=self.markup)
        machine.start()
        machine.go()
        exported = machine.markup

        # references are exported by name ...
        idle_def = [s for s in exported['states'] if s['name'] == 'idle'][0]
        self.assertEqual(['notify'], idle_def['on_enter'])
        shift = [t for t in exported['transitions']
                 if t['trigger'] == 'shift' and 'conditions' in t][0]
        self.assertEqual(['clutch_pressed'], shift['conditions'])
        done_def = [s for s in exported['states'] if s['name'] == 'done'][0]
        self.assertTrue(done_def['final'])

        # ... and resolve maps them back to the identical callables
        restored = HierarchicalMarkupMachine(markup=self.adapter.resolve(exported))
        self.assertIs(restored.states['idle'].on_enter[0], self.callbacks['notify'])
        self.assertIs(restored.get_state('operational_motion_moving').on_exit[0],
                      self.callbacks['honk'])
        stop = restored.events['stop'].transitions['operational_motion_moving'][0]
        self.assertIs(stop.before[0], self.callbacks['beep'])
        shift = restored.events['shift'].transitions['operational_console_gear_neutral'][0]
        self.assertIs(shift.conditions[0].func, self.conditions['clutch_pressed'])

        # final state semantics and the current state survive the round trip
        self.assertTrue(restored.get_state('done').final)
        self.assertEqual(flatten(machine.state), flatten(restored.state))
        for event in ['toggle', 'stop', 'finish']:
            machine.trigger(event)
            restored.trigger(event)
            self.assertEqual(flatten(machine.state), flatten(restored.state))
        self.assertTrue(restored.is_done())

    def test_initial_element_form(self):
        doc = """<scxml>
          <initial><transition target="one"/></initial>
          <state id="one">
            <initial><transition target="sub"/></initial>
            <state id="sub"/>
            <state id="other"/>
          </state>
          <state id="two"/>
        </scxml>"""
        markup, state_map = parse_scxml(doc)
        self.assertEqual('one', markup['initial'])
        one = [s for s in markup['states'] if s['name'] == 'one'][0]
        self.assertEqual('sub', one['initial'])
        machine = HierarchicalMarkupMachine(markup=markup)
        self.assertEqual('one_sub', machine.state)
        self.assertEqual('one_sub', state_map['sub'])

    def test_parse_scxml_function(self):
        markup, state_map = parse_scxml(DOCUMENT, callbacks=self.callbacks,
                                        conditions=self.conditions)
        self.assertEqual('idle', markup['initial'])
        self.assertEqual('operational_console', state_map['console'])
        machine = HierarchicalMarkupMachine(markup=markup)
        self.assertEqual('idle', machine.state)

    def test_separator_escaping(self):
        doc = """<scxml initial="a_b">
          <state id="a"><state id="b"/></state>
          <state id="a_b"><transition event="go" target="b"/></state>
        </scxml>"""
        markup, state_map = parse_scxml(doc)
        # the id 'a_b' must not clash with the hierarchical name of state 'b'
        self.assertEqual('a__b', state_map['a_b'])
        self.assertEqual('a_b', state_map['b'])
        machine = HierarchicalMarkupMachine(markup=markup)
        self.assertEqual('a__b', machine.state)
        machine.go()
        self.assertEqual('a_b', machine.state)

    def test_ambiguous_names_rejected(self):
        doc = """<scxml>
          <state id="a_"><state id="b"/></state>
          <state id="a"><state id="_b"/></state>
        </scxml>"""
        with self.assertRaises(SCXMLParseError) as ctx:
            parse_scxml(doc)
        self.assertIn('ambiguous', str(ctx.exception))

    def test_rejected_constructs(self):
        cases = [
            ('script', '<scxml><state id="a"><script>foo()</script></state></scxml>',
             'script'),
            ('entities', '<!DOCTYPE scxml [<!ENTITY x "y">]>'
                         '<scxml><state id="a"/></scxml>', 'DOCTYPE'),
            ('datamodel', '<scxml datamodel="ecmascript"><state id="a"/></scxml>',
             'datamodel'),
            ('datamodel element', '<scxml><datamodel><data id="x"/></datamodel>'
                                  '<state id="a"/></scxml>', 'datamodel'),
            ('history', '<scxml><state id="a"><history id="h"/></state></scxml>',
             'history'),
            ('unknown element', '<scxml><state id="a"><warp/></state></scxml>',
             'warp'),
            ('unmapped action', '<scxml><state id="a"><onentry><notify/></onentry>'
                                '</state></scxml>', 'notify'),
            ('unmapped condition', '<scxml><state id="a"><transition event="go" target="b" '
                                   'cond="system.online"/></state><state id="b"/></scxml>',
             'system.online'),
            ('eventless transition', '<scxml><state id="a"><transition target="b"/></state>'
                                     '<state id="b"/></scxml>', 'event'),
            ('unknown target', '<scxml><state id="a"><transition event="go" target="nowhere"/>'
                               '</state></scxml>', 'nowhere'),
            ('duplicate id', '<scxml><state id="a"/><state id="a"/></scxml>', 'duplicate'),
            ('multiple targets', '<scxml><state id="a"><transition event="go" target="b c"/>'
                                 '</state><state id="b"/><state id="c"/></scxml>', 'multiple'),
            ('wrong root', '<notscxml/>', 'scxml'),
            ('no states', '<scxml/>', 'states'),
        ]
        for name, doc, needle in cases:
            with self.subTest(case=name):
                with self.assertRaises(SCXMLParseError) as ctx:
                    SCXMLAdapter().parse(doc)
                self.assertIn(needle, str(ctx.exception))
