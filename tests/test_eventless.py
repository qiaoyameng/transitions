# -*- coding: utf-8 -*-
try:
    from builtins import object
except ImportError:
    pass

from unittest import TestCase

from transitions import MachineError
from transitions.extensions.nesting import HierarchicalMachine, StabilizationError


class TestEventlessTransitions(TestCase):

    def test_eventless_chain_after_external_event(self):
        # a device receives 'start' and automatically advances through several
        # checks until it reaches a state where it waits for external input
        states = ['idle', {'name': 'active', 'initial': 'booting',
                           'children': ['booting', 'checking', 'waiting']}]
        transitions = [
            ['start', 'idle', 'active'],
            [None, 'active_booting', 'active_checking'],
            [None, 'active_checking', 'active_waiting'],
        ]
        machine = HierarchicalMachine(states=states, transitions=transitions, initial='idle')
        machine.start()
        self.assertEqual('active_waiting', machine.state)

    def test_eventless_unmet_condition_is_stable(self):
        # an eventless transition with unmet conditions is just not taken;
        # the configuration is considered stable and no error is raised
        class Model(object):
            ready = False

        model = Model()
        machine = HierarchicalMachine(model, states=['A', 'B'], initial='A')
        machine.add_transition(None, 'A', 'B', conditions='ready')
        machine.add_transition('poke', 'A', 'A')
        self.assertTrue(model.poke())
        self.assertEqual('A', model.state)
        model.ready = True
        model.poke()
        self.assertEqual('B', model.state)

    def test_eventless_nested_definition(self):
        # eventless transitions can be defined in the 'transitions' section of nested states
        states = [{'name': 'P', 'initial': 'a', 'children': ['a', 'b'],
                   'transitions': [[None, 'a', 'b']]}, 'X']
        machine = HierarchicalMachine(states=states, initial='X',
                                      transitions=[['go', 'X', 'P']])
        machine.go()
        self.assertEqual('P_b', machine.state)

    def test_child_transition_precedes_parent_transition(self):
        order = []

        class Model(object):
            def note(self, name):
                order.append(name)

        model = Model()
        states = [{'name': 'P', 'children': ['a', 'b'], 'initial': 'a'}, 'Q', 'X']
        machine = HierarchicalMachine(model, states=states, initial='X')
        machine.add_transition('go', 'X', 'P')
        machine.add_transition(None, 'P', 'Q', before=lambda: model.note('parent'))
        machine.add_transition(None, 'P_a', 'P_b', before=lambda: model.note('child'))
        model.go()
        # the child state transition is executed in the first microstep;
        # the parent transition only applies once the child configuration is stable
        self.assertEqual(['child', 'parent'], order)
        self.assertEqual('Q', model.state)

    def test_parallel_regions_reach_final_in_different_microsteps(self):
        calls = []
        finals = []

        class Model(object):
            def on_exit_P_1_a(self):
                calls.append('exit_1a')

            def on_exit_P_2_a(self):
                calls.append('exit_2a')

            def on_enter_P_1_b(self):
                calls.append('enter_1b')

            def on_enter_P_2_b(self):
                calls.append('enter_2b')

            def on_exit_P_2_b(self):
                calls.append('exit_2b')

            def on_enter_P_2_c(self):
                calls.append('enter_2c')

        model = Model()
        states = ['idle', {'name': 'P', 'parallel': [
            {'name': '1', 'children': ['a', {'name': 'b', 'final': True}], 'initial': 'a'},
            {'name': '2', 'children': ['a', 'b', {'name': 'c', 'final': True}], 'initial': 'a'}]}]
        transitions = [
            ['start', 'idle', 'P'],
            # region 1 reaches its final state in the first microstep ...
            [None, 'P_1_a', 'P_1_b'],
            # ... region 2 needs two microsteps
            [None, 'P_2_a', 'P_2_b'],
            [None, 'P_2_b', 'P_2_c'],
        ]
        machine = HierarchicalMachine(model, states=states, transitions=transitions,
                                      initial='idle', on_final=lambda: finals.append(model.state))
        model.start()
        self.assertEqual(['P_1_b', 'P_2_c'], model.state)
        # non-conflicting transitions of both regions are executed in one microstep;
        # every microstep processes exits before enters
        self.assertEqual(['exit_1a', 'exit_2a', 'enter_1b', 'enter_2b',
                          'exit_2b', 'enter_2c'], calls)
        # on_final is only called once all parallel regions reached a final state
        self.assertEqual([['P_1_b', 'P_2_c']], finals)

    def test_conditional_loop_stops_normally(self):
        class Model(object):
            def __init__(self):
                self.iterations = 0

            def keep_looping(self):
                return self.iterations < 3

            def tick(self):
                self.iterations += 1

        model = Model()
        transitions = [
            ['go', 'idle', 'A'],
            [None, 'A', 'B'],
            {'trigger': None, 'source': 'B', 'dest': 'A',
             'conditions': 'keep_looping', 'after': 'tick'},
        ]
        machine = HierarchicalMachine(model, states=['idle', 'A', 'B'],
                                      transitions=transitions, initial='idle')
        model.go()
        # the automatic loop A -> B -> A -> ... stops as soon as the condition fails
        self.assertEqual('B', model.state)
        self.assertEqual(3, model.iterations)

    def test_non_converging_loop_raises_stabilization_error(self):
        transitions = [
            ['go', 'X', 'A'],
            [None, 'A', 'B'],
            [None, 'B', 'A'],
        ]
        machine = HierarchicalMachine(states=['X', 'A', 'B'], transitions=transitions,
                                      initial='X', max_microsteps=10)
        with self.assertRaises(StabilizationError) as context:
            machine.go()
        message = str(context.exception)
        # the error reports the traversed states and transitions ...
        self.assertIn('A -> B', message)
        self.assertIn('B -> A', message)
        self.assertIn('10', message)
        # ... and is a dedicated error, not a silently unmet condition
        self.assertIsInstance(context.exception, MachineError)

    def test_external_events_are_deferred_until_stable(self):
        log = []

        class Model(object):
            def on_enter_B(self):
                log.append('enter_B')
                # triggered while stabilization is still in progress
                self.extra()

            def on_enter_C(self):
                log.append('enter_C')

            def on_enter_D(self):
                log.append('enter_D')

        model = Model()
        transitions = [
            ['go', 'A', 'B'],
            [None, 'B', 'C'],
            ['extra', 'C', 'D'],
        ]
        machine = HierarchicalMachine(model, states=['A', 'B', 'C', 'D'], initial='A',
                                      transitions=transitions, queued=True)
        model.go()
        # 'extra' is queued and only processed after the stable configuration (C) is reached
        self.assertEqual(['enter_B', 'enter_C', 'enter_D'], log)
        self.assertEqual('D', model.state)

    def test_eventless_transition_not_bound_to_model(self):
        machine = HierarchicalMachine(states=['A', 'B'], initial='A',
                                      transitions=[[None, 'A', 'B']])
        self.assertFalse(hasattr(machine, machine.eventless_event_name))
        self.assertFalse(hasattr(machine, 'may_%s' % machine.eventless_event_name))
