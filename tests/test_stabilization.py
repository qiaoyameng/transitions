# -*- coding: utf-8 -*-
try:
    from builtins import object
except ImportError:
    pass

import asyncio
from unittest import TestCase

from transitions.extensions.nesting import (
    HierarchicalMachine, NestedEventData, EVENTLESS_TRIGGER, StabilizationError,
)
from transitions.extensions.asyncio import HierarchicalAsyncMachine


def _parallel_states():
    return [
        'IDLE',
        {
            'name': 'WORK',
            'parallel': [
                {
                    'name': 'r1',
                    'children': [{'name': 'w1'},
                                 {'name': 'w1b'},
                                 {'name': 'f1', 'final': True}],
                    'initial': 'w1',
                },
                {
                    'name': 'r2',
                    'children': [{'name': 'w2'},
                                 {'name': 'f2', 'final': True}],
                    'initial': 'w2',
                },
            ],
        },
        'COMPLETE',
    ]


class TestEventlessStabilization(TestCase):

    def _build_parallel_machine(self, **kwargs):
        machine = HierarchicalMachine(
            states=_parallel_states(),
            transitions=[['start', 'IDLE', 'WORK'],
                         ['advance', 'WORK_r1_w1', 'WORK_r1_w1b']],
            initial='IDLE', auto_transitions=False, **kwargs)
        machine.add_eventless_transition('WORK_r1_w1b', 'WORK_r1_f1')
        machine.add_eventless_transition('WORK_r2_w2', 'WORK_r2_f2')
        # completion transition of the parallel state, only after both regions are final
        machine.add_eventless_transition('WORK', 'COMPLETE')
        return machine

    def test_stabilize_after_external_event(self):
        machine = self._build_parallel_machine()
        # after 'start', region r2 reaches its final state in microstep 0 while
        # region r1 stays in 'w1' -> the configuration must remain stable there
        self.assertTrue(machine.start())
        self.assertEqual(machine.state, ['WORK_r1_w1', 'WORK_r2_f2'])
        trace = machine.stabilization_trace
        self.assertEqual(len(trace), 1)
        self.assertEqual([(t['source'], t['dest']) for t in trace[0]['transitions']],
                         [('WORK_r2_w2', 'WORK_r2_f2')])
        # a new external event is processed only once the machine is stable; afterwards
        # r1 needs two microsteps (w1b -> f1) and only then may WORK -> COMPLETE fire
        machine.advance()
        self.assertEqual(machine.state, 'COMPLETE')
        sources = [t['source'] for rec in machine.stabilization_trace
                   for t in rec['transitions']]
        self.assertEqual(sources, ['WORK_r1_w1b', 'WORK'])
        self.assertEqual(machine.stabilization_trace[0]['states'],
                         ['WORK_r1_w1b', 'WORK_r2_f2'])

    def test_parallel_regions_same_microstep(self):
        # when both regions can finalize immediately, the two non-conflicting transitions
        # must be part of the same microstep; WORK's completion follows one microstep later
        machine = HierarchicalMachine(
            states=[
                'IDLE',
                {'name': 'P', 'parallel': [
                    {'name': 'a', 'children': ['s', {'name': 'fa', 'final': True}],
                     'initial': 's'},
                    {'name': 'b', 'children': ['s', {'name': 'fb', 'final': True}],
                     'initial': 's'}]},
                'DONE',
            ],
            transitions=[['start', 'IDLE', 'P']], initial='IDLE', auto_transitions=False)
        machine.add_eventless_transition('P_a_s', 'P_a_fa')
        machine.add_eventless_transition('P_b_s', 'P_b_fb')
        machine.add_eventless_transition('P', 'DONE')
        machine.start()
        trace = machine.stabilization_trace
        self.assertEqual(len(trace), 2)
        self.assertEqual([(t['source'], t['dest']) for t in trace[0]['transitions']],
                         [('P_a_s', 'P_a_fa'), ('P_b_s', 'P_b_fb')])
        self.assertEqual([(t['source'], t['dest']) for t in trace[1]['transitions']],
                         [('P', 'DONE')])
        self.assertEqual(machine.state, 'DONE')

    def test_substate_takes_precedence_over_parent(self):
        states = [
            'S0',
            {'name': 'P', 'children': [
                {'name': 'C', 'children': ['a', 'b'], 'initial': 'a'},
                'D'], 'initial': 'C'},
            'OUT',
        ]
        machine = HierarchicalMachine(states=states, transitions=[['start', 'S0', 'P']],
                                      initial='S0', auto_transitions=False)
        # both P and P_C_a offer eventless transitions; the substate must win and the
        # parent transition must not be considered in the same microstep
        machine.add_eventless_transition('P', 'OUT')
        machine.add_eventless_transition('P_C_a', 'P_C_b')
        machine.start()
        self.assertEqual(machine.state, 'P_C_b')
        fired = [t['source'] for rec in machine.stabilization_trace
                 for t in rec['transitions']]
        self.assertEqual(fired, ['P_C_a'])

    def test_microstep_callback_order(self):
        log = []
        machine = HierarchicalMachine(
            states=['X', 'Y', 'Z'], transitions=[['kick', 'X', 'Y']],
            initial='X', auto_transitions=False)
        machine.on_exit_X(lambda: log.append('exit_X'))
        machine.on_enter_Y(lambda: log.append('enter_Y'))
        machine.on_exit_Y(lambda: log.append('exit_Y'))
        machine.on_enter_Z(lambda: log.append('enter_Z'))
        machine.add_eventless_transition(
            'Y', 'Z',
            before=lambda: log.append('before_YZ'),
            after=lambda: log.append('after_YZ'))
        machine.kick()
        # every microstep obeys: transition action before, state exit, state entry, action after
        self.assertEqual(log, ['exit_X', 'enter_Y',
                               'before_YZ', 'exit_Y', 'enter_Z', 'after_YZ'])

    def test_failed_condition_is_stable_not_error(self):
        # an eventless transition whose condition is not met must leave the machine in a
        # stable state; it must NOT be reported as non-convergence
        machine = HierarchicalMachine(
            states=['A', 'B'], transitions=[['start', 'A', 'B']],
            initial='A', auto_transitions=False, max_stabilization_steps=3)
        machine.ready = False
        machine.add_eventless_transition('B', 'A', conditions=lambda: machine.ready)
        machine.start()
        self.assertEqual(machine.state, 'B')
        self.assertEqual(machine.stabilization_trace, [])

    def test_conditional_automatic_loop_converges(self):
        states = ['IDLE',
                  {'name': 'WORK', 'children': [
                      {'name': 'S0'}, {'name': 'S1'},
                      {'name': 'F', 'final': True}], 'initial': 'S0'},
                  'COMPLETE']
        machine = HierarchicalMachine(
            states=states, transitions=[['start', 'IDLE', 'WORK']],
            initial='IDLE', auto_transitions=False)
        machine.ticks = 0

        def more():
            return machine.ticks < 4

        def tick():
            machine.ticks += 1

        machine.add_eventless_transition('WORK_S0', 'WORK_S1',
                                         conditions=more, before=tick)
        machine.add_eventless_transition('WORK_S1', 'WORK_S0',
                                         conditions=more, before=tick)
        machine.add_eventless_transition(['WORK_S0', 'WORK_S1'], 'WORK_F',
                                         unless=more)
        machine.add_eventless_transition('WORK', 'COMPLETE')
        machine.start()
        self.assertEqual(machine.ticks, 4)
        self.assertEqual(machine.state, 'COMPLETE')
        loop = [t['source'] for rec in machine.stabilization_trace
                for t in rec['transitions']]
        self.assertEqual(loop,
                         ['WORK_S0', 'WORK_S1', 'WORK_S0', 'WORK_S1',
                          'WORK_S0', 'WORK'])

    def test_unconditional_loop_does_not_converge(self):
        machine = HierarchicalMachine(
            states=['A', 'B', 'C'], transitions=[['go', 'A', 'B']],
            initial='A', auto_transitions=False, max_stabilization_steps=7)
        machine.add_eventless_transition('B', 'C')
        machine.add_eventless_transition('C', 'B')
        with self.assertRaises(StabilizationError) as cm:
            machine.go()
        err = cm.exception
        self.assertEqual(err.max_steps, 7)
        self.assertEqual(len(err.trace), 7)
        # the report contains the states visited and the transitions executed
        self.assertEqual(err.trace[0]['states'], ['B'])
        self.assertEqual([(t['source'], t['dest']) for t in err.trace[0]['transitions']],
                         [('B', 'C')])
        self.assertEqual([(t['source'], t['dest']) for t in err.trace[1]['transitions']],
                         [('C', 'B')])
        # this is a dedicated error, never a plain 'condition unmet' / False result

    def test_zero_limit_stable_configuration_does_not_raise(self):
        # a limit of 0 must only flag genuine non-convergence; if the configuration is
        # already stable there is nothing to report
        machine = HierarchicalMachine(
            states=['A', 'B'], transitions=[['go', 'A', 'B']],
            initial='A', auto_transitions=False, max_stabilization_steps=0)
        machine.go()
        self.assertEqual(machine.state, 'B')
        self.assertEqual(machine.stabilization_trace, [])

    def test_queued_external_event_deferred_until_stable(self):
        log = []
        machine = HierarchicalMachine(
            states=['A', 'B', 'C', 'D'],
            transitions=[['start', 'A', 'B'], ['poke', 'C', 'D']],
            initial='A', queued=True, auto_transitions=False)

        def enter_b():
            log.append('enter_b')
            machine.poke()

        machine.on_enter_B(enter_b)
        machine.add_eventless_transition('B', 'C')
        machine.start()
        # the queued 'poke' event is only processed after B -> C stabilization
        self.assertEqual(log, ['enter_b'])
        self.assertEqual(machine.state, 'D')

    def test_stabilization_can_be_disabled(self):
        machine = HierarchicalMachine(
            states=['A', 'B', 'C'], transitions=[['go', 'A', 'B']],
            initial='A', auto_transitions=False, stabilize=False)
        machine.add_eventless_transition('B', 'C')
        machine.go()
        self.assertEqual(machine.state, 'B')
        self.assertEqual(machine.stabilization_trace, [])

    def test_eventless_trigger_is_not_an_external_event(self):
        machine = HierarchicalMachine(
            states=['A', 'B'], transitions=[['go', 'A', 'B']],
            initial='A', auto_transitions=False)
        machine.add_eventless_transition('B', 'A')
        # the sentinel must never be bound to a model as a callable trigger
        self.assertFalse(any(name == EVENTLESS_TRIGGER for name in vars(machine)))
        self.assertNotIn(EVENTLESS_TRIGGER, machine.get_triggers('B'))
        self.assertNotIn(EVENTLESS_TRIGGER, machine.get_nested_triggers())
        listed = [(t.source, t.dest) for t in machine.get_nested_transitions()]
        self.assertEqual(listed, [('A', 'B')])
        # it can still be introspected explicitly and removed again
        explicit = machine.get_nested_transitions(EVENTLESS_TRIGGER)
        self.assertEqual([(t.source, t.dest) for t in explicit], [('B', 'A')])
        machine.remove_transition(EVENTLESS_TRIGGER, 'B', 'A')
        self.assertNotIn(EVENTLESS_TRIGGER, machine.events)

    def test_none_trigger_registers_eventless_transition(self):
        machine = HierarchicalMachine(states=['A', 'B'],
                                      transitions=[{'trigger': None,
                                                    'source': 'A', 'dest': 'B'}],
                                      initial='A', auto_transitions=False)
        event_data = NestedEventData(machine.get_state('A'), None, machine, machine, (), {})
        machine._stabilize(event_data)
        self.assertEqual(machine.state, 'B')

    def test_none_is_not_accepted_as_an_externally_callable_trigger(self):
        # using trigger=None must not create an attribute called 'None' on the model
        machine = HierarchicalMachine(states=['A', 'B'],
                                      transitions=[{'trigger': None,
                                                    'source': 'A', 'dest': 'B'}],
                                      initial='A', auto_transitions=False)
        self.assertFalse(any(name == EVENTLESS_TRIGGER for name in vars(machine)))

    def test_markup_does_not_contain_eventless_trigger(self):
        from transitions.extensions.markup import HierarchicalMarkupMachine
        machine = HierarchicalMarkupMachine(
            states=['A', 'B'], transitions=[['go', 'A', 'B']],
            initial='A', auto_transitions=False)
        machine.add_eventless_transition('B', 'A')
        triggers = {t['trigger'] for t in machine.markup['transitions']}
        self.assertEqual(triggers, {'go'})


class TestAsyncEventlessStabilization(TestCase):

    def test_async_parallel_and_non_convergence(self):
        async def run():
            states = [
                'IDLE',
                {'name': 'P', 'parallel': [
                    {'name': 'r1', 'children': [
                        {'name': 'w1'}, {'name': 'f1', 'final': True}],
                        'initial': 'w1'},
                    {'name': 'r2', 'children': [
                        {'name': 'w2'}, {'name': 'w2b'},
                        {'name': 'f2', 'final': True}],
                        'initial': 'w2'}]},
                'DONE',
            ]
            machine = HierarchicalAsyncMachine(
                states=states,
                transitions=[['start', 'IDLE', 'P'],
                             ['advance', 'P_r2_w2', 'P_r2_w2b']],
                initial='IDLE', auto_transitions=False)
            machine.add_eventless_transition('P_r1_w1', 'P_r1_f1')
            machine.add_eventless_transition('P_r2_w2b', 'P_r2_f2')
            machine.add_eventless_transition('P', 'DONE')
            await machine.start()
            self.assertEqual(machine.state, ['P_r1_f1', 'P_r2_w2'])
            await machine.advance()
            self.assertEqual(machine.state, 'DONE')

            looping = HierarchicalAsyncMachine(
                states=['A', 'B', 'C'], transitions=[['go', 'A', 'B']],
                initial='A', auto_transitions=False, max_stabilization_steps=4)
            looping.add_eventless_transition('B', 'C')
            looping.add_eventless_transition('C', 'B')
            with self.assertRaises(StabilizationError):
                await looping.go()

        asyncio.run(run())
