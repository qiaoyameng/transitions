try:
    from builtins import object
except ImportError:  # pragma: no cover
    pass

import random
from itertools import product
from unittest import TestCase

from transitions import Machine

from transitions.extensions.minimization import (
    REJECT, MinimizationError, minimize_machine,
)


def build_machine(states, transitions, initial='A', **kwargs):
    return Machine(states=list(states), initial=initial, auto_transitions=False,
                   transitions=transitions, **kwargs)


class TestMinimization(TestCase):

    def test_merges_equivalent_states(self):
        machine = build_machine(['A', 'B', 'C'], [
            {'trigger': 'x', 'source': 'A', 'dest': 'C'},
            {'trigger': 'x', 'source': 'B', 'dest': 'C'},
        ])
        result = minimize_machine(machine, ['x', 'y'], labels={'A': 'idle', 'B': 'idle', 'C': 'done'})
        self.assertEqual(result.mapping['A'], result.mapping['B'])
        self.assertNotEqual(result.mapping['A'], result.mapping['C'])
        self.assertEqual(result.partition[result.mapping['A']], ('A', 'B'))
        self.assertEqual(result.mapping['C'], 'C')

    def test_quotient_machine_runs(self):
        machine = build_machine(['A', 'B', 'C'], [
            {'trigger': 'x', 'source': 'A', 'dest': 'C'},
            {'trigger': 'x', 'source': 'B', 'dest': 'C'},
        ])
        result = minimize_machine(machine, ['x', 'y'], labels={'A': 0, 'B': 0, 'C': 1})

        class Model(object):
            pass

        quotient = result.machine
        model = Model()
        quotient.add_model(model)
        self.assertEqual(model.state, result.mapping['A'])
        model.x()
        self.assertEqual(model.state, 'C')

    def test_enumerated_observations_match(self):
        # Two states that only diverge two events ahead must remain separate,
        # but quotient traces must agree with original traces for all short words.
        machine = build_machine(['A', 'B', 'P', 'Q'], [
            {'trigger': 'x', 'source': 'A', 'dest': 'P'},
            {'trigger': 'x', 'source': 'B', 'dest': 'Q'},
            {'trigger': 'y', 'source': 'P', 'dest': 'P'},
        ])
        result = minimize_machine(machine, ['x', 'y'])
        self.assertEqual([], result.check_observations(4))
        for state in result.mapping:
            for length in range(3):
                for word in product(result.events, repeat=length):
                    self.assertEqual(result.observable_trace(state, list(word)),
                                     result.quotient_trace(state, list(word)))

    def test_distinguish_returns_event_word(self):
        machine = build_machine(['A', 'B', 'P', 'Q'], [
            {'trigger': 'x', 'source': 'A', 'dest': 'P'},
            {'trigger': 'x', 'source': 'B', 'dest': 'Q'},
            {'trigger': 'y', 'source': 'P', 'dest': 'P'},
        ])
        result = minimize_machine(machine, ['x', 'y'])
        word = result.distinguish('A', 'B')
        self.assertEqual(word, ['x', 'y'])
        self.assertNotEqual(result.observable_trace('A', word),
                            result.observable_trace('B', word))

    def test_distinguish_empty_word_for_different_labels(self):
        machine = build_machine(['A', 'B'], [
            {'trigger': 'e', 'source': 'A', 'dest': 'A'},
            {'trigger': 'e', 'source': 'B', 'dest': 'B'},
        ])
        result = minimize_machine(machine, ['e'], labels={'A': 'red', 'B': 'green'})
        self.assertEqual([], result.distinguish('A', 'B'))
        self.assertIsNone(result.distinguish('A', 'A'))

    def test_equivalent_states_have_no_distinguishing_word(self):
        machine = build_machine(['A', 'B'], [
            {'trigger': 'x', 'source': 'A', 'dest': 'B'},
            {'trigger': 'x', 'source': 'B', 'dest': 'A'},
            {'trigger': 'y', 'source': 'A', 'dest': 'A'},
            {'trigger': 'y', 'source': 'B', 'dest': 'B'},
        ])
        result = minimize_machine(machine, ['x', 'y'])
        self.assertEqual(result.mapping['A'], result.mapping['B'])
        self.assertIsNone(result.distinguish('A', 'B'))

    def test_missing_transition_is_not_a_self_loop(self):
        # S loops on 'e', T rejects on 'e'; they must not merge.
        machine = build_machine(['S', 'T'], [
            {'trigger': 'e', 'source': 'S', 'dest': 'S'},
        ])
        result = minimize_machine(machine, ['e', 'f'])
        self.assertNotEqual(result.mapping['S'], result.mapping['T'])
        word = result.distinguish('S', 'T')
        self.assertEqual(word, ['e'])
        self.assertEqual(result.observable_trace('S', ['e', 'e']),
                         [None, ((), None), ((), None)])
        self.assertEqual(result.observable_trace('T', ['e', 'e']),
                         [None, ((), REJECT), ((), REJECT)])

    def test_reject_sink_stays_rejecting(self):
        machine = build_machine(['A'], [])
        result = minimize_machine(machine, ['e'])
        self.assertEqual(result.observable_trace('A', ['e', 'e', 'e'])[1:],
                         [((), REJECT), ((), REJECT), ((), REJECT)])
        self.assertIn('rejected', result.machine.states)
        word = result.distinguish('A', None)
        # the real state label already differs from the rejection label at zero length
        self.assertEqual(word, [])
        self.assertIsNone(result.distinguish(None, None))

    def test_reject_state_can_be_disabled(self):
        machine = build_machine(['A'], [])
        result = minimize_machine(machine, ['e'], reject_state=False)
        self.assertNotIn('rejected', result.machine.states)
        self.assertEqual(result.observable_trace('A', ['e']), result.quotient_trace('A', ['e']))
        self.assertEqual([], result.check_observations(2))

    def test_explicit_reject_state_is_total(self):
        machine = build_machine(['A'], [])
        result = minimize_machine(machine, ['e'])

        class Model(object):
            pass

        model = Model()
        result.machine.add_model(model, initial='rejected')
        model.e()
        self.assertEqual(model.state, 'rejected')

    def test_internal_transition_differs_from_self_loop(self):
        # The internal transition emits action 'i'; the external self loop emits no enter/exit
        # actions (the state has none) but a transition action 's'. They must stay separate.
        machine = build_machine(['I', 'S'], [
            {'trigger': 'e', 'source': 'I', 'dest': None, 'after': 'i'},
            {'trigger': 'e', 'source': 'S', 'dest': 'S', 'after': 's'},
        ])
        result = minimize_machine(machine, ['e'])
        self.assertNotEqual(result.mapping['I'], result.mapping['S'])
        self.assertEqual(result.observable_trace('I', ['e'])[1][0], ('i',))

    def test_actionless_internal_transition_equals_self_loop(self):
        # Without callbacks the observable sequence of an internal transition and an external
        # self loop are identical, so the states may be merged.
        machine = build_machine(['I', 'S'], [
            {'trigger': 'e', 'source': 'I', 'dest': None},
            {'trigger': 'e', 'source': 'S', 'dest': 'S'},
        ])
        result = minimize_machine(machine, ['e'])
        self.assertEqual(result.mapping['I'], result.mapping['S'])
        self.assertIsNone(result.distinguish('I', 'S'))

    def test_actions_participate_by_identifier_only(self):
        states = [
            {'name': 'A', 'on_enter': ['enter_x']},
            {'name': 'B', 'on_enter': ['enter_y']},
            {'name': 'C'},
        ]
        machine = Machine(states=states, initial='A', auto_transitions=False, transitions=[
            {'trigger': 'go', 'source': 'A', 'dest': 'C', 'before': 'b1'},
            {'trigger': 'go', 'source': 'B', 'dest': 'C', 'before': 'b2'},
        ])
        result = minimize_machine(machine, ['go'], labels={'A': 0, 'B': 0, 'C': 1})
        self.assertNotEqual(result.mapping['A'], result.mapping['B'])
        self.assertEqual(result.distinguish('A', 'B'), ['go'])

        machine = build_machine(['A', 'B', 'C'], [
            {'trigger': 'go', 'source': 'A', 'dest': 'C', 'before': 'b', 'after': 'a'},
            {'trigger': 'go', 'source': 'B', 'dest': 'C', 'before': 'b', 'after': 'a'},
        ])
        result = minimize_machine(machine, ['go'], labels={'A': 0, 'B': 0, 'C': 1})
        self.assertEqual(result.mapping['A'], result.mapping['B'])

    def test_enter_exit_actions_appear_in_trace(self):
        states = [
            {'name': 'A', 'on_exit': ['ex']},
            {'name': 'B', 'on_enter': ['en']},
        ]
        machine = Machine(states=states, initial='A', auto_transitions=False, transitions=[
            {'trigger': 'go', 'source': 'A', 'dest': 'B',
             'prepare': 'p', 'before': 'bf', 'after': 'af'},
        ])
        result = minimize_machine(machine, ['go'])
        trace = result.observable_trace('A', ['go'])
        self.assertEqual(trace[1], (('p', 'bf', 'ex', 'en', 'af'), None))

    def test_raw_callables_require_action_key(self):
        machine = build_machine(['S'], [
            {'trigger': 'e', 'source': 'S', 'dest': 'S', 'before': lambda: None},
        ])
        with self.assertRaises(MinimizationError):
            minimize_machine(machine, ['e'])

        def named_action():
            pass

        machine = build_machine(['S'], [
            {'trigger': 'e', 'source': 'S', 'dest': 'S', 'before': named_action},
        ])
        result = minimize_machine(machine, ['e'], action_key=lambda callback: callback.__name__)
        self.assertEqual(result.mapping['S'], 'S')

    def test_rejects_conditional_branching(self):
        machine = build_machine(['S'], [
            {'trigger': 'e', 'source': 'S', 'dest': 'S', 'conditions': ['guard']},
        ])
        with self.assertRaises(MinimizationError):
            minimize_machine(machine, ['e'])

    def test_rejects_multiple_candidates(self):
        machine = build_machine(['S', 'T'], [
            {'trigger': 'e', 'source': 'S', 'dest': 'S'},
            {'trigger': 'e', 'source': 'S', 'dest': 'T'},
        ])
        with self.assertRaises(MinimizationError):
            minimize_machine(machine, ['e'])

    def test_rejects_unknown_trigger(self):
        machine = build_machine(['S', 'T'], [
            {'trigger': 'e', 'source': 'S', 'dest': 'T'},
        ])
        with self.assertRaises(MinimizationError):
            minimize_machine(machine, ['f'])

    def test_rejects_duplicate_alphabet_entries(self):
        machine = build_machine(['S'], [])
        with self.assertRaises(MinimizationError):
            minimize_machine(machine, ['e', 'e'])

    def test_rejects_ignore_invalid_triggers(self):
        machine = Machine(states=['S'], initial='S', auto_transitions=False,
                          ignore_invalid_triggers=True)
        with self.assertRaises(MinimizationError):
            minimize_machine(machine, ['e'])

        machine = Machine(states=[{'name': 'S', 'ignore_invalid_triggers': True}],
                          initial='S', auto_transitions=False)
        with self.assertRaises(MinimizationError):
            minimize_machine(machine, ['e'])

    def test_labels_callable_and_mapping(self):
        machine = build_machine(['S'], [])
        result = minimize_machine(machine, ['e'], labels=lambda name: 'L')
        self.assertEqual(result.labels['S'], 'L')
        with self.assertRaises(MinimizationError):
            minimize_machine(machine, ['e'], labels={})

    def test_unknown_events_in_trace(self):
        machine = build_machine(['S'], [])
        result = minimize_machine(machine, ['e'])
        with self.assertRaises(MinimizationError):
            result.observable_trace('S', ['nope'])

    def test_no_reject_state_when_machine_is_complete(self):
        machine = build_machine(['A', 'B'], [
            {'trigger': 'e', 'source': 'A', 'dest': 'B'},
            {'trigger': 'e', 'source': 'B', 'dest': 'A'},
        ])
        result = minimize_machine(machine, ['e'])
        self.assertNotIn('rejected', result.machine.states)
        self.assertEqual([], result.check_observations(3))

    def test_fuzz_random_machines(self):
        rng = random.Random(4242)
        for _ in range(40):
            states = ['s%d' % index for index in range(rng.randint(1, 6))]
            events = ['e%d' % index for index in range(rng.randint(1, 3))]
            transitions = []
            for state in states:
                for event in events:
                    if rng.random() < 0.6:
                        transitions.append({'trigger': event, 'source': state,
                                            'dest': rng.choice(states + [None])})
            machine = build_machine(states, transitions, initial=states[0])
            result = minimize_machine(machine, events)
            self.assertEqual([], result.check_observations(len(states) + 1))

            for first in list(result.mapping):
                for second in list(result.mapping) + [None]:
                    word = result.distinguish(first, second)
                    if word is None:
                        self.assertEqual(result.observable_trace(first, word or []),
                                         result.observable_trace(second, word or []))
                    else:
                        self.assertNotEqual(result.observable_trace(first, word),
                                            result.observable_trace(second, word))
