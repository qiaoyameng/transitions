try:
    from builtins import object
except ImportError:
    pass

import threading
import time

from unittest import TestCase

from transitions import Machine, MachineError
from transitions.extensions import HierarchicalMachine, LockedMachine
from transitions.extensions.factory import LockedHierarchicalMachine


class Model(object):

    def __init__(self):
        self.message = None


class TestBatchUpdate(TestCase):

    def setUp(self):
        self.machine = Machine(states=['A', 'B', 'C'], initial='A')

    def test_add_replace_remove(self):
        self.machine.add_transition('go', 'A', 'B')
        self.machine.update_transitions([
            {'op': 'replace', 'trigger': 'go', 'source': 'A', 'dest': 'C'},
            {'op': 'add', 'trigger': 'go', 'source': 'C', 'dest': 'B'},
            {'op': 'add', 'trigger': 'back', 'source': 'B', 'dest': 'A'},
        ])
        self.machine.go()
        self.assertEqual('C', self.machine.state)
        self.machine.go()
        self.assertEqual('B', self.machine.state)
        self.machine.back()
        self.assertEqual('A', self.machine.state)
        # remove a single transition of a multi-transition trigger
        self.machine.update_transitions([
            {'op': 'remove', 'trigger': 'go', 'source': 'C', 'dest': 'B'},
        ])
        self.assertEqual([('A', 'C')], [(t.source, t.dest) for t in self.machine.get_transitions('go')])

    def test_invalid_batch_leaves_machine_unchanged(self):
        self.machine.add_transition('go', 'A', 'B')
        events_before = self.machine.events
        bound_go = self.machine.go
        transitions_before = [(t.source, t.dest) for t in self.machine.get_transitions()]
        with self.assertRaises(ValueError):
            self.machine.update_transitions([
                {'op': 'add', 'trigger': 'new_trigger', 'source': 'B', 'dest': 'C'},
                {'op': 'add', 'trigger': 'other', 'source': 'A', 'dest': 'UNKNOWN'},
            ])
        # transition table and bound methods are untouched
        self.assertIs(events_before, self.machine.events)
        self.assertIs(bound_go, self.machine.go)
        self.assertEqual(transitions_before, [(t.source, t.dest) for t in self.machine.get_transitions()])
        self.assertFalse(hasattr(self.machine, 'new_trigger'))
        self.assertFalse(hasattr(self.machine, 'other'))
        # machine is still functional
        self.machine.go()
        self.assertEqual('B', self.machine.state)

    def test_invalid_operations(self):
        transitions_before = self.machine.get_transitions()
        with self.assertRaises(ValueError):
            self.machine.update_transitions([{'op': 'fly', 'trigger': 'go', 'source': 'A', 'dest': 'B'}])
        with self.assertRaises(ValueError):
            self.machine.update_transitions([{'op': 'add', 'source': 'A', 'dest': 'B'}])
        with self.assertRaises(ValueError):
            self.machine.update_transitions([{'op': 'add', 'trigger': 'go', 'dest': 'B'}])
        with self.assertRaises(ValueError):
            self.machine.update_transitions([{'op': 'remove', 'trigger': 'go', 'conditions': 'x'}])
        with self.assertRaises(ValueError):
            self.machine.update_transitions(['not a dict'])
        self.assertEqual(transitions_before, self.machine.get_transitions())

    def test_unknown_states_rejected(self):
        transitions_before = self.machine.get_transitions()
        with self.assertRaises(ValueError):
            self.machine.update_transitions([{'op': 'add', 'trigger': 'go', 'source': 'X', 'dest': 'A'}])
        with self.assertRaises(ValueError):
            self.machine.update_transitions([{'op': 'add', 'trigger': 'go', 'source': 'A', 'dest': 'X'}])
        self.assertEqual(transitions_before, self.machine.get_transitions())

    def test_duplicate_definitions_rejected(self):
        self.machine.add_transition('go', 'A', 'B')
        # duplicate of an existing transition
        with self.assertRaises(ValueError):
            self.machine.update_transitions([{'op': 'add', 'trigger': 'go', 'source': 'A', 'dest': 'B'}])
        # duplicate within the same batch
        with self.assertRaises(ValueError):
            self.machine.update_transitions([
                {'op': 'add', 'trigger': 'new', 'source': 'A', 'dest': 'B'},
                {'op': 'add', 'trigger': 'new', 'source': 'A', 'dest': 'B'},
            ])
        self.assertNotIn('new', self.machine.events)
        self.assertEqual([('A', 'B')], [(t.source, t.dest) for t in self.machine.get_transitions('go')])

    def test_replace_and_remove_require_match(self):
        with self.assertRaises(ValueError):
            self.machine.update_transitions([{'op': 'replace', 'trigger': 'go', 'source': 'A', 'dest': 'B'}])
        self.machine.add_transition('go', 'A', 'B')
        with self.assertRaises(ValueError):
            self.machine.update_transitions([{'op': 'remove', 'trigger': 'go', 'source': 'B'}])
        with self.assertRaises(ValueError):
            self.machine.update_transitions([{'op': 'remove', 'trigger': 'unknown'}])
        self.assertEqual([('A', 'B')], [(t.source, t.dest) for t in self.machine.get_transitions('go')])

    def test_trigger_name_conflicts(self):
        model = Model()
        machine = Machine(model, states=['A', 'B'], initial='A')
        transitions_before = machine.get_transitions()
        # conflicts with dynamically generated methods
        for trigger in ['is_A', 'on_enter_A', 'on_exit_B', 'may_to_A', 'before_to_A', 'trigger']:
            with self.assertRaises(ValueError, msg=trigger):
                machine.update_transitions([{'op': 'add', 'trigger': trigger, 'source': 'A', 'dest': 'B'}])
        # conflict with the model attribute
        with self.assertRaises(ValueError):
            machine.update_transitions([{'op': 'add', 'trigger': 'state', 'source': 'A', 'dest': 'B'}])
        # conflict with an existing model attribute
        model.jump = lambda: None
        with self.assertRaises(ValueError):
            machine.update_transitions([{'op': 'add', 'trigger': 'jump', 'source': 'A', 'dest': 'B'}])
        self.assertEqual(transitions_before, machine.get_transitions())
        self.assertTrue(model.is_A())  # sanity: dynamic state methods still bound

    def test_remove_last_transition_removes_trigger(self):
        self.machine.add_transition('go', 'A', 'B')
        self.machine.update_transitions([{'op': 'remove', 'trigger': 'go'}])
        self.assertNotIn('go', self.machine.events)
        self.assertFalse(hasattr(self.machine, 'go'))
        with self.assertRaises(AttributeError):
            self.machine.trigger('go')

    def test_remove_keeps_trigger_with_remaining_transitions(self):
        self.machine.add_transition('go', 'A', 'B')
        self.machine.add_transition('go', 'B', 'C')
        self.machine.update_transitions([{'op': 'remove', 'trigger': 'go', 'source': 'A'}])
        self.assertIn('go', self.machine.events)
        self.assertTrue(hasattr(self.machine, 'go'))
        self.assertEqual([('B', 'C')], [(t.source, t.dest) for t in self.machine.get_transitions('go')])

    def test_internal_and_reflexive_transitions(self):
        self.machine.update_transitions([
            {'op': 'add', 'trigger': 'ping', 'source': 'A', 'dest': None},
            {'op': 'add', 'trigger': 'bump', 'source': ['A', 'B'], 'dest': '='},
        ])
        self.assertTrue(self.machine.ping())
        self.assertEqual('A', self.machine.state)
        self.machine.bump()
        self.assertEqual('A', self.machine.state)
        self.machine.update_transitions([{'op': 'remove', 'trigger': 'ping', 'source': 'A', 'dest': None}])
        self.assertNotIn('ping', self.machine.events)

    def test_running_event_uses_initial_rule_set(self):
        machine = Machine(states=['A', 'B', 'C'], initial='A')
        machine.add_transition('go', 'A', 'B', before='apply_batch')
        machine.apply_batch = lambda: machine.update_transitions(
            [{'op': 'replace', 'trigger': 'go', 'source': 'A', 'dest': 'C'}])
        machine.go()
        # the batch was committed by a callback of the running event; the event itself
        # must still complete with the rule set valid at its start
        self.assertEqual('B', machine.state)
        machine.to_A()
        machine.go()
        # events triggered after the commit use the new rules
        self.assertEqual('C', machine.state)

    def test_multiple_models(self):
        model_a, model_b = Model(), Model()
        machine = Machine(model=[model_a, model_b], states=['A', 'B'], initial='A')
        machine.update_transitions([{'op': 'add', 'trigger': 'go', 'source': 'A', 'dest': 'B'}])
        model_a.go()
        self.assertTrue(model_a.is_B())
        self.assertTrue(model_b.is_A())
        machine.update_transitions([{'op': 'remove', 'trigger': 'go'}])
        self.assertFalse(hasattr(model_a, 'go'))
        self.assertFalse(hasattr(model_b, 'go'))


class TestBatchUpdateHierarchical(TestCase):

    def setUp(self):
        self.machine = HierarchicalMachine(
            states=['A', {'name': 'P', 'children': ['1', '2'], 'initial': '1'}, 'C'], initial='A')

    def test_add_with_nested_paths(self):
        self.machine.update_transitions([
            {'op': 'add', 'trigger': 'enter', 'source': 'A', 'dest': 'P'},
            {'op': 'add', 'trigger': 'swap', 'source': 'P_1', 'dest': 'P_2'},
        ])
        self.machine.enter()
        self.assertEqual('P_1', self.machine.state)
        self.machine.swap()
        self.assertEqual('P_2', self.machine.state)

    def test_inherited_state_paths(self):
        self.machine.update_transitions([
            {'op': 'add', 'trigger': 'enter', 'source': 'A', 'dest': 'P'},
            # defined for the parent state and inherited by its children
            {'op': 'add', 'trigger': 'leave', 'source': 'P', 'dest': 'C'},
        ])
        self.machine.enter()
        self.assertEqual('P_1', self.machine.state)
        self.machine.leave()
        self.assertEqual('C', self.machine.state)
        # remove the inherited transition again via the parent path
        self.machine.update_transitions([{'op': 'remove', 'trigger': 'leave', 'source': 'P'}])
        self.assertNotIn('leave', self.machine.events)
        self.assertFalse(hasattr(self.machine, 'leave'))
        self.machine.to_P()
        self.assertEqual('P_1', self.machine.state)

    def test_invalid_path_leaves_machine_unchanged(self):
        events_before = self.machine.events
        transitions_before = self.machine.get_transitions()
        with self.assertRaises(ValueError):
            self.machine.update_transitions([
                {'op': 'add', 'trigger': 'enter', 'source': 'A', 'dest': 'P'},
                {'op': 'add', 'trigger': 'swap', 'source': 'P_1', 'dest': 'P_3'},
            ])
        self.assertIs(events_before, self.machine.events)
        self.assertEqual(transitions_before, self.machine.get_transitions())
        self.assertFalse(hasattr(self.machine, 'enter'))

    def test_remove_last_trigger_hierarchical(self):
        self.machine.update_transitions([{'op': 'add', 'trigger': 'go', 'source': 'A', 'dest': 'C'}])
        self.assertTrue(hasattr(self.machine, 'go'))
        self.machine.update_transitions([{'op': 'remove', 'trigger': 'go'}])
        self.assertNotIn('go', self.machine.events)
        self.assertFalse(hasattr(self.machine, 'go'))

    def test_replace_hierarchical(self):
        self.machine.update_transitions([{'op': 'add', 'trigger': 'go', 'source': 'A', 'dest': 'P_1'}])
        self.machine.update_transitions([{'op': 'replace', 'trigger': 'go', 'source': 'A', 'dest': 'P_2'}])
        self.machine.go()
        self.assertEqual('P_2', self.machine.state)

    def test_root_scope_required(self):
        with self.machine('P'):
            with self.assertRaises(MachineError):
                self.machine.update_transitions([{'op': 'add', 'trigger': 'go', 'source': 'A', 'dest': 'C'}])

    def test_locked_hierarchical_machine(self):
        machine = LockedHierarchicalMachine(
            states=['A', {'name': 'P', 'children': ['1', '2'], 'initial': '1'}], initial='A')
        machine.update_transitions([
            {'op': 'add', 'trigger': 'enter', 'source': 'A', 'dest': 'P'},
            {'op': 'add', 'trigger': 'leave', 'source': 'P', 'dest': 'A'},
        ])
        machine.enter()
        self.assertEqual('P_1', machine.state)
        machine.leave()
        self.assertEqual('A', machine.state)
        machine.update_transitions([{'op': 'remove', 'trigger': 'leave'}])
        self.assertNotIn('leave', machine.events)
        self.assertFalse(hasattr(machine, 'leave'))


class TestBatchUpdateLocked(TestCase):

    def test_running_event_uses_initial_rule_set_locked(self):
        machine = LockedMachine(states=['A', 'B', 'C'], initial='A')
        machine.add_transition('go', 'A', 'B', before='apply_batch')
        machine.apply_batch = lambda: machine.update_transitions(
            [{'op': 'replace', 'trigger': 'go', 'source': 'A', 'dest': 'C'}])
        machine.go()
        self.assertEqual('B', machine.state)
        machine.to_A()
        machine.go()
        self.assertEqual('C', machine.state)

    def test_update_is_serialized_with_running_events(self):
        machine = LockedMachine(states=['A', 'B'], initial='A')
        processed = []

        def slow_callback():
            time.sleep(0.2)
            processed.append(True)

        machine.add_transition('go', 'A', 'B', before=slow_callback)
        thread = threading.Thread(target=machine.go)
        thread.start()
        time.sleep(0.05)  # give the event time to start
        machine.update_transitions([{'op': 'add', 'trigger': 'back', 'source': 'B', 'dest': 'A'}])
        # the commit is serialized with event processing; the running event (including
        # its user callbacks) must have finished before the commit returns
        self.assertTrue(processed)
        thread.join()
        self.assertEqual('B', machine.state)
        machine.back()
        self.assertEqual('A', machine.state)

    def test_concurrent_triggers_and_updates(self):
        machine = LockedMachine(states=['A', 'B'], initial='A', ignore_invalid_triggers=True)
        machine.add_transition('go', 'A', 'B')
        machine.add_transition('back', 'B', 'A')
        errors = []

        def hammer():
            for _ in range(100):
                try:
                    machine.go()
                    machine.back()
                except Exception as err:  # pylint: disable=broad-except
                    errors.append(err)

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for thread in threads:
            thread.start()
        for i in range(50):
            if i % 2 == 0:
                machine.update_transitions([{'op': 'add', 'trigger': 'ping', 'source': 'A', 'dest': None}])
            else:
                machine.update_transitions([{'op': 'remove', 'trigger': 'ping'}])
        for thread in threads:
            thread.join()
        self.assertEqual([], errors)
        self.assertIn(machine.state, ('A', 'B'))
        # the final commit removed the trigger completely
        self.assertNotIn('ping', machine.events)
        self.assertFalse(hasattr(machine, 'ping'))
