try:
    from builtins import object
except ImportError:
    pass

import threading
import time
from unittest import TestCase

from transitions import Machine, MachineError
from transitions.extensions import HierarchicalMachine, LockedMachine


class Model(object):
    pass


def transition_table(machine):
    """A comparable snapshot of all transitions currently registered on a machine."""
    return sorted(
        (name, trans.source, trans.dest)
        for name, event in machine.events.items()
        for transitions in event.transitions.values()
        for trans in transitions)


class SlowModel(object):

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def slow(self):
        self.started.set()
        self.release.wait(timeout=10)


class TestBatchMachine(TestCase):

    def setUp(self):
        self.model = Model()
        self.machine = Machine(model=self.model, states=['A', 'B', 'C'], initial='A')
        self.machine.add_transition('go', 'A', 'B')
        self.machine.add_transition('go', 'B', 'C')

    def test_batch_add_remove_replace(self):
        self.machine.modify_transitions([
            {'op': 'replace', 'trigger': 'go', 'source': 'A', 'dest': 'C'},
            {'op': 'remove', 'trigger': 'go', 'source': 'B'},
            {'op': 'add', 'trigger': 'reset', 'source': 'C', 'dest': 'A'},
        ])
        self.model.go()
        self.assertEqual(self.model.state, 'C')
        # the B -> C transition of 'go' has been removed
        with self.assertRaises(MachineError):
            self.model.go()
        self.model.reset()
        self.assertEqual(self.model.state, 'A')

    def test_batch_wildcards_and_internal_transitions(self):
        self.machine.modify_transitions([
            {'op': 'add', 'trigger': 'touch', 'source': 'A', 'dest': None},
            {'op': 'add', 'trigger': 'finish', 'source': '*', 'dest': 'C'},
        ])
        self.model.touch()
        self.assertEqual(self.model.state, 'A')
        self.model.finish()
        self.assertEqual(self.model.state, 'C')
        self.machine.modify_transitions([
            {'op': 'remove', 'trigger': 'touch', 'dest': None},
        ])
        self.assertNotIn('touch', self.machine.events)

    def test_batch_invalid_mid_batch_leaves_everything_unchanged(self):
        before = transition_table(self.machine)
        with self.assertRaises(ValueError):
            self.machine.modify_transitions([
                {'op': 'add', 'trigger': 'reset', 'source': 'C', 'dest': 'A'},
                {'op': 'add', 'trigger': 'jump', 'source': 'A', 'dest': 'UNKNOWN'},
            ])
        self.assertEqual(transition_table(self.machine), before)
        # no dynamic methods have been bound to the model
        self.assertFalse(hasattr(self.model, 'reset'))
        self.assertFalse(hasattr(self.model, 'jump'))
        # the machine still works as before
        self.model.go()
        self.assertEqual(self.model.state, 'B')

    def test_batch_invalid_source_state(self):
        with self.assertRaises(ValueError):
            self.machine.modify_transitions([
                {'op': 'add', 'trigger': 'jump', 'source': 'UNKNOWN', 'dest': 'A'},
            ])
        self.assertNotIn('jump', self.machine.events)

    def test_batch_duplicate_definitions(self):
        # duplicates of already registered transitions are rejected
        with self.assertRaises(ValueError):
            self.machine.modify_transitions([
                {'op': 'add', 'trigger': 'go', 'source': 'A', 'dest': 'B'},
            ])
        # duplicates within the same batch are rejected as well
        with self.assertRaises(ValueError):
            self.machine.modify_transitions([
                {'op': 'add', 'trigger': 'hop', 'source': 'A', 'dest': 'B'},
                {'op': 'add', 'trigger': 'hop', 'source': 'A', 'dest': 'B'},
            ])
        self.assertNotIn('hop', self.machine.events)
        self.assertFalse(hasattr(self.model, 'hop'))

    def test_batch_trigger_name_conflicts(self):
        self.model.blocked = "occupied"
        with self.assertRaises(ValueError):
            self.machine.modify_transitions([
                {'op': 'add', 'trigger': 'blocked', 'source': 'A', 'dest': 'B'},
            ])
        self.assertNotIn('blocked', self.machine.events)
        self.assertEqual(self.model.blocked, "occupied")
        # trigger names must not shadow the model attribute
        with self.assertRaises(ValueError):
            self.machine.modify_transitions([
                {'op': 'add', 'trigger': 'state', 'source': 'A', 'dest': 'B'},
            ])

    def test_batch_remove_and_replace_require_existing_transitions(self):
        before = transition_table(self.machine)
        with self.assertRaises(ValueError):
            self.machine.modify_transitions([
                {'op': 'remove', 'trigger': 'unknown'},
            ])
        with self.assertRaises(ValueError):
            self.machine.modify_transitions([
                {'op': 'remove', 'trigger': 'go', 'source': 'C'},
            ])
        with self.assertRaises(ValueError):
            self.machine.modify_transitions([
                {'op': 'replace', 'trigger': 'unknown', 'source': 'A', 'dest': 'B'},
            ])
        self.assertEqual(transition_table(self.machine), before)

    def test_batch_malformed_operations(self):
        with self.assertRaises(ValueError):
            self.machine.modify_transitions([{'op': 'explode', 'trigger': 'go'}])
        with self.assertRaises(ValueError):
            self.machine.modify_transitions([{'op': 'add', 'trigger': 'x', 'source': 'A'}])
        with self.assertRaises(ValueError):
            self.machine.modify_transitions([{'op': 'add', 'trigger': 'x', 'dest': 'A'}])
        with self.assertRaises(ValueError):
            self.machine.modify_transitions(['not a dict'])

    def test_batch_remove_last_trigger_of_same_name(self):
        self.machine.modify_transitions([
            {'op': 'remove', 'trigger': 'go', 'source': 'A'},
        ])
        # B -> C is left, so the trigger and its model methods stay
        self.assertIn('go', self.machine.events)
        self.assertTrue(hasattr(self.model, 'go'))
        self.machine.modify_transitions([
            {'op': 'remove', 'trigger': 'go'},
        ])
        # the last transition of 'go' has been removed; the trigger is gone entirely
        self.assertNotIn('go', self.machine.events)
        self.assertFalse(hasattr(self.model, 'go'))
        self.assertFalse(hasattr(self.model, 'may_go'))
        with self.assertRaises(AttributeError):
            self.model.go()
        # re-adding a trigger with the same name works again
        self.machine.modify_transitions([
            {'op': 'add', 'trigger': 'go', 'source': 'A', 'dest': 'C'},
        ])
        self.assertTrue(hasattr(self.model, 'go'))
        self.model.go()
        self.assertEqual(self.model.state, 'C')

    def test_batch_does_not_execute_user_callbacks(self):
        calls = []
        self.model.check = lambda: calls.append('check') or True
        self.machine.modify_transitions([
            {'op': 'add', 'trigger': 'advance', 'source': 'A', 'dest': 'C',
             'conditions': 'check', 'before': 'check', 'after': 'check'},
        ])
        # neither validation nor commit executed the callbacks
        self.assertEqual(calls, [])
        self.model.advance()
        self.assertEqual(calls, ['check', 'check', 'check'])
        self.assertEqual(self.model.state, 'C')


class TestBatchLockedMachine(TestCase):

    def test_commit_is_serialized_with_running_events(self):
        model = SlowModel()
        machine = LockedMachine(model=model, states=['A', 'B', 'C'], initial='A')
        machine.add_transition('go', 'A', 'B', before='slow')

        thread = threading.Thread(target=model.go)
        thread.start()
        # wait until the running event executes its (blocking) callback
        self.assertTrue(model.started.wait(timeout=5))

        commit_thread = threading.Thread(
            target=machine.modify_transitions,
            args=([{'op': 'replace', 'trigger': 'go', 'source': 'A', 'dest': 'C'}],))
        commit_thread.start()
        time.sleep(0.1)
        # the running event holds the machine context, so the commit has to wait
        self.assertTrue(commit_thread.is_alive())
        self.assertEqual(model.state, 'A')
        model.release.set()
        thread.join(timeout=5)
        commit_thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(commit_thread.is_alive())
        # the running event used the consistent rule set from its start
        self.assertEqual(model.state, 'B')
        # events triggered after the commit use the new rules
        model.to_A()
        model.go()
        self.assertEqual(model.state, 'C')


class TestBatchHierarchicalMachine(TestCase):

    def setUp(self):
        self.model = Model()
        self.machine = HierarchicalMachine(
            model=self.model,
            states=['A', {'name': 'B', 'children': ['1', '2'], 'initial': '1'}],
            initial='A')

    def test_batch_nested_paths_and_inherited_parent_rules(self):
        self.machine.modify_transitions([
            {'op': 'add', 'trigger': 'go', 'source': 'B', 'dest': 'A'},
            {'op': 'add', 'trigger': 'enter', 'source': 'A', 'dest': 'B_2'},
        ])
        self.model.enter()
        self.assertEqual(self.model.state, 'B_2')
        # the transition defined for parent state 'B' is inherited by substate 'B_2'
        self.model.go()
        self.assertEqual(self.model.state, 'A')
        # removing the parent rule removes it for substates as well
        self.machine.modify_transitions([
            {'op': 'remove', 'trigger': 'go'},
        ])
        self.assertNotIn('go', self.machine.events)
        self.assertFalse(hasattr(self.model, 'go'))
        self.model.enter()
        self.assertEqual(self.model.state, 'B_2')
        with self.assertRaises(AttributeError):
            self.model.go()

    def test_batch_replace_nested_transition(self):
        self.machine.modify_transitions([
            {'op': 'add', 'trigger': 'go', 'source': 'B_1', 'dest': 'B_2'},
        ])
        self.model.to_B()
        self.assertEqual(self.model.state, 'B_1')
        self.machine.modify_transitions([
            {'op': 'replace', 'trigger': 'go', 'source': 'B_1', 'dest': 'A'},
        ])
        self.model.go()
        self.assertEqual(self.model.state, 'A')

    def test_batch_invalid_nested_path_rolls_back(self):
        before = transition_table(self.machine)
        with self.assertRaises(ValueError):
            self.machine.modify_transitions([
                {'op': 'add', 'trigger': 'go', 'source': 'B', 'dest': 'A'},
                {'op': 'add', 'trigger': 'jump', 'source': 'B_3', 'dest': 'A'},
            ])
        self.assertEqual(transition_table(self.machine), before)
        self.assertFalse(hasattr(self.model, 'go'))
        self.assertFalse(hasattr(self.model, 'jump'))
        with self.assertRaises(ValueError):
            self.machine.modify_transitions([
                {'op': 'add', 'trigger': 'go', 'source': 'A', 'dest': 'B_3'},
            ])
        self.assertEqual(transition_table(self.machine), before)
