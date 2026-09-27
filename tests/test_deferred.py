from __future__ import absolute_import

from unittest import TestCase

from transitions import Machine, MachineError
from transitions.extensions import HierarchicalMachine


class TestDeferred(TestCase):

    def setUp(self):
        self.log = []

    def create_machine(self, **kwargs):
        states = [{'name': 'calibrating', 'defer': ['configure', 'step']},
                  'idle', 'ready', 'done']
        machine = Machine(states=states, initial='calibrating', **kwargs)
        machine.add_transition('finish', 'calibrating', 'idle')
        machine.add_transition('configure', 'idle', 'ready',
                               after=lambda *args, **kwargs: self.log.append('configure'))
        machine.add_transition('step', 'ready', 'done',
                               after=lambda *args, **kwargs: self.log.append('step'))
        return machine

    def test_defer_and_replay_in_submission_order(self):
        machine = self.create_machine()
        self.assertTrue(machine.configure())  # deferred, not executed yet
        self.assertTrue(machine.step())       # deferred as well
        self.assertEqual(machine.state, 'calibrating')
        self.assertEqual(self.log, [])

        self.assertTrue(machine.finish())
        # 'step' is only valid from 'ready'; reaching 'done' proves submission order
        self.assertEqual(machine.state, 'done')
        self.assertEqual(self.log, ['configure', 'step'])
        self.assertEqual(len(machine._deferred_queue), 0)

    def test_replayed_event_changes_state_again(self):
        machine = self.create_machine()
        machine.configure()
        machine.finish()
        # the replayed 'configure' event moved the machine to 'ready'
        self.assertEqual(machine.state, 'ready')
        self.assertEqual(self.log, ['configure'])

    def test_deferred_arguments_are_preserved(self):
        received = []
        machine = Machine(states=[{'name': 'calibrating', 'defer': ['configure']}, 'idle'],
                          initial='calibrating')
        machine.add_transition('finish', 'calibrating', 'idle')
        machine.add_transition('configure', 'idle', 'calibrating',
                               after=lambda *args, **kwargs: received.append((args, kwargs)))
        machine.configure('fast', mode=2)
        machine.finish()
        self.assertEqual(received, [(('fast',), {'mode': 2})])

    def test_still_deferred_events_are_not_replayed_in_a_loop(self):
        machine = Machine(states=[{'name': 'A', 'defer': ['e']},
                                  {'name': 'B', 'defer': ['e']}, 'C'],
                          initial='A')
        machine.add_transition('go', 'A', 'B')
        machine.add_transition('e', 'C', 'A')
        machine.e()
        machine.go()  # 'e' is still deferred in 'B' and must stay queued
        self.assertEqual(machine.state, 'B')
        self.assertEqual(len(machine._deferred_queue), 1)
        machine.to_C()  # now 'e' can be processed
        self.assertEqual(machine.state, 'A')
        self.assertEqual(len(machine._deferred_queue), 0)

    def test_max_deferred_overflow_discards_oldest(self):
        machine = Machine(states=[{'name': 'calibrating',
                                   'defer': ['first', 'second', 'third']}, 'idle'],
                          initial='calibrating', max_deferred=2)
        machine.add_transition('finish', 'calibrating', 'idle')
        for trigger in ['first', 'second', 'third']:
            machine.add_transition(trigger, 'idle', None,
                                   after=lambda *a, **k: self.log.append('x'))
        machine.first()
        machine.second()
        with self.assertLogs('transitions.core', level='WARNING') as cm:
            machine.third()
        self.assertTrue(any('first' in line for line in cm.output))
        self.assertEqual(len(machine._deferred_queue), 2)
        machine.finish()
        # 'first' has been discarded; only 'second' and 'third' are replayed
        self.assertEqual(len(self.log), 2)

    def test_max_deferred_must_be_positive(self):
        with self.assertRaises(ValueError):
            Machine(states=['A'], initial='A', max_deferred=0)

    def test_multiple_models(self):
        class Model:
            pass

        model_a, model_b = Model(), Model()
        machine = self.create_machine(model=None)
        machine.add_model(model_a)
        machine.add_model(model_b)

        model_a.configure()
        model_b.configure()
        self.assertEqual(len(machine._deferred_queue), 2)

        model_a.finish()
        self.assertEqual(model_a.state, 'ready')
        self.assertEqual(model_b.state, 'calibrating')
        self.assertEqual(len(machine._deferred_queue), 1)

        model_b.finish()
        self.assertEqual(model_b.state, 'ready')
        self.assertEqual(len(machine._deferred_queue), 0)

    def test_remove_model_purges_deferred_events(self):
        class Model:
            pass

        model_a, model_b = Model(), Model()
        machine = self.create_machine(model=None)
        machine.add_model(model_a)
        machine.add_model(model_b)

        model_a.configure()
        model_b.configure()
        machine.remove_model(model_a)
        self.assertEqual(len(machine._deferred_queue), 1)
        self.assertIs(machine._deferred_queue[0][1].model, model_b)

        model_b.finish()
        self.assertEqual(model_b.state, 'ready')
        self.assertEqual(len(machine._deferred_queue), 0)

    def test_deferred_queue_is_separate_from_transition_queue(self):
        machine = self.create_machine(queued=True)
        machine.configure()
        self.assertEqual(len(machine._deferred_queue), 1)
        self.assertEqual(len(machine._transition_queue), 0)
        machine.finish()
        self.assertEqual(machine.state, 'ready')
        self.assertEqual(len(machine._deferred_queue), 0)

    def test_replayed_invalid_event_raises(self):
        machine = Machine(states=[{'name': 'calibrating', 'defer': ['configure']}, 'idle'],
                          initial='calibrating')
        machine.add_transition('finish', 'calibrating', 'idle')
        machine.add_transition('configure', 'ready', 'done')  # not valid from 'idle'
        machine.configure()
        with self.assertRaises(MachineError):
            machine.finish()


class TestDeferredHierarchical(TestCase):

    def setUp(self):
        self.log = []

    def test_parent_declaration_defers_for_children(self):
        machine = HierarchicalMachine(
            states=[{'name': 'cal', 'children': ['a', 'b'], 'initial': 'a', 'defer': ['go']},
                    'idle'], initial='cal')
        machine.add_transition('go', 'idle', 'cal',
                               after=lambda *a, **k: self.log.append('go'))
        machine.add_transition('finish', 'cal', 'idle')
        self.assertEqual(machine.state, 'cal_a')
        self.assertTrue(machine.go())  # deferred since parent 'cal' declares it
        self.assertEqual(machine.state, 'cal_a')
        machine.finish()
        self.assertEqual(machine.state, 'cal_a')  # replayed: idle -> cal
        self.assertEqual(self.log, ['go'])

    def test_child_declaration_defers_independently(self):
        machine = HierarchicalMachine(
            states=[{'name': 'cal', 'children': [{'name': 'a', 'defer': ['go']}, 'b'],
                     'initial': 'a'},
                    'idle'], initial='cal')
        machine.add_transition('go', 'cal_b', 'cal_a',
                               after=lambda *a, **k: self.log.append('go'))
        machine.add_transition('step', 'cal_a', 'cal_b')
        machine.go()  # deferred: child 'a' declares it even though parent does not
        self.assertEqual(machine.state, 'cal_a')
        self.assertEqual(self.log, [])
        machine.step()  # sibling 'b' does not defer 'go' -> replay
        self.assertEqual(machine.state, 'cal_a')
        self.assertEqual(self.log, ['go'])

    def test_parallel_regions_defer_until_no_region_requires_it(self):
        machine = HierarchicalMachine(
            states=[{'name': 'P', 'parallel': [
                {'name': '1', 'children': [{'name': 'a', 'defer': ['tick']}, 'b'], 'initial': 'a'},
                {'name': '2', 'children': [{'name': 'x', 'defer': ['tick']}, 'y'], 'initial': 'x'}]},
                'idle'], initial='P')
        machine.add_transition('tick', 'P', None,
                               after=lambda *a, **k: self.log.append('tick'))
        machine.add_transition('advance1', 'P_1_a', 'P_1_b')
        machine.add_transition('advance2', 'P_2_x', 'P_2_y')

        machine.tick()
        self.assertEqual(machine.state, ['P_1_a', 'P_2_x'])
        self.assertEqual(len(machine._deferred_queue), 1)

        machine.advance1()  # region 1 no longer defers but region 2 still does
        self.assertEqual(machine.state, ['P_1_b', 'P_2_x'])
        self.assertEqual(len(machine._deferred_queue), 1)
        self.assertEqual(self.log, [])

        machine.advance2()  # no region defers 'tick' anymore -> replay
        self.assertEqual(machine.state, ['P_1_b', 'P_2_y'])
        self.assertEqual(self.log, ['tick'])
        self.assertEqual(len(machine._deferred_queue), 0)
