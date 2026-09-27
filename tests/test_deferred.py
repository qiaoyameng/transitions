try:
    from builtins import object
except ImportError:
    pass

from unittest import TestCase

from transitions import Machine, MachineError
from transitions.extensions import HierarchicalMachine


class Collector(object):

    def __init__(self):
        self.log = []

    def record(self, *args, **kwargs):
        self.log.append((args, kwargs))


class TestDeferredCore(TestCase):

    def create_machine(self, model, **kwargs):
        machine_kwargs = dict(states=[{'name': 'calibrating', 'deferred': ['ping']}, 'ready', 'done'],
                              initial='calibrating', auto_transitions=False)
        machine_kwargs.update(kwargs)
        machine = Machine(model, **machine_kwargs)
        machine.add_transition('ping', 'ready', 'ready', after='record')
        machine.add_transition('finish', 'calibrating', 'ready')
        return machine

    def test_defer_and_replay_in_submission_order(self):
        model = Collector()
        machine = self.create_machine(model)
        self.assertTrue(model.ping(1))
        self.assertTrue(model.ping(2))
        self.assertTrue(model.ping(3))
        # events are deferred: no state change, no callback execution
        self.assertEqual('calibrating', model.state)
        self.assertEqual([], model.log)
        model.finish()
        self.assertEqual('ready', model.state)
        # deferred events are replayed in submission order after the state has been left
        self.assertEqual([((1,), {}), ((2,), {}), ((3,), {})], model.log)
        self.assertFalse(machine._deferred_queues[model])

    def test_deferred_arguments_kept_at_submission_time(self):
        model = Collector()
        self.create_machine(model)
        model.ping(1, mode='fast')
        model.ping(2, mode='slow')
        model.finish()
        self.assertEqual([((1,), {'mode': 'fast'}), ((2,), {'mode': 'slow'})], model.log)

    def test_replayed_event_changes_state(self):
        model = Collector()
        machine = Machine(model, states=[{'name': 'calibrating', 'deferred': ['advance']},
                                         'stage1', 'stage2'],
                          initial='calibrating', auto_transitions=False)
        machine.add_transition('advance', 'stage1', 'stage2')
        machine.add_transition('finish', 'calibrating', 'stage1')
        model.advance()
        self.assertEqual('calibrating', model.state)
        model.finish()
        # the deferred event is replayed after 'finish' and changes the state again
        self.assertEqual('stage2', model.state)

    def test_still_deferred_events_are_not_requeued(self):
        model = Collector()
        machine = Machine(model, states=[{'name': 'A', 'deferred': ['e1', 'e2']},
                                         {'name': 'B', 'deferred': ['e2']}, 'C'],
                          initial='A', auto_transitions=False)
        machine.add_transition('e1', 'B', 'C', after='record')
        machine.add_transition('e2', 'C', 'C', after='record')
        machine.add_transition('go', 'A', 'B')
        model.e1('first')
        model.e2('second')
        model.go()
        # e1 is no longer deferred in B and moves the model to C;
        # e2 is still deferred in B and must not be re-queued (which would loop) ...
        self.assertEqual('C', model.state)
        # ... but becomes ready once e1 moved the model to C
        self.assertEqual([(('first',), {}), (('second',), {})], model.log)

    def test_event_stays_deferred_when_state_still_defers(self):
        model = Collector()
        machine = Machine(model, states=[{'name': 'A', 'deferred': ['ping']},
                                         {'name': 'B', 'deferred': ['ping']}, 'C'],
                          initial='A', auto_transitions=False)
        machine.add_transition('ping', 'C', 'C', after='record')
        machine.add_transition('go', 'A', 'B')
        machine.add_transition('leave', 'B', 'C')
        model.ping(1)
        model.go()
        # B still defers 'ping': the event remains stored and is not processed
        self.assertEqual('B', model.state)
        self.assertEqual([], model.log)
        self.assertEqual(1, len(machine._deferred_queues[model]))
        model.leave()
        self.assertEqual('C', model.state)
        self.assertEqual([((1,), {})], model.log)

    def test_max_deferred_drops_oldest(self):
        model = Collector()
        machine = self.create_machine(model, max_deferred=2)
        model.ping(1)
        model.ping(2)
        model.ping(3)
        self.assertEqual(2, len(machine._deferred_queues[model]))
        with self.assertLogs('transitions.core', level='WARNING'):
            model.ping(4)
        model.finish()
        # the oldest events have been dropped to satisfy max_deferred
        self.assertEqual([((3,), {}), ((4,), {})], model.log)

    def test_deferred_trigger_returns_true(self):
        model = Collector()
        self.create_machine(model)
        self.assertTrue(model.ping())
        self.assertTrue(model.finish())

    def test_multiple_models(self):
        model_a = Collector()
        model_b = Collector()
        machine = Machine([model_a, model_b],
                          states=[{'name': 'calibrating', 'deferred': ['ping']}, 'ready'],
                          initial='calibrating', auto_transitions=False)
        machine.add_transition('ping', 'ready', 'ready', after='record')
        machine.add_transition('finish', 'calibrating', 'ready')
        # only model_a is in the deferring state; model_b is set to 'ready' directly
        machine.set_state('ready', model=model_b)
        model_a.ping('a')
        model_b.ping('b')
        # model_a's event is deferred while model_b's event is processed right away
        self.assertEqual([], model_a.log)
        self.assertEqual([(('b',), {})], model_b.log)
        model_a.finish()
        self.assertEqual([(('a',), {})], model_a.log)

    def test_remove_model_clears_deferred_events(self):
        model_a = Collector()
        model_b = Collector()
        machine = Machine([model_a, model_b],
                          states=[{'name': 'calibrating', 'deferred': ['ping']}, 'ready'],
                          initial='calibrating', auto_transitions=False)
        machine.add_transition('ping', 'ready', 'ready', after='record')
        model_a.ping('a')
        model_b.ping('b')
        self.assertEqual(2, len(machine._deferred_queues))
        machine.remove_model(model_a)
        # deferred events of the removed model are discarded; other models are unaffected
        self.assertNotIn(model_a, machine._deferred_queues)
        self.assertEqual(1, len(machine._deferred_queues[model_b]))

    def test_queued_machine_separate_queues(self):
        model = Collector()
        machine = Machine(model, states=[{'name': 'A', 'deferred': ['tick']}, 'B'],
                          initial='A', queued=True, auto_transitions=False)
        machine.add_transition('tick', 'B', 'B', after='record')
        machine.add_transition('finish', 'A', 'B', after=lambda: model.tick('queued'))
        model.tick('deferred')
        # deferred events are stored separately from the (empty) transition queue
        self.assertEqual(1, len(machine._deferred_queues[model]))
        self.assertEqual(0, len(machine._transition_queue))
        model.finish()
        self.assertEqual('B', model.state)
        # events queued during the transition are processed before deferred events are re-submitted
        self.assertEqual([(('queued',), {}), (('deferred',), {})], model.log)
        self.assertEqual(0, len(machine._transition_queue))
        self.assertFalse(machine._deferred_queues[model])

    def test_deferred_invalid_after_replay_raises(self):
        model = Collector()
        machine = Machine(model, states=[{'name': 'A', 'deferred': ['ping']}, 'B'],
                          initial='A', auto_transitions=False)
        machine.add_transition('ping', 'A', 'A')  # never executed since 'A' defers 'ping'
        machine.add_transition('go', 'A', 'B')
        model.ping()
        # 'ping' is not deferred in B anymore and no valid transition exists
        with self.assertRaises(MachineError):
            model.go()


class TestDeferredNested(TestCase):

    def create_machine(self, model, **kwargs):
        states = [
            {'name': 'calib', 'deferred': ['tick', 'tock'], 'children': [
                {'name': 'raw'},
                {'name': 'fine', 'deferred': ['tock']},
            ]},
            'done',
        ]
        machine_kwargs = dict(states=states, initial='calib_raw', auto_transitions=False)
        machine_kwargs.update(kwargs)
        machine = HierarchicalMachine(model, **machine_kwargs)
        machine.add_transition('tick', 'done', 'done', after='record')
        machine.add_transition('tock', 'done', 'done', after='record')
        machine.add_transition('finish', 'calib', 'done')
        return machine

    def test_parent_declaration_is_inherited(self):
        model = Collector()
        self.create_machine(model)
        self.assertTrue(model.tick(1))
        self.assertTrue(model.tock(2))
        self.assertEqual('calib_raw', model.state)
        self.assertEqual([], model.log)
        model.finish()
        self.assertEqual('done', model.state)
        self.assertEqual([((1,), {}), ((2,), {})], model.log)

    def test_child_declaration_takes_precedence(self):
        model = Collector()
        machine = self.create_machine(model)
        machine.add_transition('tick', 'calib_fine', 'done')
        model.to('calib_fine')
        # 'calib_fine' declares its own deferred events which override the parent's declaration:
        # 'tick' is processed immediately ...
        model.tick()
        self.assertEqual('done', model.state)
        model.to('calib_fine')
        # ... while 'tock' is still deferred
        self.assertTrue(model.tock(5))
        self.assertEqual('calib_fine', model.state)
        self.assertEqual([], model.log)
        model.finish()
        self.assertEqual([((5,), {})], model.log)

    def test_nested_replay_in_submission_order(self):
        model = Collector()
        self.create_machine(model)
        model.tock('a')
        model.tick('b')
        model.tock('c')
        model.finish()
        self.assertEqual([(('a',), {}), (('b',), {}), (('c',), {})], model.log)

    def test_parallel_any_region_defers(self):
        model = Collector()
        states = [{'name': 'P', 'parallel': [
            {'name': 'r1', 'children': [{'name': 'a', 'deferred': ['ping']}, 'b'], 'initial': 'a'},
            {'name': 'r2', 'children': ['x', 'y'], 'initial': 'x'},
        ]}]
        machine = HierarchicalMachine(model, states=states, initial='P', auto_transitions=False)
        machine.add_transition('ping', 'P_r1_b', 'P_r1_b', after='record')
        machine.add_transition('step', 'P_r1_a', 'P_r1_b')
        self.assertEqual(['P_r1_a', 'P_r2_x'], model.state)
        # region r1 requires 'ping' to be deferred even though region r2 does not
        self.assertTrue(model.ping(1))
        self.assertEqual([], model.log)
        model.step()
        # no region defers 'ping' anymore: the event is replayed
        self.assertEqual(['P_r1_b', 'P_r2_x'], model.state)
        self.assertEqual([((1,), {})], model.log)

    def test_nested_multiple_models_and_removal(self):
        model_a = Collector()
        model_b = Collector()
        machine = self.create_machine([model_a, model_b])
        model_a.tick('a')
        model_b.tick('b')
        self.assertEqual(2, len(machine._deferred_queues))
        machine.remove_model(model_a)
        self.assertNotIn(model_a, machine._deferred_queues)
        model_b.finish()
        self.assertEqual([(('b',), {})], model_b.log)

    def test_nested_max_deferred(self):
        model = Collector()
        machine = self.create_machine(model, max_deferred=1)
        model.tick(1)
        with self.assertLogs('transitions.core', level='WARNING'):
            model.tick(2)
        self.assertEqual(1, len(machine._deferred_queues[model]))
        model.finish()
        self.assertEqual([((2,), {})], model.log)
