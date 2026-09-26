# -*- coding: utf-8 -*-
try:
    from builtins import object
except ImportError:
    pass

from collections import OrderedDict

from transitions.extensions.nesting import (
    HierarchicalMachine, NestedState, HistoryState, ShallowHistory, DeepHistory)

from .test_core import TestCase


class Model(object):
    pass


# A device workflow: the device can be paused (put into standby) from every operational mode.
# Resuming should return to the mode (and submode) that was active before the pause.
device_states = [
    'off',
    'standby',
    {'name': 'on', 'children': [
        {'name': 'playback', 'children': ['paused', 'playing'], 'initial': 'paused'},
        {'name': 'settings', 'children': ['audio', 'video'], 'initial': 'audio'},
        ShallowHistory('H', default='settings')],
     'initial': 'playback'},
]

device_transitions = [
    ['power', 'off', 'on'],
    ['pause', 'on', 'standby'],
    ['resume', 'standby', 'on_H'],
    ['shut_down', ['standby', 'on'], 'off'],
    ['play', 'on_playback_paused', 'on_playback_playing'],
    ['stop', 'on_playback_playing', 'on_playback_paused'],
    ['tweak', 'on_playback', 'on_settings'],
    ['watch', 'on_settings', 'on_playback'],
]


class TestShallowHistory(TestCase):

    def setUp(self):
        self.machine = HierarchicalMachine(states=device_states, transitions=device_transitions,
                                           initial='off')

    def test_resume_returns_to_paused_substate(self):
        self.machine.power()
        self.machine.tweak()
        self.assertEqual(self.machine.state, 'on_settings_audio')
        self.machine.pause()
        self.machine.resume()
        # shallow history remembers 'settings' but enters it with its initial substate
        self.assertEqual(self.machine.state, 'on_settings_audio')

    def test_shallow_history_forgets_deep_configuration(self):
        self.machine.power()
        self.machine.play()
        self.assertEqual(self.machine.state, 'on_playback_playing')
        self.machine.pause()
        self.machine.resume()
        # only one level is remembered; 'playback' is re-entered with its initial state
        self.assertEqual(self.machine.state, 'on_playback_paused')

    def test_default_target_without_history(self):
        machine = HierarchicalMachine(states=device_states, transitions=device_transitions,
                                      initial='standby')
        machine.resume()
        self.assertEqual(machine.state, 'on_settings_audio')

    def test_parent_initial_without_history_or_default(self):
        states = [
            'standby',
            {'name': 'on', 'children': ['a', 'b', ShallowHistory('H')], 'initial': 'a'},
        ]
        transitions = [['pause', 'on', 'standby'], ['resume', 'standby', 'on_H']]
        machine = HierarchicalMachine(states=states, transitions=transitions, initial='standby')
        machine.resume()
        self.assertEqual(machine.state, 'on_a')

    def test_repeated_pause_resume(self):
        self.machine.power()
        for expected in ('on_playback_paused', 'on_settings_audio', 'on_playback_paused'):
            self.machine.pause()
            self.assertEqual(self.machine.state, 'standby')
            self.machine.resume()
            self.assertEqual(self.machine.state, expected)
            if expected.endswith('paused'):
                self.machine.tweak()
            else:
                self.machine.watch()

    def test_history_reset_after_new_exit(self):
        self.machine.power()
        self.machine.play()
        self.machine.pause()
        self.machine.resume()
        self.assertEqual(self.machine.state, 'on_playback_paused')
        self.machine.tweak()
        self.machine.pause()
        self.machine.resume()
        self.assertEqual(self.machine.state, 'on_settings_audio')


class TestDeepHistory(TestCase):

    def setUp(self):
        states = [
            'off',
            'standby',
            {'name': 'on', 'children': [
                {'name': 'playback', 'children': ['paused', 'playing'], 'initial': 'paused'},
                {'name': 'settings', 'children': ['audio', 'video'], 'initial': 'audio'},
                DeepHistory('H', default='settings')],
             'initial': 'playback'},
        ]
        self.machine = HierarchicalMachine(states=states, transitions=device_transitions,
                                           initial='off')

    def test_resume_returns_to_deep_substate(self):
        self.machine.power()
        self.machine.play()
        self.machine.pause()
        self.machine.resume()
        # deep history remembers the entire subtree
        self.assertEqual(self.machine.state, 'on_playback_playing')

    def test_default_target_with_path(self):
        states = [
            'standby',
            {'name': 'on', 'children': [
                {'name': 'playback', 'children': ['paused', 'playing'], 'initial': 'paused'},
                DeepHistory('H', default='playback_playing')],
             'initial': 'playback'},
        ]
        transitions = [['pause', 'on', 'standby'], ['resume', 'standby', 'on_H']]
        machine = HierarchicalMachine(states=states, transitions=transitions, initial='standby')
        machine.resume()
        # a default containing the separator is treated as a path
        self.assertEqual(machine.state, 'on_playback_playing')

    def test_repeated_pause_resume(self):
        self.machine.power()
        self.machine.play()
        for _ in range(3):
            self.machine.pause()
            self.machine.resume()
            self.assertEqual(self.machine.state, 'on_playback_playing')
            self.machine.stop()
            self.machine.pause()
            self.machine.resume()
            self.assertEqual(self.machine.state, 'on_playback_paused')
            self.machine.play()


class TestParallelHistory(TestCase):

    def setUp(self):
        self.transitions = [
            ['start', 'standby', 'active'],
            ['pause', 'active', 'standby'],
            ['resume', 'standby', 'active_H'],
            ['upgrade', 'active_video_hd', 'active_video_uhd'],
            ['amp', 'active_audio_stereo', 'active_audio_surround'],
        ]

    @staticmethod
    def parallel_states(history):
        return [
            'standby',
            {'name': 'active', 'children': [
                {'name': 'video', 'children': ['hd', 'uhd'], 'initial': 'hd'},
                {'name': 'audio', 'children': ['stereo', 'surround'], 'initial': 'stereo'},
                history],
             'initial': ['video', 'audio']},
        ]

    def test_deep_history_restores_parallel_regions(self):
        machine = HierarchicalMachine(states=self.parallel_states(DeepHistory('H')),
                                      transitions=self.transitions, initial='standby')
        machine.start()
        machine.upgrade()
        machine.amp()
        self.assertEqual(sorted(machine.state), ['active_audio_surround', 'active_video_uhd'])
        machine.pause()
        machine.resume()
        self.assertEqual(sorted(machine.state), ['active_audio_surround', 'active_video_uhd'])

    def test_shallow_history_resets_parallel_regions(self):
        machine = HierarchicalMachine(states=self.parallel_states(ShallowHistory('H')),
                                      transitions=self.transitions, initial='standby')
        machine.start()
        machine.upgrade()
        machine.amp()
        machine.pause()
        machine.resume()
        # shallow history remembers the regions but re-enters them with their initial states
        self.assertEqual(sorted(machine.state), ['active_audio_stereo', 'active_video_hd'])

    def test_nested_parallel_history(self):
        states = [
            'standby',
            {'name': 'on', 'children': [
                {'name': 'active', 'children': [
                    {'name': 'video', 'children': ['hd', 'uhd'], 'initial': 'hd'},
                    {'name': 'audio', 'children': ['stereo', 'surround'], 'initial': 'stereo'}],
                 'initial': ['video', 'audio']},
                'idle',
                DeepHistory('H')],
             'initial': 'active'},
        ]
        transitions = [
            ['pause', 'on', 'standby'],
            ['resume', 'standby', 'on_H'],
            ['upgrade', 'on_active_video_hd', 'on_active_video_uhd'],
            ['amp', 'on_active_audio_stereo', 'on_active_audio_surround'],
        ]
        machine = HierarchicalMachine(states=states, transitions=transitions, initial='standby')
        machine.to_on()
        machine.upgrade()
        machine.amp()
        machine.pause()
        machine.resume()
        self.assertEqual(sorted(machine.state), ['on_active_audio_surround', 'on_active_video_uhd'])


class TestHistoryModels(TestCase):

    def test_history_is_kept_per_model(self):
        states = [
            'standby',
            {'name': 'on', 'children': ['a', 'b', DeepHistory('H')], 'initial': 'a'},
        ]
        transitions = [
            ['pause', 'on', 'standby'],
            ['resume', 'standby', 'on_H'],
            ['go', 'on_a', 'on_b'],
        ]
        first, second = Model(), Model()
        machine = HierarchicalMachine(model=first, states=states, transitions=transitions,
                                      initial='standby')
        machine.add_model(second)
        first.to_on()
        first.go()
        first.pause()
        second.to_on()
        second.pause()
        # the second model must not overwrite the history of the first one
        first.resume()
        second.resume()
        self.assertEqual(first.state, 'on_b')
        self.assertEqual(second.state, 'on_a')

    def test_history_stored_on_model(self):
        machine = HierarchicalMachine(states=device_states, transitions=device_transitions,
                                      initial='off')
        model = Model()
        machine.add_model(model)
        model.power()
        model.pause()
        store = machine._get_history_store(model)
        self.assertIn('on_H', store)
        self.assertEqual(store['on_H'], OrderedDict([('playback', None)]))
        # the machine's own history is not affected by the model
        self.assertEqual(machine._get_history_store(machine), {})


class TestHistoryInternalTransitions(TestCase):

    def setUp(self):
        states = [
            'standby',
            {'name': 'on', 'children': ['a', 'b', DeepHistory('H')], 'initial': 'a'},
        ]
        transitions = [
            ['ping', 'on', None],      # internal transition on the parent
            ['poke', 'on_a', None],    # internal transition on a child
            ['pause', 'on', 'standby'],
            ['resume', 'standby', 'on_H'],
            ['go', 'on_a', 'on_b'],
        ]
        self.machine = HierarchicalMachine(states=states, transitions=transitions,
                                           initial='standby')

    def test_internal_transition_records_no_history(self):
        self.machine.to_on()
        self.machine.ping()
        self.machine.poke()
        # internal transitions do not exit states and must not create (fake) history
        self.assertEqual(self.machine._get_history_store(self.machine), {})
        self.machine.pause()
        self.assertIn('on_H', self.machine._get_history_store(self.machine))

    def test_internal_transition_does_not_corrupt_history(self):
        self.machine.to_on()
        self.machine.go()
        self.machine.pause()
        self.machine.resume()
        self.assertEqual(self.machine.state, 'on_b')
        self.machine.ping()
        self.machine.pause()
        self.machine.resume()
        self.assertEqual(self.machine.state, 'on_b')


class TestHistoryCallbackOrder(TestCase):

    def test_enter_exit_order(self):
        calls = []

        class Tracker(object):
            def track(self, name):
                calls.append(name)

        tracker = Tracker()
        for name in ['on', 'on_play', 'on_play_a', 'on_play_b', 'standby']:
            setattr(Tracker, 'on_enter_%s' % name,
                    (lambda n: lambda self: self.track('enter_' + n))(name))
            setattr(Tracker, 'on_exit_%s' % name,
                    (lambda n: lambda self: self.track('exit_' + n))(name))

        states = [
            'standby',
            {'name': 'on', 'children': [
                {'name': 'play', 'children': ['a', 'b'], 'initial': 'a'},
                DeepHistory('H')],
             'initial': 'play'},
        ]
        transitions = [
            ['pause', 'on', 'standby'],
            ['resume', 'standby', 'on_H'],
            ['go', 'on_play_a', 'on_play_b'],
        ]
        machine = HierarchicalMachine(model=tracker, states=states, transitions=transitions,
                                      initial='standby')
        tracker.to_on()
        tracker.go()
        self.assertEqual(tracker.state, 'on_play_b')
        calls[:] = []
        tracker.pause()
        # children are exited before their parents
        self.assertEqual(calls, ['exit_on_play_b', 'exit_on_play', 'exit_on', 'enter_standby'])
        calls[:] = []
        tracker.resume()
        # parents are entered before their children
        self.assertEqual(calls, ['exit_standby', 'enter_on', 'enter_on_play', 'enter_on_play_b'])


class TestHistorySeparator(TestCase):

    def setUp(self):
        self.default_separator = NestedState.separator
        NestedState.separator = '.'

    def tearDown(self):
        NestedState.separator = self.default_separator

    def test_custom_separator_deep(self):
        states = [
            'standby',
            {'name': 'on', 'children': [
                {'name': 'play', 'children': ['x', 'y'], 'initial': 'x'},
                DeepHistory('H', default='play.y')],
             'initial': 'play'},
        ]
        transitions = [
            ['pause', 'on', 'standby'],
            ['resume', 'standby', 'on.H'],
            ['go', 'on.play.x', 'on.play.y'],
        ]
        machine = HierarchicalMachine(states=states, transitions=transitions, initial='standby')
        machine.resume()
        self.assertEqual(machine.state, 'on.play.y')
        machine.pause()
        machine.to_on()
        machine.go()
        machine.pause()
        machine.resume()
        self.assertEqual(machine.state, 'on.play.y')

    def test_custom_separator_shallow(self):
        states = [
            'standby',
            {'name': 'on', 'children': [
                {'name': 'play', 'children': ['x', 'y'], 'initial': 'x'},
                ShallowHistory('H')],
             'initial': 'play'},
        ]
        transitions = [
            ['pause', 'on', 'standby'],
            ['resume', 'standby', 'on.H'],
            ['go', 'on.play.x', 'on.play.y'],
        ]
        machine = HierarchicalMachine(states=states, transitions=transitions, initial='standby')
        machine.to_on()
        machine.go()
        machine.pause()
        machine.resume()
        self.assertEqual(machine.state, 'on.play.x')


class TestDeviceWorkflow(TestCase):
    """A pause/resume workflow of a media device using history pseudo-states."""

    def test_pause_and_resume_workflow(self):
        device = HierarchicalMachine(states=device_states, transitions=device_transitions,
                                     initial='off')
        # resuming before the device was ever on uses the history state's default target
        device.to_standby()
        device.resume()
        self.assertEqual(device.state, 'on_settings_audio')
        device.shut_down()
        # turning the device on enters the initial mode
        device.power()
        self.assertEqual(device.state, 'on_playback_paused')
        # operate the device and pause it while playing
        device.play()
        device.pause()
        self.assertEqual(device.state, 'standby')
        # shallow history: resume returns to playback mode but not to 'playing'
        device.resume()
        self.assertEqual(device.state, 'on_playback_paused')
        # the device can still be shut down from a resumed mode
        device.shut_down()
        self.assertEqual(device.state, 'off')


class TestHistoryStateDefinition(TestCase):

    def test_history_state_is_not_entered(self):
        machine = HierarchicalMachine(states=device_states, transitions=device_transitions,
                                      initial='off')
        machine.power()
        machine.pause()
        machine.resume()
        # the pseudo-state never shows up as an actual state
        self.assertNotIn('H', machine.state.split('_'))

    def test_default_spec_building(self):
        pseudo = DeepHistory('H', default=['a', 'b_c'])
        spec = pseudo.build_default_spec()
        self.assertEqual(spec, OrderedDict([('a', None), ('b', OrderedDict([('c', None)]))]))
        self.assertIsNone(ShallowHistory('H').build_default_spec())
        self.assertFalse(HistoryState('H').deep)
        self.assertTrue(DeepHistory('H').deep)

    def test_history_state_requires_parent(self):
        with self.assertRaises(ValueError):
            HierarchicalMachine(states=['a', ShallowHistory('H')], initial='a')
