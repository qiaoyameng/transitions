# -*- coding: utf-8 -*-

from unittest import TestCase

from transitions.extensions import HierarchicalMachine, HistoryState
from transitions.extensions.nesting import NestedState

default_separator = NestedState.separator


class Dummy(object):
    pass


def device_states(history):
    """States of a device which can be paused/suspended and resumed."""
    return [
        {'name': 'running', 'initial': 'idle', 'children': [
            'idle',
            {'name': 'working', 'initial': 'a', 'children': ['a', 'b']},
            history,
        ]},
        'paused',
    ]


def device_transitions(separator='_'):
    return [
        {'trigger': 'pause', 'source': 'running', 'dest': 'paused'},
        {'trigger': 'work', 'source': separator.join(['running', 'idle']),
         'dest': separator.join(['running', 'working'])},
        {'trigger': 'advance', 'source': separator.join(['running', 'working', 'a']),
         'dest': separator.join(['running', 'working', 'b'])},
        {'trigger': 'ping', 'source': 'running', 'dest': None},  # internal transition
    ]


class TestHistoryState(TestCase):

    def setUp(self):
        self.state_cls = NestedState

    def get_machine(self, history, **kwargs):
        states = device_states(history)
        transitions = device_transitions(self.state_cls.separator) + [
            {'trigger': 'resume', 'source': 'paused',
             'dest': 'running{0}{1}'.format(self.state_cls.separator, history.name)},
        ]
        kwargs.setdefault('initial', 'running')
        return HierarchicalMachine(states=states, transitions=transitions, **kwargs)

    def test_shallow_history(self):
        machine = self.get_machine(HistoryState('hist'))
        machine.work()
        machine.advance()
        self.assertEqual(machine.state, 'running_working_b')
        machine.pause()
        machine.resume()
        # shallow history restores the direct child ('working') but enters its initial state
        self.assertEqual(machine.state, 'running_working_a')

    def test_deep_history(self):
        machine = self.get_machine(HistoryState('hist', deep=True))
        machine.work()
        machine.advance()
        machine.pause()
        machine.resume()
        # deep history restores the entire subtree
        self.assertEqual(machine.state, 'running_working_b')

    def test_no_history_enters_default(self):
        # without a recorded history, the configured initial state of the parent scope is entered
        for deep in (False, True):
            machine = self.get_machine(HistoryState('hist', deep=deep), initial='paused')
            machine.resume()
            self.assertEqual(machine.state, 'running_idle')

    def test_repeated_pause_resume(self):
        machine = self.get_machine(HistoryState('hist', deep=True))
        machine.work()
        machine.pause()
        machine.resume()
        self.assertEqual(machine.state, 'running_working_a')
        machine.advance()
        machine.pause()
        machine.resume()
        self.assertEqual(machine.state, 'running_working_b')
        machine.pause()
        machine.resume()
        self.assertEqual(machine.state, 'running_working_b')

    def test_history_updated_on_each_exit(self):
        machine = self.get_machine(HistoryState('hist', deep=True))
        machine.work()
        machine.pause()
        machine.resume()
        machine.to_running_idle()
        machine.pause()
        machine.resume()
        self.assertEqual(machine.state, 'running_idle')

    def test_internal_transition_does_not_record_history(self):
        machine = self.get_machine(HistoryState('hist', deep=True))
        machine.work()
        machine.advance()
        machine.pause()
        machine.resume()
        self.assertEqual(machine.state, 'running_working_b')
        # internal transitions must not create (fake) exit history
        machine.ping()
        machine.to_running_idle()
        machine.pause()
        machine.resume()
        self.assertEqual(machine.state, 'running_idle')

    def test_multiple_models_do_not_share_history(self):
        machine = self.get_machine(HistoryState('hist', deep=True), model=Dummy())
        model_a, model_b = machine.models[0], Dummy()
        machine.add_model(model_b, initial='running')
        machine.trigger_event(model_a, 'work')
        machine.trigger_event(model_a, 'advance')
        machine.trigger_event(model_b, 'work')
        machine.trigger_event(model_a, 'pause')
        machine.trigger_event(model_b, 'pause')
        machine.trigger_event(model_a, 'resume')
        machine.trigger_event(model_b, 'resume')
        self.assertEqual(model_a.state, 'running_working_b')
        self.assertEqual(model_b.state, 'running_working_a')

    def test_history_state_object_as_dest(self):
        history = HistoryState('hist', deep=True)
        states = device_states(history)
        transitions = device_transitions(self.state_cls.separator) + [
            {'trigger': 'resume', 'source': 'paused', 'dest': history},
        ]
        machine = HierarchicalMachine(states=states, transitions=transitions, initial='running')
        machine.work()
        machine.pause()
        machine.resume()
        self.assertEqual(machine.state, 'running_working_a')

    def test_enter_exit_callback_order(self):
        log = []

        def make_cb(name):
            return lambda: log.append(name)

        sep = self.state_cls.separator
        states = [
            {'name': 'running', 'initial': 'working', 'children': [
                {'name': 'working', 'initial': 'a', 'children': [
                    {'name': 'a', 'on_enter': make_cb('enter_a'), 'on_exit': make_cb('exit_a')},
                    'b'],
                 'on_enter': make_cb('enter_working'), 'on_exit': make_cb('exit_working')},
                HistoryState('hist', deep=True),
            ], 'on_enter': make_cb('enter_running'), 'on_exit': make_cb('exit_running')},
            'paused',
        ]
        transitions = [
            {'trigger': 'pause', 'source': 'running', 'dest': 'paused'},
            {'trigger': 'resume', 'source': 'paused', 'dest': sep.join(['running', 'hist'])},
        ]
        machine = HierarchicalMachine(states=states, transitions=transitions, initial='running')
        self.assertEqual(machine.state, 'running_working_a')
        del log[:]
        machine.pause()
        self.assertEqual(log, ['exit_a', 'exit_working', 'exit_running'])
        del log[:]
        machine.resume()
        self.assertEqual(log, ['enter_running', 'enter_working', 'enter_a'])


class TestParallelHistory(TestCase):

    def setUp(self):
        self.state_cls = NestedState

    def get_machine(self, history, **kwargs):
        sep = self.state_cls.separator
        states = [
            {'name': 'on', 'initial': 'display', 'children': [
                {'name': 'display', 'parallel': [
                    {'name': 'brightness', 'children': ['low', 'high'], 'initial': 'low'},
                    {'name': 'volume', 'children': ['mute', 'loud'], 'initial': 'mute'},
                ]},
                history,
            ]},
            'off',
        ]
        transitions = [
            {'trigger': 'suspend', 'source': 'on', 'dest': 'off'},
            {'trigger': 'resume', 'source': 'off', 'dest': sep.join(['on', history.name])},
            {'trigger': 'brighten', 'source': sep.join(['on', 'display', 'brightness', 'low']),
             'dest': sep.join(['on', 'display', 'brightness', 'high'])},
            {'trigger': 'amplify', 'source': sep.join(['on', 'display', 'volume', 'mute']),
             'dest': sep.join(['on', 'display', 'volume', 'loud'])},
        ]
        kwargs.setdefault('initial', 'on')
        return HierarchicalMachine(states=states, transitions=transitions, **kwargs)

    def test_deep_history_parallel(self):
        machine = self.get_machine(HistoryState('hist', deep=True))
        self.assertEqual(sorted(machine.state), sorted(['on_display_brightness_low', 'on_display_volume_mute']))
        machine.brighten()
        machine.amplify()
        machine.suspend()
        self.assertEqual(machine.state, 'off')
        machine.resume()
        # the entire active subtree including all parallel regions is restored
        self.assertEqual(sorted(machine.state), sorted(['on_display_brightness_high', 'on_display_volume_loud']))

    def test_shallow_history_parallel(self):
        machine = self.get_machine(HistoryState('hist'))
        machine.brighten()
        machine.amplify()
        machine.suspend()
        machine.resume()
        # shallow history restores 'display' which enters its (parallel) initial configuration
        self.assertEqual(sorted(machine.state), sorted(['on_display_brightness_low', 'on_display_volume_mute']))

    def test_nested_parallel_deep_history(self):
        sep = self.state_cls.separator
        states = [
            {'name': 'on', 'initial': 'main', 'children': [
                {'name': 'main', 'parallel': [
                    {'name': 'display', 'children': [
                        {'name': 'brightness', 'children': ['low', 'high'], 'initial': 'low'},
                    ], 'initial': 'brightness'},
                    {'name': 'audio', 'children': [
                        {'name': 'volume', 'children': ['mute', 'loud'], 'initial': 'mute'},
                        {'name': 'balance', 'children': ['left', 'right'], 'initial': 'left'},
                    ], 'initial': 'volume'},
                ]},
                HistoryState('hist', deep=True),
            ]},
            'off',
        ]
        transitions = [
            {'trigger': 'suspend', 'source': 'on', 'dest': 'off'},
            {'trigger': 'resume', 'source': 'off', 'dest': sep.join(['on', 'hist'])},
            {'trigger': 'brighten', 'source': sep.join(['on', 'main', 'display', 'brightness', 'low']),
             'dest': sep.join(['on', 'main', 'display', 'brightness', 'high'])},
            {'trigger': 'amplify', 'source': sep.join(['on', 'main', 'audio', 'volume', 'mute']),
             'dest': sep.join(['on', 'main', 'audio', 'volume', 'loud'])},
        ]
        machine = HierarchicalMachine(states=states, transitions=transitions, initial='on')
        self.assertEqual(sorted(machine.state),
                         sorted(['on_main_display_brightness_low', 'on_main_audio_volume_mute']))
        machine.brighten()
        machine.amplify()
        machine.suspend()
        machine.resume()
        self.assertEqual(sorted(machine.state),
                         sorted(['on_main_display_brightness_high', 'on_main_audio_volume_loud']))

    def test_repeated_suspend_resume_parallel(self):
        machine = self.get_machine(HistoryState('hist', deep=True))
        for _ in range(3):
            machine.brighten()
            machine.suspend()
            machine.resume()
            self.assertIn('on_display_brightness_high', machine.state)
            machine.to_on_display_brightness_low()
            machine.suspend()
            machine.resume()
            self.assertIn('on_display_brightness_low', machine.state)


class TestHistoryCustomSeparator(TestCase):

    def setUp(self):
        self.state_cls = NestedState
        self.state_cls.separator = '.'

    def tearDown(self):
        self.state_cls.separator = default_separator

    def test_custom_separator(self):
        states = device_states(HistoryState('hist', deep=True))
        transitions = device_transitions(self.state_cls.separator) + [
            {'trigger': 'resume', 'source': 'paused', 'dest': 'running.hist'},
        ]
        machine = HierarchicalMachine(states=states, transitions=transitions, initial='running')
        self.assertEqual(machine.state, 'running.idle')
        machine.work()
        machine.advance()
        self.assertEqual(machine.state, 'running.working.b')
        machine.pause()
        machine.resume()
        self.assertEqual(machine.state, 'running.working.b')
        machine.pause()
        machine.to_running.hist()
        self.assertEqual(machine.state, 'running.working.b')
