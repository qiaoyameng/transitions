# -*- coding: utf-8 -*-
"""Device workflow example: suspend/resume with shallow and deep history states.

A device boots into 'on_idle'. When the user starts working, the device enters
the parallel state 'on_active' which manages a 'display' and a 'radio' region
independently. Suspending the device exits 'on' entirely. Resuming targets the
deep history state 'on_history' and restores the exact previously active
configuration (e.g. 'bright' display and 'online' radio) instead of the
configured default ('dim' display and 'offline' radio).

Run with: python examples/device_pause_resume.py
"""

from transitions.extensions import HierarchicalMachine, HistoryState


class Device(object):

    def show_state(self):
        print("  -> current state: {0}".format(self.state))


states = [
    {'name': 'on', 'initial': 'idle', 'children': [
        'idle',
        {'name': 'active', 'parallel': [
            {'name': 'display', 'children': ['dim', 'bright'], 'initial': 'dim'},
            {'name': 'radio', 'children': ['offline', 'online'], 'initial': 'offline'},
        ]},
        # shallow history: HistoryState('history') restores only the direct children of 'on'
        # deep history restores the entire active subtree, including both parallel regions
        HistoryState('history', deep=True),
    ]},
    'suspended',
    'off',
]

transitions = [
    {'trigger': 'boot', 'source': 'off', 'dest': 'on'},
    {'trigger': 'start_work', 'source': 'on_idle', 'dest': 'on_active'},
    {'trigger': 'brighten', 'source': 'on_active_display_dim', 'dest': 'on_active_display_bright'},
    {'trigger': 'connect', 'source': 'on_active_radio_offline', 'dest': 'on_active_radio_online'},
    {'trigger': 'suspend', 'source': 'on', 'dest': 'suspended'},
    # resume targets the history pseudo state of 'on'
    {'trigger': 'resume', 'source': 'suspended', 'dest': 'on_history'},
    {'trigger': 'shutdown', 'source': ['on', 'suspended'], 'dest': 'off'},
]


def main():
    device = Device()
    machine = HierarchicalMachine(model=device, states=states, transitions=transitions,
                                  initial='off', ignore_invalid_triggers=True)

    print("boot the device (enters the initial state 'on_idle')")
    device.boot()
    device.show_state()

    print("start working (enters both parallel regions with their defaults)")
    device.start_work()
    device.show_state()

    print("adjust display and radio")
    device.brighten()
    device.connect()
    device.show_state()

    print("suspend the device (exits 'on' and records the active configuration)")
    device.suspend()
    device.show_state()

    print("resume the device (deep history restores 'bright' and 'online')")
    device.resume()
    device.show_state()
    assert sorted(device.state) == sorted(['on_active_display_bright', 'on_active_radio_online'])

    print("shut down and boot again (no history-related magic, 'on' starts with 'idle')")
    device.shutdown()
    device.boot()
    device.show_state()


if __name__ == '__main__':
    main()
