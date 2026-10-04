try:
    from builtins import object
except ImportError:
    pass

from collections import OrderedDict

from transitions.extensions.nesting import NestedState as State, _build_state_list
from transitions.extensions import HierarchicalGraphMachine
from transitions import MachineError
from .test_nesting import TestNestedTransitions as TestNested
from .test_pygraphviz import pgv
from .test_graphviz import pgv as gv
from unittest import skipIf

try:
    from unittest.mock import MagicMock
except ImportError:
    from mock import MagicMock  # type: ignore


class TestParallel(TestNested):

    def setUp(self):
        super(TestParallel, self).setUp()
        self.states = ['A', 'B', {'name': 'C',
                                  'parallel': [{'name': '1', 'children': ['a', 'b'],
                                                'initial': 'a',
                                                'transitions': [['go', 'a', 'b']]},
                                               {'name': '2', 'children': ['a', 'b'],
                                                'initial': 'a',
                                                'transitions': [['go', 'a', 'b']]}]}]
        self.transitions = [['reset', 'C', 'A']]

    def test_init(self):
        m = self.stuff.machine_cls(states=self.states)
        m.to_C()
        self.assertEqual(['C{0}1{0}a'.format(State.separator), 'C{0}2{0}a'.format(State.separator)], m.state)

    def test_enter(self):
        m = self.stuff.machine_cls(states=self.states, transitions=self.transitions, initial='A')
        m.to_C()
        m.go()
        self.assertEqual(['C{0}1{0}b'.format(State.separator), 'C{0}2{0}b'.format(State.separator)], m.state)

    def test_exit(self):

        class Model:

            def __init__(self):
                self.mock = MagicMock()

            def on_exit_C(self):
                self.mock()

            def on_exit_C_1(self):
                self.mock()

            def on_exit_C_2(self):
                self.mock()

        model1 = Model()
        m = self.stuff.machine_cls(model1, states=self.states, transitions=self.transitions, initial='A')
        m.add_transition('reinit', 'C', 'C')
        model1.to_C()
        self.assertEqual(['C{0}1{0}a'.format(State.separator), 'C{0}2{0}a'.format(State.separator)], model1.state)
        model1.reset()
        self.assertTrue(model1.is_A())
        self.assertEqual(3, model1.mock.call_count)

        model2 = Model()
        m.add_model(model2, initial='C')
        model2.reinit()
        self.assertEqual(['C{0}1{0}a'.format(State.separator), 'C{0}2{0}a'.format(State.separator)], model2.state)
        self.assertEqual(3, model2.mock.call_count)
        model2.reset()
        self.assertTrue(model2.is_A())
        self.assertEqual(6, model2.mock.call_count)
        for mod in m.models:
            mod.trigger('to_C')
        for mod in m.models:
            mod.trigger('reset')
        self.assertEqual(6, model1.mock.call_count)
        self.assertEqual(9, model2.mock.call_count)

    def test_parent_transition(self):
        m = self.stuff.machine_cls(states=self.states)
        m.add_transition('switch', 'C{0}2{0}a'.format(State.separator), 'C{0}2{0}b'.format(State.separator))
        m.to_C()
        m.switch()
        self.assertEqual(['C{0}1{0}a'.format(State.separator), 'C{0}2{0}b'.format(State.separator)], m.state)

    def test_shallow_parallel(self):
        sep = self.state_cls.separator
        states = [
            {
                'name': 'P', 'parallel':
                [
                    '1',  # no initial state
                    {'name': '2', 'children': ['a', 'b'], 'initial': 'b'}
                ]
            },
            'X'
        ]
        m = self.machine_cls(states=states, initial='P')
        self.assertEqual(['P{0}1'.format(sep), 'P{0}2{0}b'.format(sep)], m.state)
        m.to_X()
        self.assertEqual('X', m.state)
        m.to_P()
        self.assertEqual(['P{0}1'.format(sep), 'P{0}2{0}b'.format(sep)], m.state)
        with self.assertRaises(MachineError):
            m.to('X')

    def test_multiple(self):
        states = ['A',
                  {'name': 'B',
                   'parallel': [
                       {'name': '1', 'parallel': [
                           {'name': 'a', 'children': ['x', 'y', 'z'], 'initial': 'z'},
                           {'name': 'b', 'children': ['x', 'y', 'z'], 'initial': 'y'}
                       ]},
                       {'name': '2', 'children': ['a', 'b', 'c'], 'initial': 'a'},
                   ]}]

        m = self.stuff.machine_cls(states=states, initial='A')
        self.assertTrue(m.is_A())
        m.to_B()
        self.assertEqual([['B{0}1{0}a{0}z'.format(State.separator),
                           'B{0}1{0}b{0}y'.format(State.separator)],
                          'B{0}2{0}a'.format(State.separator)], m.state)

        # check whether we can initialize a new machine in a parallel state
        m2 = self.machine_cls(states=states, initial=m.state)
        self.assertEqual([['B{0}1{0}a{0}z'.format(State.separator),
                           'B{0}1{0}b{0}y'.format(State.separator)],
                          'B{0}2{0}a'.format(State.separator)], m2.state)
        m.to_A()
        self.assertEqual('A', m.state)
        m2.to_A()
        self.assertEqual(m.state, m2.state)

    def test_deep_initial(self):
        exit_mock = MagicMock()
        m = self.machine_cls(initial=['B{0}1'.format(State.separator), 'B{0}2{0}a'.format(State.separator)])
        m.on_exit('B', exit_mock)
        m.on_exit('B{0}1'.format(State.separator), exit_mock)
        m.on_exit('B{0}2'.format(State.separator), exit_mock)
        m.on_exit('B{0}2{0}a'.format(State.separator), exit_mock)
        m.to_B()
        self.assertEqual('B', m.state)
        self.assertEqual(4, exit_mock.call_count)

    def test_parallel_initial(self):
        m = self.machine_cls(states=['A', 'B', {'name': 'C', 'parallel': ['1', '2']}], initial='C')
        m = self.machine_cls(states=['A', 'B', {'name': 'C', 'parallel': ['1', '2']}], initial=['C_1', 'C_2'])

    def test_parallel_reflexive(self):
        exit_c_1_mock = MagicMock()
        m = self.machine_cls(states=['A', 'B', {'name': 'C', 'parallel': [{'name': '1',
                                                                           'on_exit': exit_c_1_mock}, '2']}],
                             transitions=[['test',
                                           'C{0}2'.format(State.separator),
                                           'C{0}2'.format(State.separator)]], initial='C')
        m.test()
        self.assertEqual(["C{0}1".format(State.separator),
                          "C{0}2".format(State.separator)], m.state)
        self.assertFalse(exit_c_1_mock.called)

    def test_multiple_deeper(self):
        sep = self.state_cls.separator
        states = ['A',
                  {'name': 'P',
                   'parallel': [
                       '1',
                       {'name': '2', 'parallel': [
                           {'name': 'a'},
                           {'name': 'b', 'parallel': [
                               {'name': 'x', 'parallel': ['1', '2']}, 'y'
                           ]}
                       ]},
                   ]}]
        ref_state = ['P{0}1'.format(sep),
                     ['P{0}2{0}a'.format(sep),
                     [['P{0}2{0}b{0}x{0}1'.format(sep),
                       'P{0}2{0}b{0}x{0}2'.format(sep)],
                      'P{0}2{0}b{0}y'.format(sep)]]]

        m = self.stuff.machine_cls(states=states, initial='A')
        self.assertTrue(m.is_A())
        m.to_P()
        self.assertEqual(ref_state, m.state)
        m.to_A()

    def test_model_state_conversion(self):
        sep = self.state_cls.separator
        states = ['P{0}1'.format(sep),
                  ['P{0}2{0}a'.format(sep),
                   [['P{0}2{0}b{0}x{0}1'.format(sep),
                     'P{0}2{0}b{0}x{0}2'.format(sep)],
                    'P{0}2{0}b{0}y'.format(sep)]]]
        tree = OrderedDict(
            [('P', OrderedDict(
                [('1', OrderedDict()),
                 ('2', OrderedDict(
                     [('a', OrderedDict()),
                      ('b', OrderedDict(
                          [('x', OrderedDict(
                              [('1', OrderedDict()),
                               ('2', OrderedDict())])),
                           ('y', OrderedDict())]
                      ))]
                 ))]
            ))]
        )  # type: ignore
        m = self.machine_cls()
        self.assertEqual(tree, m.build_state_tree(states, sep))
        self.assertEqual(states, _build_state_list(tree, sep))

    def test_may_transition_with_parallel(self):
        states = ['A',
                  {'name': 'P',
                   'parallel': [
                       {'name': '1', 'states': ["a", "b"], "initial": 'a'},
                       {'name': '2', 'states': ["a", "b"], "initial": 'a',
                        'transitions': [['valid', 'a', 'b']]}
                   ]}]
        m = self.machine_cls(states=states, initial="A")
        assert not m.may_valid()
        assert m.to_P()
        assert m.is_P(allow_substates=True)
        assert m.is_P_1_a()
        assert m.is_P_2_a()
        assert m.may_valid()
        assert m.valid()
        assert m.is_P_1_a()
        assert not m.is_P_2_a()
        assert m.is_P_2_b()

    def test_is_state_parallel(self):
        states = ['A',
                  {'name': 'P',
                   'parallel': [
                       '1',
                       {'name': '2', 'parallel': [
                           {'name': 'a'},
                           {'name': 'b', 'parallel': [
                               {'name': 'x', 'parallel': ['1', '2']}, 'y'
                           ]}
                       ]},
                   ]}]
        m = self.machine_cls(states=states, initial="A")
        assert m.is_A()
        assert not m.is_P_2()
        assert not m.is_P_2_a()
        assert not m.is_P_2_b()
        assert not m.is_P_2_b_x()
        assert not m.is_P(allow_substates=True)
        m.to_P()
        assert m.is_P_1()
        assert m.is_P_2_a()
        assert not m.is_P_2()
        assert m.is_P(allow_substates=True)
        assert m.is_P_2(allow_substates=True)
        assert not m.is_A(allow_substates=True)

    def test_synchronized_order_join(self):
        class OrderModel(object):
            def __init__(self):
                self.calls = []

            def _record(self, name):
                self.calls.append(name)

            def on_enter_order_payment_ready(self):
                self._record('enter_payment_ready')

            def on_exit_order_payment_pending(self):
                self._record('exit_payment_pending')

            def on_enter_order_stock_ready(self):
                self._record('enter_stock_ready')

            def on_exit_order_stock_pending(self):
                self._record('exit_stock_pending')

            def on_enter_idle(self):
                self._record('enter_idle')

            def on_exit_order_payment_ready(self):
                self._record('exit_payment_ready')

            def on_exit_order_stock_ready(self):
                self._record('exit_stock_ready')

            def on_exit_order(self):
                self._record('exit_order')

        states = ['idle', {'name': 'order', 'parallel': [
            {'name': 'payment', 'children': ['pending', 'ready'], 'initial': 'pending'},
            {'name': 'stock', 'children': ['pending', 'ready'], 'initial': 'pending'}
        ]}]
        model = OrderModel()
        machine = self.machine_cls(model=model, states=states, initial='idle', auto_transitions=False)
        machine.add_transition('pay', (['order_payment_pending', 'order_stock_pending'],),
                               ['order_payment_ready', 'order_stock_pending'],
                               conditions=lambda: model.calls.append('condition') or True)
        machine.add_transition('prepare_stock', (['order_payment_ready', 'order_stock_pending'],),
                               ['order_payment_ready', 'order_stock_ready'])
        machine.add_transition('ship', (['order_payment_ready', 'order_stock_ready'],), 'idle')
        machine.add_transition('start_order', 'idle', 'order')

        model.start_order()
        model.calls[:] = []

        self.assertTrue(model.trigger('pay'))
        self.assertEqual(['order_payment_ready', 'order_stock_pending'], model.state)
        self.assertEqual(['condition', 'exit_payment_pending', 'enter_payment_ready'], model.calls)
        self.assertIn('order_stock_pending', model.state)
        self.assertFalse(model.may_pay())
        self.assertEqual(['condition', 'exit_payment_pending', 'enter_payment_ready'], model.calls)

        self.assertTrue(model.prepare_stock())
        self.assertEqual(['order_payment_ready', 'order_stock_ready'], model.state)
        self.assertEqual(['condition', 'exit_payment_pending', 'enter_payment_ready',
                          'exit_stock_pending', 'enter_stock_ready'], model.calls)

        model.calls[:] = []
        self.assertTrue(model.ship())
        self.assertEqual('idle', model.state)
        self.assertEqual(['exit_payment_ready', 'exit_stock_ready',
                          'exit_order', 'enter_idle'], model.calls)

    def test_synchronized_models_use_independent_configurations(self):
        states = ['idle', {'name': 'order', 'parallel': [
            {'name': 'payment', 'children': ['pending', 'ready'], 'initial': 'pending'},
            {'name': 'stock', 'children': ['pending', 'ready'], 'initial': 'pending'}
        ]}]
        first = type('ModelOne', (), {})()
        second = type('ModelTwo', (), {})()
        machine = self.machine_cls(model=[first, second], states=states, initial='order', auto_transitions=False)
        machine.add_transition('pay', [['order_payment_pending', 'order_stock_pending']],
                               ['order_payment_ready', 'order_stock_pending'])
        machine.set_state(['order_payment_ready', 'order_stock_pending'], model=first)

        self.assertFalse(first.may_pay())
        self.assertEqual(['order_payment_ready', 'order_stock_pending'], first.state)
        self.assertTrue(second.pay())
        self.assertEqual(['order_payment_ready', 'order_stock_pending'], second.state)

    def test_synchronized_invalid_configuration(self):
        states = ['idle', {'name': 'order', 'parallel': [
            {'name': 'payment', 'children': ['pending', 'ready'], 'initial': 'pending'},
            {'name': 'stock', 'children': ['pending', 'ready'], 'initial': 'pending'}
        ]}]
        machine = self.machine_cls(states=states, initial='idle', auto_transitions=False)

        with self.assertRaises(ValueError):
            machine.add_transition('pay', [['order_payment_pending', 'order_payment_ready']],
                                   ['order_payment_ready', 'order_stock_pending'])
        with self.assertRaises(ValueError):
            machine.add_transition('fork', 'idle',
                                   ['order_payment_pending', 'order_payment_ready'])

    def test_synchronized_fork(self):
        states = ['idle', {'name': 'order', 'parallel': [
            {'name': 'payment', 'children': ['pending', 'ready'], 'initial': 'pending'},
            {'name': 'stock', 'children': ['pending', 'ready'], 'initial': 'pending'}
        ]}]
        model = type('OrderModel', (), {})()
        machine = self.machine_cls(model=model, states=states, initial='idle', auto_transitions=False)
        machine.add_transition('start_order', 'idle',
                               ['order_payment_pending', 'order_stock_pending'])

        self.assertTrue(model.start_order())
        self.assertEqual(['order_payment_pending', 'order_stock_pending'], model.state)


@skipIf(pgv is None, "pygraphviz is not available")
class TestParallelWithPyGraphviz(TestParallel):

    def setUp(self):
        class PGVMachine(HierarchicalGraphMachine):

            def __init__(self, *args, **kwargs):
                kwargs['graph_engine'] = "pygraphviz"
                super(PGVMachine, self).__init__(*args, **kwargs)

        super(TestParallelWithPyGraphviz, self).setUp()
        self.machine_cls = PGVMachine


@skipIf(gv is None, "graphviz is not available")
class TestParallelWithGraphviz(TestParallel):

    def setUp(self):
        class GVMachine(HierarchicalGraphMachine):

            def __init__(self, *args, **kwargs):
                kwargs['graph_engine'] = "graphviz"
                super(GVMachine, self).__init__(*args, **kwargs)

        super(TestParallelWithGraphviz, self).setUp()
        self.machine_cls = GVMachine
