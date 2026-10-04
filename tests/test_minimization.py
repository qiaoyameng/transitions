"""Tests for the experimental deterministic machine minimization tool."""

from itertools import product
from unittest import TestCase

from transitions import Machine, State
from transitions.experimental.minimization import (
    REJECTED, MinimizationError, minimize_machine, simulate)


def build_machine(states, transitions, initial=None, **kwargs):
    if initial is None:
        initial = states[0].name if isinstance(states[0], State) else states[0]
    return Machine(states=states, initial=initial, transitions=transitions,
                   auto_transitions=False, **kwargs)


def replay(result, state, events):
    return simulate(result._model, state, events)  # pylint: disable=protected-access


class MinimizationTests(TestCase):

    def test_merge_alias_states(self):
        # WAIT1 and WAIT2 are the same state under two names.
        states = ["IDLE", "WAIT1", "WAIT2", "DONE"]
        transitions = [
            ["connect", "IDLE", "WAIT1"],
            ["retry", "WAIT1", "WAIT2"],
            ["retry", "WAIT2", "WAIT1"],
            ["connected", "WAIT1", "DONE"],
            ["connected", "WAIT2", "DONE"],
            ["reset", "DONE", "IDLE"],
            ["reset", "WAIT1", "IDLE"],
            ["reset", "WAIT2", "IDLE"],
        ]
        machine = build_machine(states, transitions)
        labels = {name: name if name in ("IDLE", "DONE") else "WAIT"
                  for name in states}
        result = minimize_machine(machine, labels=labels)

        self.assertEqual(set(result.quotient.states), {"IDLE", "WAIT1", "DONE"})
        self.assertEqual(result.mapping, {
            "IDLE": "IDLE", "WAIT1": "WAIT1", "WAIT2": "WAIT1", "DONE": "DONE"})
        self.assertEqual(result.partition["WAIT1"], frozenset({"WAIT1", "WAIT2"}))
        self.assertTrue(result.equivalent("WAIT1", "WAIT2"))
        self.assertFalse(result.equivalent("IDLE", "WAIT1"))
        self.assertEqual(result.quotient.initial, "IDLE")

    def test_representative_is_earliest_declared_state(self):
        states = ["A", "B", "C"]
        transitions = [["go", "A", "B"], ["go", "B", "A"],
                       ["go", "C", "B"]]
        machine = build_machine(states, transitions)
        result = minimize_machine(machine, labels={s: "L" for s in states})
        # A and B form a two-cycle with the same output; C joins A after one go.
        self.assertEqual(result.mapping["B"], "A")

    def test_missing_transition_distinct_from_self_loop(self):
        machine = build_machine(["P", "Q"], [["tick", "P", "P"]], initial="P")
        result = minimize_machine(machine, labels={"P": "L", "Q": "L"})
        self.assertFalse(result.equivalent("P", "Q"))
        witness = result.witness("P", "Q")
        self.assertEqual(witness, ["tick"])
        trace_p = replay(result, "P", witness)
        trace_q = replay(result, "Q", witness)
        self.assertNotEqual(trace_p, trace_q)
        self.assertEqual(trace_q[1][0][1], ("rejected",))
        # The rejected event keeps the state (another rejection follows).
        trace_q2 = replay(result, "Q", ["tick", "tick"])
        self.assertEqual([step[1] for step in trace_q2[1]],
                         [("rejected",), ("rejected",)])

    def test_internal_transition_distinct_from_rejection(self):
        transitions = [
            {"trigger": "nop", "source": "I", "dest": None, "after": "log"},
            ["go", "I", "K"],
            ["go", "J", "K"],
            ["go", "K", "K"],
        ]
        machine = build_machine(["I", "J", "K"], transitions, initial="I")
        result = minimize_machine(machine, labels={s: "L" for s in ["I", "J", "K"]})
        self.assertFalse(result.equivalent("I", "J"))
        witness = result.witness("I", "J")
        self.assertEqual(witness, ["nop"])
        self.assertNotEqual(replay(result, "I", witness), replay(result, "J", witness))

    def test_enter_actions_participate_in_comparison(self):
        states = [State("A", on_enter="enter_a"),
                  State("B", on_enter="enter_b"), State("C")]
        transitions = [["go", "A", "C"], ["go", "B", "C"],
                       ["back", "C", "B"], ["again", "C", "A"]]
        machine = build_machine(states, transitions)
        result = minimize_machine(machine, labels={s: "L" for s in ["A", "B", "C"]})
        self.assertFalse(result.equivalent("A", "B"))
        witness = result.witness("A", "B")
        self.assertIsNotNone(witness)
        self.assertNotEqual(replay(result, "A", witness),
                            replay(result, "B", witness))
        self.assertIsNone(result.witness("A", "A"))

    def test_exit_and_transition_actions_participate(self):
        states = [State("S1", on_exit="bye1"),
                  State("S2", on_exit="bye2"), State("T")]
        transitions = [["go", "S1", "T"], ["go", "S2", "T"],
                       ["back", "T", "S1"]]
        machine = build_machine(states, transitions, initial="S1")
        result = minimize_machine(machine, labels={"S1": "s", "S2": "s", "T": "t"})
        self.assertFalse(result.equivalent("S1", "S2"))
        witness = result.witness("S1", "S2")
        self.assertEqual(witness, ["go"])
        self.assertNotEqual(replay(result, "S1", witness),
                            replay(result, "S2", witness))

    def test_callbacks_are_never_executed(self):
        def boom(*_args, **_kwargs):
            raise AssertionError("callbacks must not be executed")

        states = [State("A", on_enter=boom, on_exit=boom),
                  State("B", on_enter=boom, on_exit=boom)]
        transitions = [
            {"trigger": "go", "source": "A", "dest": "B",
             "prepare": boom, "before": boom, "after": boom},
            {"trigger": "go", "source": "B", "dest": "A",
             "prepare": boom, "before": boom, "after": boom},
        ]
        machine = build_machine(states, transitions)
        result = minimize_machine(machine, labels={s: "L" for s in ["A", "B"]},
                                  action_key=lambda value: getattr(value, "__name__", value))
        self.assertEqual(set(result.quotient.states), {"A"})

    def test_default_action_key_rejects_callables(self):
        states = [State("A", on_enter=lambda: None), "B"]
        machine = build_machine(states, [["go", "A", "B"], ["go", "B", "A"]])
        with self.assertRaises(MinimizationError):
            minimize_machine(machine, labels={"A": "a", "B": "b"})

    def test_reject_conditions(self):
        for keyword in ("conditions", "unless"):
            transitions = [
                {"trigger": "go", "source": "A", "dest": "B", keyword: "guard"},
                ["go", "B", "A"],
            ]
            machine = build_machine(["A", "B"], transitions)
            with self.assertRaises(MinimizationError):
                minimize_machine(machine, labels={"A": "a", "B": "b"})

    def test_reject_multiple_candidates(self):
        transitions = [["go", "A", "B"], ["go", "A", "C"],
                       ["go", "B", "A"], ["go", "C", "A"]]
        machine = build_machine(["A", "B", "C"], transitions)
        with self.assertRaises(MinimizationError):
            minimize_machine(machine, labels={"A": "a", "B": "b", "C": "c"})

    def test_reject_nested_states(self):
        from transitions.extensions.nesting import HierarchicalMachine
        machine = HierarchicalMachine(
            states=["A", {"name": "B", "children": ["1", "2"]}], initial="A")
        with self.assertRaises(MinimizationError):
            minimize_machine(machine, labels={"A": "a", "B": "b"})

    def test_reject_conflicting_rejection_policy(self):
        states = [State("A", ignore_invalid_triggers=True),
                  State("B", ignore_invalid_triggers=False)]
        machine = build_machine(states, [["go", "A", "B"]])
        with self.assertRaises(MinimizationError):
            minimize_machine(machine, labels={"A": "a", "B": "b"})

    def test_explicit_alphabet_validation(self):
        machine = build_machine(["A", "B"], [["go", "A", "B"]])
        with self.assertRaises(MinimizationError):
            minimize_machine(machine, labels={"A": "a", "B": "b"},
                             alphabet=["go", "nope"])
        with self.assertRaises(MinimizationError):
            minimize_machine(machine, labels={"A": "a", "B": "b"},
                             alphabet=["go", "go"])

    def test_quotient_behaves_like_original_on_short_strings(self):
        states = ["IDLE", "WAIT1", "WAIT2", "DONE", "OTHER"]
        transitions = [
            ["connect", "IDLE", "WAIT1"],
            ["retry", "WAIT1", "WAIT2"],
            ["retry", "WAIT2", "WAIT1"],
            ["connected", "WAIT1", "DONE"],
            ["connected", "WAIT2", "DONE"],
            ["reset", "DONE", "IDLE"],
            ["reset", "WAIT1", "IDLE"],
            ["reset", "WAIT2", "IDLE"],
            ["connect", "OTHER", "DONE"],
        ]
        machine = build_machine(states, transitions)
        labels = {name: ("ready" if name == "DONE" else
                         "waiting" if name in ("WAIT1", "WAIT2") else
                         "idle" if name == "IDLE" else "other")
                  for name in states}
        result = minimize_machine(machine, labels=labels)
        self.assertIsNone(result.verify_quotient())

    def test_witness_always_distinct_non_equivalent_pairs(self):
        # Every pair that the partition keeps apart must come with a concrete
        # event string that produces different observable sequences.
        states = ["S0", "S1", "S2", "S3"]
        transitions = [
            ["a", "S0", "S1"], ["b", "S0", "S2"],
            ["a", "S1", "S2"], ["b", "S1", "S3"],
            ["a", "S2", "S3"], ["b", "S2", "S2"],
            ["a", "S3", "S0"], ["b", "S3", "S1"],
        ]
        machine = build_machine(states, transitions, initial="S0")
        labels = {name: "L" for name in states}
        result = minimize_machine(machine, labels=labels)
        for left, right in product(states, states):
            if left >= right or result.equivalent(left, right):
                continue
            witness = result.witness(left, right)
            self.assertIsNotNone(witness, (left, right))
            self.assertNotEqual(replay(result, left, witness),
                                replay(result, right, witness),
                                (left, right, witness))

    def test_witness_for_label_difference_is_empty(self):
        machine = build_machine(["A", "B"], [["go", "A", "B"], ["go", "B", "A"]])
        result = minimize_machine(machine, labels={"A": "a", "B": "b"})
        self.assertEqual(result.witness("A", "B"), [])

    def test_quotient_machine_is_runnable(self):
        states = [State("IDLE", on_enter="e_idle", on_exit="x_idle"),
                  State("WAIT1", on_enter="e_wait", on_exit="x_wait"),
                  State("WAIT2", on_enter="e_wait", on_exit="x_wait"),
                  State("DONE", on_enter="e_done", on_exit="x_done")]
        transitions = [
            ["connect", "IDLE", "WAIT1"],
            ["retry", "WAIT1", "WAIT2"],
            ["retry", "WAIT2", "WAIT1"],
            {"trigger": "connected", "source": "WAIT1", "dest": "DONE", "after": "notify"},
            {"trigger": "connected", "source": "WAIT2", "dest": "DONE", "after": "notify"},
            ["reset", "DONE", "IDLE"],
            ["reset", "WAIT1", "IDLE"],
            ["reset", "WAIT2", "IDLE"],
        ]
        machine = build_machine(states, transitions)
        labels = {"IDLE": "idle", "WAIT1": "wait", "WAIT2": "wait", "DONE": "done"}
        result = minimize_machine(machine, labels=labels)

        calls = []

        class Model(object):
            state = None

            def e_idle(self):
                calls.append("e_idle")

            def x_idle(self):
                calls.append("x_idle")

            def e_wait(self):
                calls.append("e_wait")

            def x_wait(self):
                calls.append("x_wait")

            def e_done(self):
                calls.append("e_done")

            def x_done(self):
                calls.append("x_done")

            def notify(self):
                calls.append("notify")

        model = Model()
        result.quotient.add_model(model)
        model.connect()
        self.assertEqual(model.state, "WAIT1")
        self.assertEqual(calls, ["x_idle", "e_wait"])
        calls.clear()
        model.connected()
        self.assertEqual(model.state, "DONE")
        self.assertEqual(calls, ["x_wait", "e_done", "notify"])

    def test_explicit_labels_callable(self):
        machine = build_machine(["A", "B"], [["go", "A", "B"], ["go", "B", "A"]])
        result = minimize_machine(machine, labels=lambda state: state.name)
        self.assertFalse(result.equivalent("A", "B"))
        self.assertEqual(result.witness("A", "B"), [])

    def test_rejected_token_is_unique(self):
        self.assertEqual(REJECTED, REJECTED)
        # 'tick' is registered on B only; from A it is a missing transition.
        machine = build_machine(["A", "B"], [["tick", "B", "B"]], initial="A")
        result = minimize_machine(machine, labels={"A": "a", "B": "b"})
        trace = replay(result, "A", ["tick"])
        rejected_step = trace[1][0][1]
        self.assertEqual(rejected_step, ("rejected",))
        # A rejected event is structurally distinct from any taken self loop.
        self_loop = replay(result, "B", ["tick"])[1][0][1]
        self.assertEqual(self_loop[0], "taken")
        self.assertNotEqual(rejected_step, self_loop)
        self.assertIs(REJECTED, REJECTED)

    def test_missing_label_raises_clear_error(self):
        machine = build_machine(["A", "B"], [["go", "A", "B"]])
        with self.assertRaises(MinimizationError) as context:
            minimize_machine(machine, labels={"A": "a"})
        self.assertIn("B", str(context.exception))

    def test_empty_alphabet(self):
        machine = build_machine(["A", "B"], [["go", "A", "B"], ["go", "B", "A"]])
        result = minimize_machine(machine, labels={"A": "a", "B": "b"}, alphabet=[])
        self.assertEqual(list(result.quotient.states), ["A", "B"])
        self.assertIsNone(result.verify_quotient())

    def test_enum_state_names(self):
        from enum import Enum

        class Color(Enum):
            RED = 1
            AMBER = 2
            BLUE = 3

        machine = build_machine(
            list(Color),
            [["go", Color.RED, Color.AMBER],
             ["go", Color.BLUE, Color.AMBER],
             ["go", Color.AMBER, Color.RED]])
        result = minimize_machine(
            machine, labels={"RED": "stop", "BLUE": "stop", "AMBER": "go"})
        self.assertEqual(result.mapping["BLUE"], "RED")
        self.assertEqual(set(result.quotient.states), {"RED", "AMBER"})
        self.assertIsNone(result.verify_quotient())
