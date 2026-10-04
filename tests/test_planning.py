from unittest import TestCase

from transitions import Machine
from transitions.experimental.planning import ACTION, RESET
from transitions.experimental.planning import execute_test_plan
from transitions.experimental.planning import generate_transition_pair_test_plans
from transitions.extensions import HierarchicalMachine


class TestTransitionPairPlanning(TestCase):

    def test_plans_and_executes_stop_restart_pairs(self):
        machine = Machine(
            states=["idle", "running", "stopped"],
            initial="idle",
            auto_transitions=False,
            transitions=[
                ["start", "idle", "running"],
                ["stop", "running", "stopped"],
                ["restart", "stopped", "running"],
            ])

        plan = generate_transition_pair_test_plans(machine)[0]
        execution = execute_test_plan(plan, machine)

        self.assertTrue(execution.successful)
        self.assertEqual(execution.covered_pairs, set(plan.required_pairs))
        stop_step = next(step for step in plan.steps if step.identity[0] == "stop")
        restart_step = next(step for step in plan.steps if step.identity[0] == "restart")
        self.assertEqual(restart_step, plan.steps[plan.steps.index(stop_step) + 1])
        self.assertEqual(ACTION, stop_step.kind)
        self.assertEqual(ACTION, restart_step.kind)
        adjacent = list(zip(plan.steps, plan.steps[1:]))
        self.assertTrue(any(first.identity[0] == "start" and second.identity[0] == "stop"
                            for first, second in adjacent))
        self.assertTrue(any(first.identity[0] == "stop" and second.identity[0] == "restart"
                            for first, second in adjacent))
        self.assertTrue(any(first.identity[0] == "restart" and second.identity[0] == "stop"
                            for first, second in adjacent))

    def test_expensive_reset_is_only_used_for_dead_branch(self):
        machine = Machine(
            states=["idle", "running", "stopped", "error"],
            initial="idle",
            auto_transitions=False,
            transitions=[
                ["start", "idle", "running"],
                ["stop", "running", "stopped"],
                ["restart", "stopped", "idle"],
                ["fail", "running", "error"],
                ["fix", "error", "idle"],
            ])

        plan = generate_transition_pair_test_plans(
            machine,
            costs={"restart": 1},
            reset={"trigger": "fix", "cost": 100})[0]
        reset_steps = [step for step in plan.steps if step.kind == RESET]

        self.assertEqual(1, len(reset_steps))
        self.assertEqual("error", reset_steps[0].source)
        self.assertEqual("idle", reset_steps[0].dest)
        self.assertEqual(100, reset_steps[0].cost)
        self.assertTrue(execute_test_plan(plan, machine).successful)

    def test_planner_does_not_start_again_from_an_arbitrary_state(self):
        machine = Machine(
            states=["start", "branch", "end"],
            initial="start",
            auto_transitions=False,
            transitions=[
                ["go", "start", "branch"],
                ["finish", "branch", "end"],
                ["return", "branch", "start"],
            ])

        plan = generate_transition_pair_test_plans(machine)[0]
        for index, step in enumerate(plan.steps[1:]):
            self.assertEqual(plan.steps[index].dest, step.source)

    def test_hierarchical_machine_expands_supported_configurations(self):
        machine = HierarchicalMachine(
            states=["off", {"name": "motor", "initial": "idle", "children": ["idle", "running"]}],
            initial="off",
            auto_transitions=False,
            transitions=[
                ["power", "off", "motor"],
                ["start", "motor_idle", "motor_running"],
                ["stop", "motor_running", "motor_idle"],
                ["restart", "motor_running", "motor_idle"],
                ["poweroff", "motor", "off"],
            ])

        plan = generate_transition_pair_test_plans(machine)[0]
        execution = execute_test_plan(plan, machine)

        self.assertEqual("off", plan.initial_configuration)
        self.assertEqual([], plan.coverage_gaps)
        self.assertTrue(execution.successful)
        self.assertEqual(set(plan.required_pairs), execution.covered_pairs)

    def test_conditional_and_parallel_behaviour_are_reported_as_gaps(self):
        conditional_machine = Machine(
            states=["A", "B"],
            initial="A",
            auto_transitions=False,
            transitions=[["go", "A", "B", lambda: True]])
        conditional_plan = generate_transition_pair_test_plans(conditional_machine)[0]

        self.assertEqual("conditional_transition", conditional_plan.coverage_gaps[0].kind)
        self.assertEqual([], conditional_plan.events)

        parallel_machine = HierarchicalMachine(
            states=["A", {"name": "P", "parallel": ["x", "y"]}],
            initial="A",
            auto_transitions=False)
        parallel_plan = generate_transition_pair_test_plans(parallel_machine)[0]

        self.assertEqual("parallel_configuration", parallel_plan.coverage_gaps[0].kind)

    def test_steps_distinguish_actions_and_resets(self):
        machine = Machine(
            states=["idle", "running", "error"],
            initial="idle",
            auto_transitions=False,
            transitions=[
                ["start", "idle", "running"],
                ["fail", "running", "error"],
                ["fix", "error", "idle"],
            ])

        plan = generate_transition_pair_test_plans(
            machine, reset={"trigger": "fix", "cost": 3})[0]

        self.assertIn(RESET, [step.kind for step in plan.steps])
        self.assertTrue(all(step.kind in (ACTION, RESET) for step in plan.steps))
        self.assertTrue(execute_test_plan(plan, machine).successful)

    def test_external_reset_operation_is_executed_and_marked(self):
        class Model:
            state = "idle"
            reset_count = 0

            def external_reset(self):
                self.reset_count += 1

        model = Model()
        machine = Machine(
            model=model,
            states=["idle", "running", "error"],
            initial="idle",
            auto_transitions=False,
            transitions=[
                ["start", "idle", "running"],
                ["fail", "running", "error"],
            ])

        plan = generate_transition_pair_test_plans(machine, reset={
            "identity": "external_reset",
            "operation": model.external_reset,
            "dest": "idle",
            "cost": 9,
            "sources": ["error"],
        })[0]
        reset_steps = [step for step in plan.steps if step.kind == RESET]

        self.assertEqual(1, len(reset_steps))
        self.assertIsNone(reset_steps[0].transition)
        execution = execute_test_plan(plan, machine)

        self.assertTrue(execution.successful)
        self.assertEqual(1, model.reset_count)
        self.assertEqual("external_reset", execution.transition_identities[-2][0])
