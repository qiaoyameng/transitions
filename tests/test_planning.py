from __future__ import absolute_import

from enum import Enum
from unittest import TestCase

from transitions import Machine
from transitions.experimental.planning import (
    CoverageGap,
    PlanExecution,
    Step,
    TransitionPair,
    execute_plan,
    execute_plans,
    generate_transition_pair_plans,
    verify_coverage,
)


class TestPlanning(TestCase):

    def setUp(self):
        self.states = ['idle', 'running', 'stopped', 'fault']
        self.transitions = [
            ['start', 'idle', 'running'],
            ['stop', 'running', 'stopped'],
            ['restart', 'stopped', 'running'],
            ['fail', 'running', 'fault'],
            ['recover', 'fault', 'idle'],
            ['reset', 'stopped', 'idle'],
            ['reset', 'fault', 'idle'],
        ]

    def machine(self, reset_cost=None, **kwargs):
        machine = Machine(
            states=self.states,
            transitions=self.transitions,
            initial='idle',
            auto_transitions=False,
            **kwargs
        )
        reset = None if reset_cost is None else ('reset', reset_cost)
        return machine, generate_transition_pair_plans(
            machine,
            costs={'recover': 5},
            reset=reset
        )

    def pair(self, plans, source, dest):
        source_transition = self.transition_obj(plans, source)
        target_transition = self.transition_obj(plans, dest)
        pair = TransitionPair(source_transition, target_transition)
        self.assertIn(pair, plans.pairs)
        return pair

    @staticmethod
    def transition_obj(plans, name):
        for plan in plans.plans:
            for step in plan.steps:
                if step.trigger == name:
                    return step.transition
        raise AssertionError(name)

    @staticmethod
    def executed_pairs(executions):
        return {pair for execution in executions for pair in execution.pairs}

    def test_plans_cover_reachable_adjacent_transition_pairs(self):
        machine, plans = self.machine()
        expected = {
            self.pair(plans, 'start', 'stop'),
            self.pair(plans, 'stop', 'restart'),
            self.pair(plans, 'restart', 'stop'),
            self.pair(plans, 'fail', 'recover'),
            self.pair(plans, 'recover', 'start'),
        }
        self.assertTrue(expected.issubset(set(plans.pairs)))
        self.assertEqual([], list(plans.gaps))
        self.assertFalse(plans.optimal)

        executions = execute_plans(machine, plans.plans)
        self.assertTrue(all(isinstance(execution, PlanExecution) for execution in executions))
        self.assertTrue(expected.issubset(self.executed_pairs(executions)))
        required, missing = verify_coverage(executions, plans.pairs)
        self.assertEqual(set(plans.pairs), set(required))
        self.assertEqual(set(), missing)

    def test_enum_state_configurations_are_canonicalized(self):
        class Status(Enum):
            idle = 'idle'
            running = 'running'

        machine = Machine(
            states=list(Status),
            transitions=[['start', Status.idle, Status.running],
                         ['stop', Status.running, Status.idle]],
            initial=Status.idle,
            auto_transitions=False
        )
        plans = generate_transition_pair_plans(machine, reset=None)
        self.assertTrue(plans.pairs)
        _, missing = verify_coverage(execute_plans(machine, plans.plans), plans.pairs)
        self.assertEqual(set(), missing)

    def test_expensive_reset_prefers_real_normal_recovery(self):
        machine, plans = self.machine(reset_cost=100)
        self.assertTrue(plans.plans)
        self.assertEqual([], [step for plan in plans.plans for step in plan.steps if step.reset])
        self.assertIn(
            self.pair(plans, 'stop', 'restart'),
            self.executed_pairs(execute_plans(machine, plans.plans))
        )

        cheap_machine = Machine(
            states=self.states,
            transitions=self.transitions,
            initial='idle',
            auto_transitions=False
        )
        cheap_plans = generate_transition_pair_plans(
            cheap_machine,
            costs={'recover': 100},
            reset=('reset', 0)
        )
        self.assertTrue(any(step.reset for plan in cheap_plans.plans for step in plan.steps))
        self.assertTrue(all(isinstance(step, Step) and step.cost >= 0
                            for plan in cheap_plans.plans for step in plan.steps))

    def test_reset_steps_are_marked_and_split_pairs(self):
        machine = Machine(
            states=self.states,
            transitions=self.transitions,
            initial='idle',
            auto_transitions=False
        )
        plans = generate_transition_pair_plans(machine, costs={'recover': 100}, reset=('reset', 0))
        execution = execute_plan(machine, plans.plans[0])
        reset_indexes = [index for index, step in enumerate(execution.steps) if step.reset]
        self.assertTrue(reset_indexes)
        for index in reset_indexes:
            self.assertNotIn(
                TransitionPair(execution.transitions[index - 1], execution.transitions[index]),
                execution.pairs
            )
            if index + 1 < len(execution.transitions):
                self.assertNotIn(
                    TransitionPair(execution.transitions[index], execution.transitions[index + 1]),
                    execution.pairs
                )

    def test_conditional_transitions_are_reported_as_gaps(self):
        machine = Machine(
            states=['idle', 'running'],
            transitions=[
                {'trigger': 'start', 'source': 'idle', 'dest': 'running', 'conditions': 'never'},
                ['stop', 'running', 'idle'],
            ],
            initial='idle',
            auto_transitions=False
        )
        machine.never = lambda: False
        plans = generate_transition_pair_plans(machine)
        self.assertEqual((), plans.plans)
        gap = plans.gaps[0]
        self.assertEqual('conditional', gap.kind)
        self.assertEqual('start', gap.trigger)
        self.assertTrue(gap.transition is machine.events['start'].transitions['idle'][0])

    def test_queued_machines_are_reported_as_gap(self):
        machine = Machine(
            states=['idle', 'running'],
            transitions=[['start', 'idle', 'running'], ['stop', 'running', 'idle']],
            initial='idle',
            auto_transitions=False,
            queued=True
        )
        plans = generate_transition_pair_plans(machine)
        self.assertEqual((), plans.pairs)
        self.assertEqual(CoverageGap('unsupported', plans.gaps[0].reason, None, None, None), plans.gaps[0])

    def test_hierarchical_machine_expands_active_configurations(self):
        from transitions.extensions import HierarchicalMachine

        states = [
            'idle',
            {
                'name': 'running',
                'parallel': [
                    {'name': 'motor', 'children': ['off', 'on'], 'initial': 'off'},
                    {'name': 'valve', 'children': ['closed', 'open'], 'initial': 'closed'},
                ],
            },
        ]
        transitions = [
            ['start', 'idle', 'running'],
            ['motor_on', 'running_motor_off', 'running_motor_on'],
            ['open_valve', 'running_valve_closed', 'running_valve_open'],
            ['stop', 'running', 'idle'],
        ]
        machine = HierarchicalMachine(
            states=states,
            transitions=transitions,
            initial='idle',
            auto_transitions=False
        )
        plans = generate_transition_pair_plans(machine, reset=None)
        self.assertTrue(plans.pairs)
        self.assertEqual([], list(plans.gaps))
        executions = execute_plans(machine, plans.plans)
        _, missing = verify_coverage(executions, plans.pairs)
        self.assertEqual(set(), missing)
