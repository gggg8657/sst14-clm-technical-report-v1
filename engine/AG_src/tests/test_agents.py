"""
test_agents.py
Tests for all 6 agent classes in the Co-Scientist pipeline.
"""

import sys
import unittest
from abc import ABC

sys.path.insert(0, '/Users/kimsoyeon/ai4sci_kaeri')

from AG_src.agents.base_agent import BaseAgent, AgentMessage, MessageType
from AG_src.agents.planner import PlannerAgent
from AG_src.agents.builder import BuilderAgent, BuilderStepResult
from AG_src.agents.qc_ranker import QCRankerAgent
from AG_src.agents.critic import ScientistCriticAgent, FAILURE_ACTION_MAP
from AG_src.agents.reporter import ReporterAgent
from AG_src.agents.diversity_manager import DiversityManagerAgent


class TestBaseAgentAbstract(unittest.TestCase):
    def test_base_agent_is_abstract(self):
        """BaseAgent cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseAgent(name="x", role="y", description="z")  # type: ignore[abstract]


class TestMessageTypeEnum(unittest.TestCase):
    def test_message_type_enum(self):
        """MessageType has expected values."""
        self.assertEqual(MessageType.INFO.value, "info")
        self.assertEqual(MessageType.REQUEST.value, "request")
        self.assertEqual(MessageType.DECISION.value, "decision")
        self.assertEqual(MessageType.ALERT.value, "alert")


class TestAgentMessageCreation(unittest.TestCase):
    def test_agent_message_creation(self):
        """AgentMessage creates correctly with sender, receiver, msg_type, content."""
        msg = AgentMessage(
            sender="Planner",
            receiver="Builder",
            content={"key": "value"},
            message_type=MessageType.REQUEST,
        )
        self.assertEqual(msg.sender, "Planner")
        self.assertEqual(msg.receiver, "Builder")
        self.assertEqual(msg.message_type, MessageType.REQUEST)
        self.assertIn("key", msg.content)
        self.assertIsInstance(msg.timestamp, str)
        self.assertTrue(len(msg.timestamp) > 0)


class TestPlannerAgent(unittest.TestCase):
    def setUp(self):
        self.agent = PlannerAgent()

    def test_planner_instantiates(self):
        """PlannerAgent() creates successfully."""
        self.assertIsInstance(self.agent, PlannerAgent)
        self.assertIsInstance(self.agent, BaseAgent)

    def test_planner_has_execute(self):
        """PlannerAgent has execute() method."""
        self.assertTrue(hasattr(self.agent, 'execute'))
        self.assertTrue(callable(self.agent.execute))

    def test_planner_has_create_initial_plan(self):
        """PlannerAgent has create_initial_plan() method."""
        self.assertTrue(hasattr(self.agent, 'create_initial_plan'))
        self.assertTrue(callable(self.agent.create_initial_plan))


class TestBuilderAgent(unittest.TestCase):
    def setUp(self):
        self.agent = BuilderAgent(dry_run=True)

    def test_builder_instantiates(self):
        """BuilderAgent() creates successfully."""
        self.assertIsInstance(self.agent, BuilderAgent)
        self.assertIsInstance(self.agent, BaseAgent)

    def test_builder_has_execute_pipeline(self):
        """BuilderAgent has execute_pipeline() method."""
        self.assertTrue(hasattr(self.agent, 'execute_pipeline'))
        self.assertTrue(callable(self.agent.execute_pipeline))

    def test_builder_step_result_renamed(self):
        """BuilderStepResult (not StepResult) is the correct class name."""
        result = BuilderStepResult(step_id="step01_test", status="success")
        self.assertIsInstance(result, BuilderStepResult)
        self.assertEqual(result.step_id, "step01_test")
        self.assertEqual(result.status, "success")
        # Verify the class is named BuilderStepResult, not StepResult
        self.assertEqual(result.__class__.__name__, "BuilderStepResult")


class TestQCRankerAgent(unittest.TestCase):
    def setUp(self):
        self.agent = QCRankerAgent()

    def test_qc_ranker_instantiates(self):
        """QCRankerAgent() creates successfully."""
        self.assertIsInstance(self.agent, QCRankerAgent)
        self.assertIsInstance(self.agent, BaseAgent)

    def test_qc_ranker_has_execute(self):
        """QCRankerAgent has execute() method."""
        self.assertTrue(hasattr(self.agent, 'execute'))
        self.assertTrue(callable(self.agent.execute))


class TestScientistCriticAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ScientistCriticAgent()

    def test_critic_instantiates(self):
        """ScientistCriticAgent() creates successfully."""
        self.assertIsInstance(self.agent, ScientistCriticAgent)
        self.assertIsInstance(self.agent, BaseAgent)

    def test_critic_has_failure_action_map(self):
        """FAILURE_ACTION_MAP is a dict-like attribute at module level."""
        self.assertIsInstance(FAILURE_ACTION_MAP, dict)
        self.assertTrue(len(FAILURE_ACTION_MAP) > 0)
        # Each entry should be a dict with at least 'actions' key
        for key, value in FAILURE_ACTION_MAP.items():
            self.assertIsInstance(value, dict)
            self.assertIn('actions', value)


class TestReporterAgent(unittest.TestCase):
    def test_reporter_instantiates(self):
        """ReporterAgent() creates successfully."""
        agent = ReporterAgent()
        self.assertIsInstance(agent, ReporterAgent)
        self.assertIsInstance(agent, BaseAgent)


class TestDiversityManagerAgent(unittest.TestCase):
    def test_diversity_manager_instantiates(self):
        """DiversityManagerAgent() creates successfully."""
        agent = DiversityManagerAgent()
        self.assertIsInstance(agent, DiversityManagerAgent)
        self.assertIsInstance(agent, BaseAgent)


class TestAllAgentsHaveExecute(unittest.TestCase):
    def test_all_agents_have_execute(self):
        """All 6 concrete agents have execute() method."""
        agents = [
            PlannerAgent(),
            BuilderAgent(dry_run=True),
            QCRankerAgent(),
            ScientistCriticAgent(),
            ReporterAgent(),
            DiversityManagerAgent(),
        ]
        for agent in agents:
            with self.subTest(agent=agent.__class__.__name__):
                self.assertTrue(
                    hasattr(agent, 'execute'),
                    f"{agent.__class__.__name__} missing execute()"
                )
                self.assertTrue(
                    callable(agent.execute),
                    f"{agent.__class__.__name__}.execute is not callable"
                )


if __name__ == '__main__':
    unittest.main()
