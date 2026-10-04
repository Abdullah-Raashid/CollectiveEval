import asyncio
import unittest
from collections.abc import Coroutine
from typing import Any

from collectiveeval.budget import BudgetLedger, InferenceBudget
from collectiveeval.context import StrategyContext
from collectiveeval.core import BenchmarkExample, ModelSpec, TaskType
from collectiveeval.metrics import score_prediction
from collectiveeval.providers import MockProvider
from collectiveeval.strategies import (
    AdaptiveRouterStrategy,
    CriticReviser,
    HeterogeneousPanel,
    MultiAgentDebate,
    SelfConsistency,
    SingleAgent,
)


def run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


def qa_example(**metadata: Any) -> BenchmarkExample:
    merged_metadata = {"difficulty": "easy", "split": "mock"}
    merged_metadata.update({"source": "synthetic_unit_test", "tags": ["qa"]})
    merged_metadata.update(metadata)
    return BenchmarkExample(
        id="qa",
        task_type=TaskType.GROUNDED_QA,
        input={
            "source_text": "第3条: 解約通知は30日前まで。",
            "question": "何日前?",
            "evidence_units": [{"id": "doc-3", "text": "第3条: 解約通知は30日前まで。"}],
        },
        gold={"answerable": True, "answer": "30日前", "evidence": ["doc-3"]},
        metadata=merged_metadata,
    )


def context(max_calls: int | None = None, *, mock_mode: str = "gold_fixture") -> StrategyContext:
    return StrategyContext(
        {"mock": MockProvider(mock_mode=mock_mode)},
        BudgetLedger(InferenceBudget(max_calls=max_calls), run_id="test-run"),
    )


class StrategyTests(unittest.TestCase):
    def assert_perfect_qa(self, output: dict) -> None:
        scores = score_prediction(qa_example(), output)
        self.assertEqual(scores["task_score"], 1.0)

    def test_single_agent(self) -> None:
        result = run(SingleAgent().run(qa_example(), context()))

        self.assertEqual(result.model_calls, 1)
        self.assert_perfect_qa(result.output)

    def test_self_consistency_majority(self) -> None:
        example = qa_example(
            mock_outputs=[
                {
                    "content": {
                        "answer": "不明",
                        "citations": [],
                        "abstain": False,
                        "confidence": 0.3,
                    },
                    "confidence": 0.3,
                },
                {
                    "content": {
                        "answer": "30日前",
                        "citations": ["doc-3"],
                        "abstain": False,
                        "confidence": 0.8,
                    },
                    "confidence": 0.8,
                },
                {
                    "content": {
                        "answer": "30日前",
                        "citations": ["doc-3"],
                        "abstain": False,
                        "confidence": 0.7,
                    },
                    "confidence": 0.7,
                },
                {
                    "content": {
                        "answer": "不明",
                        "citations": [],
                        "abstain": False,
                        "confidence": 0.2,
                    },
                    "confidence": 0.2,
                },
            ]
        )
        result = run(SelfConsistency(k=4).run(example, context()))

        self.assertEqual(result.model_calls, 4)
        self.assertEqual(result.output["answer"], "30日前")

    def test_critic_reviser_repairs_weak_generator(self) -> None:
        strategy = CriticReviser(generator=ModelSpec(model="mock-weak"))
        result = run(strategy.run(qa_example(), context()))

        self.assertEqual(result.metadata["revised"], True)
        self.assert_perfect_qa(result.output)

    def test_debate_uses_agents_times_rounds_budget(self) -> None:
        result = run(MultiAgentDebate(agents=3, rounds=2).run(qa_example(), context()))

        self.assertEqual(result.model_calls, 6)
        self.assertEqual(len(result.candidates), 6)
        self.assert_perfect_qa(result.output)

    def test_heterogeneous_panel(self) -> None:
        result = run(
            HeterogeneousPanel(
                models=[
                    ModelSpec(model="mock-weak", role="cheap"),
                    ModelSpec(model="mock-accurate", role="accurate"),
                    ModelSpec(model="mock-accurate-alt", role="alternate"),
                ]
            ).run(qa_example(), context())
        )

        self.assertEqual(result.model_calls, 3)
        self.assertEqual(result.output["answer"], "30日前")

    def test_adaptive_router_accepts_certain_and_escalates_uncertain(self) -> None:
        accepted = run(AdaptiveRouterStrategy().run(qa_example(), context()))
        self.assertEqual(accepted.metadata["route"], "accept")
        self.assertEqual(accepted.model_calls, 1)

        uncertain_strategy = AdaptiveRouterStrategy(cheap_model=ModelSpec(model="mock-uncertain"))
        escalated = run(uncertain_strategy.run(qa_example(difficulty="hard"), context(max_calls=6)))
        self.assertIn(escalated.metadata["route"], {"critic", "debate"})
        self.assertGreater(escalated.model_calls, 1)
        self.assert_perfect_qa(escalated.output)

    def test_default_mock_provider_is_not_gold_aware(self) -> None:
        result = run(SingleAgent().run(qa_example(), context(mock_mode="fixture")))

        self.assertNotEqual(result.output["answer"], "30日前")


if __name__ == "__main__":
    unittest.main()
