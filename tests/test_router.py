import unittest

from collectiveeval.core import BenchmarkExample, TaskType
from collectiveeval.router import (
    HeuristicRouter,
    LearnedRouter,
    RouterTrainingExample,
    extract_uncertainty_signals,
)


class RouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.example = BenchmarkExample(
            id="qa",
            task_type=TaskType.GROUNDED_QA,
            input={
                "question": "何日前?",
                "source_text": "第3条: 解約通知は30日前まで。",
                "evidence_units": [{"id": "doc-3", "text": "第3条: 解約通知は30日前まで。"}],
            },
            gold={"answerable": True, "answer": "30日前", "evidence": ["doc-3"]},
            metadata={
                "source": "synthetic_unit_test",
                "difficulty": "hard",
                "tags": ["qa"],
                "split": "dev",
            },
        )

    def test_heuristic_router_escalates_uncertain_missing_citation(self) -> None:
        signals = extract_uncertainty_signals(
            self.example,
            {"answer": "30日前", "citations": [], "abstain": False, "confidence": 0.2},
            confidence=0.2,
            model_id="mock/cheap-weak",
        )
        self.assertIn(HeuristicRouter().route(signals), {"critic", "debate"})

    def test_learned_router_trains_only_on_dev(self) -> None:
        good = extract_uncertainty_signals(
            self.example,
            {"answer": "30日前", "citations": ["doc-3"], "abstain": False, "confidence": 0.95},
            confidence=0.95,
            model_id="mock/strong",
        )
        bad = extract_uncertainty_signals(
            self.example,
            {"answer": "不明", "citations": [], "abstain": False, "confidence": 0.1},
            confidence=0.1,
            model_id="mock/cheap-weak",
        )
        router = LearnedRouter()
        router.train(
            [
                RouterTrainingExample(good, escalation_helped=False, split="dev"),
                RouterTrainingExample(bad, escalation_helped=True, split="dev"),
            ]
        )

        self.assertGreaterEqual(router.escalation_probability(bad), 0.0)
        with self.assertRaises(ValueError):
            LearnedRouter().train([RouterTrainingExample(good, False, split="test")])


if __name__ == "__main__":
    unittest.main()
