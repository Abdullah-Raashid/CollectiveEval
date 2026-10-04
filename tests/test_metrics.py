import unittest

from collectiveeval.core import BenchmarkExample, TaskType
from collectiveeval.metrics import citation_scores, exact_match, score_prediction, token_f1


class MetricTests(unittest.TestCase):
    def test_text_metrics(self) -> None:
        self.assertEqual(exact_match(" 30日前 ", "30日前"), 1.0)
        self.assertGreater(token_f1("契約更新", "契約"), 0.0)

    def test_citation_scores(self) -> None:
        scores = citation_scores(["a", "b"], ["b", "c"])
        self.assertEqual(scores["citation_precision"], 0.5)
        self.assertEqual(scores["citation_recall"], 0.5)
        self.assertEqual(scores["citation_f1"], 0.5)

    def test_grounded_qa_score(self) -> None:
        example = BenchmarkExample(
            id="qa",
            task_type=TaskType.GROUNDED_QA,
            input={
                "question": "いつ?",
                "source_text": "契約書第3条: 30日前までに通知する。",
                "evidence_units": [{"id": "doc-3", "text": "契約書第3条: 30日前までに通知する。"}],
            },
            gold={"answerable": True, "answer": "30日前", "evidence": ["doc-3"]},
            metadata={
                "source": "synthetic_unit_test",
                "difficulty": "easy",
                "tags": ["qa"],
                "split": "mock",
            },
        )
        output = {"answer": "30日前", "citations": ["doc-3"], "abstain": False, "confidence": 0.9}
        scores = score_prediction(example, output)

        self.assertEqual(scores["schema_compliance"], 1.0)
        self.assertEqual(scores["task_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
