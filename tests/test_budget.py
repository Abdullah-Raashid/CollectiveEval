import unittest

from collectiveeval.budget import BudgetExceeded, BudgetLedger, InferenceBudget, ModelCallRecord


class BudgetTests(unittest.TestCase):
    def test_records_usage_until_budget_exceeded(self) -> None:
        ledger = BudgetLedger(InferenceBudget(max_calls=1, max_total_tokens=10))
        ledger.record_call(
            ModelCallRecord(
                run_id="run",
                example_id="ex",
                strategy="single",
                provider="mock",
                model="mock",
                role="generator",
                input_tokens=3,
                output_tokens=4,
                latency_ms=2.0,
                estimated_cost_usd=0.01,
                finish_reason="stop",
            )
        )

        self.assertEqual(ledger.model_calls, 1)
        self.assertEqual(ledger.total_tokens, 7)
        with self.assertRaises(BudgetExceeded):
            ledger.ensure_can_start_call(1)

    def test_retains_completed_token_overrun(self) -> None:
        ledger = BudgetLedger(InferenceBudget(max_total_tokens=5))
        ledger.record_call(
            ModelCallRecord(
                run_id="run",
                example_id="ex",
                strategy="single",
                provider="mock",
                model="mock",
                role="generator",
                input_tokens=3,
                output_tokens=3,
                latency_ms=0.0,
                estimated_cost_usd=0.0,
                finish_reason="stop",
            )
        )
        self.assertEqual(ledger.model_calls, 1)
        self.assertEqual(ledger.total_tokens, 6)
        self.assertTrue(ledger.records[0].metadata["post_call_budget_overrun"])
        with self.assertRaises(BudgetExceeded):
            ledger.reserve_call_start(0, 1)


if __name__ == "__main__":
    unittest.main()
