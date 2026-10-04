import os
import unittest
from pathlib import Path
from unittest.mock import patch

from collectiveeval.core import ModelSpec, ProviderRequest, UsageSource
from collectiveeval.datasets import BENCHMARK_V3_1_VERSION, load_jsonl
from collectiveeval.pilot import (
    DEV_PATH,
    Phase8Settings,
    base_model_config,
    select_pilot_examples,
    select_smoke_examples,
    settings_from_env,
    strategy_configs,
    verify_phase8_dev_freeze,
    verify_router_artifact_scope,
)
from collectiveeval.providers import (
    OpenAICompatibleProvider,
    build_provider_registry,
    normalize_openai_response,
)


class Phase8PilotTests(unittest.TestCase):
    def test_freeze_and_router_scope_checks_pass(self) -> None:
        freeze = verify_phase8_dev_freeze()
        router_scope = verify_router_artifact_scope()

        self.assertTrue(freeze["ok"])
        self.assertEqual(freeze["actual"]["version"], BENCHMARK_V3_1_VERSION)
        self.assertFalse(freeze["test_file_opened"])
        self.assertTrue(router_scope["ok"])

    def test_smoke_and_pilot_selection_are_dev_only_and_stratified(self) -> None:
        dev = {example.id: example for example in load_jsonl(DEV_PATH)}
        smoke_ids = select_smoke_examples()
        selection = select_pilot_examples(sample_size=32)

        self.assertEqual(len(smoke_ids), 4)
        self.assertEqual(selection["sample_size"], 32)
        self.assertEqual(set(selection["counts"]["task_type"].values()), {8})
        self.assertEqual(selection["counts"]["difficulty"], {"easy": 8, "hard": 16, "medium": 8})
        for task_audit in selection["audit"]["by_task"].values():
            self.assertEqual(task_audit["difficulty"], {"easy": 2, "hard": 4, "medium": 2})
            self.assertGreaterEqual(len(task_audit["surface_form_family"]), 4)
        for example_id in [*smoke_ids, *selection["example_ids"]]:
            self.assertEqual(dev[example_id].metadata["split"], "dev")

    def test_provider_settings_do_not_default_to_mock_or_hidden_model(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = settings_from_env()

        self.assertIn("COLLECTIVEEVAL_PROVIDER", settings.missing_requirements())
        self.assertIn("COLLECTIVEEVAL_MODEL", settings.missing_requirements())
        self.assertNotEqual(settings.provider, "mock")

    def test_strategy_configs_use_v3_1_dev_dataset_and_real_provider(self) -> None:
        settings = Phase8Settings(
            provider="openai",
            model="configured-real-model",
            api_key_env="OPENAI_API_KEY",
            base_url=None,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
            pilot_cap_usd=1.0,
            smoke_max_cost_usd=0.25,
        )
        configs = strategy_configs(
            settings,
            dataset_path=Path("reports/pilot_v1/pilot_dev32.jsonl"),
            condition="matched_tokens",
            sample_size=32,
        )

        self.assertEqual(set(configs), {
            "single_agent",
            "self_consistency_k2",
            "critic_reviser",
            "debate_2agents_1revision",
            "heuristic_adaptive_router",
        })
        for config in configs.values():
            self.assertEqual(config["benchmark_version"], BENCHMARK_V3_1_VERSION)
            self.assertEqual(config["model"]["provider"], "openai")
            self.assertNotIn("api_key", config["model"])
            self.assertEqual(config["dataset"]["path"], "reports/pilot_v1/pilot_dev32.jsonl")
            self.assertTrue(config["phase8"]["test_split_forbidden"])
            self.assertEqual(config["concurrency"]["limit"], 1)
            self.assertTrue(config["checkpoint"]["resume_completed_examples"])
        debate = configs["debate_2agents_1revision"]
        self.assertEqual(debate["strategy"]["agents"], 2)
        self.assertEqual(debate["strategy"]["rounds"], 2)
        self.assertEqual(debate["strategy"]["revision_rounds"], 1)
        router = configs["heuristic_adaptive_router"]
        self.assertEqual(router["strategy"]["call_estimate_type"], "worst_case_escalation")

    def test_ollama_settings_allow_zero_cost_and_no_api_key_for_localhost(self) -> None:
        env = {
            "COLLECTIVEEVAL_PROVIDER": "ollama",
            "COLLECTIVEEVAL_BASE_URL": "http://localhost:11434/v1",
            "COLLECTIVEEVAL_MODEL": "gemma3:4b",
            "COLLECTIVEEVAL_INPUT_COST_PER_1K": "0",
            "COLLECTIVEEVAL_OUTPUT_COST_PER_1K": "0",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = settings_from_env()

        self.assertEqual(settings.missing_requirements(), [])
        self.assertTrue(settings.local_provider)
        self.assertEqual(settings.base_url_class, "local")
        self.assertEqual(settings.concurrency_limit, 1)

    def test_ollama_model_config_persists_local_provider_metadata(self) -> None:
        settings = Phase8Settings(
            provider="ollama",
            model="gemma3:4b",
            api_key_env="OPENAI_API_KEY",
            base_url="http://localhost:11434/v1",
            input_cost_per_1k=0.0,
            output_cost_per_1k=0.0,
            pilot_cap_usd=1.0,
            smoke_max_cost_usd=0.25,
            model_identity={"name": "gemma3:4b", "digest": "sha256:localdigest"},
        )

        model = base_model_config(settings, temperature=0.1)

        self.assertEqual(model["provider"], "ollama")
        self.assertEqual(model["model"], "gemma3:4b")
        self.assertEqual(model["base_url"], "http://localhost:11434/v1")
        self.assertEqual(model["input_cost_per_1k"], 0.0)
        self.assertEqual(model["output_cost_per_1k"], 0.0)
        self.assertEqual(model["provider_options"]["base_url_class"], "local")
        self.assertEqual(model["provider_options"]["model_digest"], "sha256:localdigest")

    def test_ollama_registry_uses_openai_compatible_provider_without_real_key(self) -> None:
        config = {
            "model": {
                "provider": "ollama",
                "model": "gemma3:4b",
                "base_url": "http://localhost:11434/v1",
            }
        }
        registry = build_provider_registry(config)
        provider = registry["ollama"]

        self.assertIsInstance(provider, OpenAICompatibleProvider)
        with patch.dict(os.environ, {}, clear=True):
            headers = provider._headers(
                ModelSpec(
                    provider="ollama",
                    model="gemma3:4b",
                    base_url="http://localhost:11434/v1",
                )
            )

        self.assertEqual(headers["Authorization"], "Bearer collectiveeval-local-placeholder")

    def test_ollama_native_usage_counts_are_preserved_when_returned(self) -> None:
        request = ProviderRequest(
            example=load_jsonl(DEV_PATH)[0],
            model=ModelSpec(provider="ollama", model="gemma3:4b"),
            strategy="single_agent",
            role="generator",
            prompt="Return JSON",
            prompt_version="test.v1",
        )

        response = normalize_openai_response(
            {
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "prompt_eval_count": 21,
                "eval_count": 13,
            },
            request=request,
            latency_ms=10.0,
            provider_id="ollama",
        )

        self.assertEqual(response.usage.input_tokens, 21)
        self.assertEqual(response.usage.output_tokens, 13)
        self.assertEqual(response.usage_source, UsageSource.PROVIDER_REPORTED)


if __name__ == "__main__":
    unittest.main()
