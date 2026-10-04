# CollectiveEval

**Budget-aware evaluation of multi-agent LLM inference on Japanese enterprise tasks.**

CollectiveEval asks when orchestration improves answer quality, and whether that
improvement survives shared inference budgets. The distinguishing comparison is
**quality under matched token, cost, and latency allowances**, not simply whether
five agents beat one agent while consuming five times the resources.

The framework includes six inference strategies, frozen synthetic benchmark splits,
deterministic evaluators, provider-attempt accounting, paired bootstrap analysis,
SQLite persistence, and gated experiment execution.

## Project Status

- Benchmark v3.1 is frozen. The benchmark is **synthetic**, not private enterprise data.
- A real Ollama/Gemma 3 4B DEV pilot and Phase 8.2 corrective runs completed.
- Phase 9's held-out protocol is frozen. **No held-out TEST results exist yet.**
- The primary router is heuristic. A mock-trained learned router is not scientific evidence.

This showcase includes code, tests, DEV data, and a compact
[pilot evidence snapshot](docs/phase8_dev_results.json). Local databases,
machine-specific protocol bundles, historical raw artifacts, mock-trained router
artifacts, and **all TEST splits** are excluded. Public evidence is an aggregate
export, not the full raw reproduction bundle.

## Strategies

| Strategy | Procedure |
|---|---|
| SingleAgent | One model, one answer |
| SelfConsistency | Multiple requested generations and deterministic aggregation |
| MultiAgentDebate | Independent answers, peer arguments, and revision rounds |
| CriticReviser | Generator, specialist critic, and conditional revision |
| HeterogeneousPanel | Candidate models/roles followed by judging |
| AdaptiveRouter | Cheap initial answer with uncertainty-based conditional escalation |

The primary protocol freezes five recipes: SingleAgent, SelfConsistency K=2,
CriticReviser, two-agent debate with one revision round, and HeuristicAdaptiveRouter.
HeterogeneousPanel and learned routing are not in that primary experiment.

## Real DEV Pilot

32 frozen DEV examples: eight per task family; eight easy, eight medium, sixteen
hard. All primary recipes use local `gemma3:4b` through Ollama, concurrency one.

| Strategy | Natural Score | Matched-Token Score | Delta vs SingleAgent | Paired 95% CI |
|---|---:|---:|---:|---|
| SingleAgent | 0.582813 | 0.582813 | 0 | Reference |
| SelfConsistency K=2 | 0.587618 | 0.587618 | +0.004805 | [0, +0.013516] |
| CriticReviser | 0.612778 | 0.612778 | +0.029965 | [-0.017361, +0.098351] |
| Debate | 0.562848 | 0.562848 | -0.019965 | [-0.039063, -0.004774] |
| HeuristicAdaptiveRouter | 0.598561 | 0.598561 | +0.015748 | [-0.016805, +0.064049] |

Values are exported from completed experiment artifacts. Paired score intervals
are identical in the two corrected conditions; resource consumption is not.
Bootstrap: 1,000 resamples, seed 20261003. No automatic significance claims.

On this small DEV pilot, correcting conservative admission removed the earlier
CriticReviser/Router matched-token regressions. Debate remained worse and more
resource-intensive. This is **not a general multi-agent superiority claim**.

Important qualifications:

- Shared token allowances are not identical realized spending; the 4,000-token
  ceiling was mostly nonbinding. Natural allowance was 12,000 tokens.
- Temperatures differ by recipe, so orchestration is not causally isolated.
- Groundedness/failure annotations are deterministic heuristics, not human judgment.
- Distinct SelfConsistency seeds produced disagreement on 9/32 examples, not
  guaranteed independent samples or improved selected-output scores.
- Local API pricing is zero; compute, energy, and elapsed time are not universally free.

See [findings and methodology](docs/phase8_dev_results.md) and
[machine-readable evidence](docs/phase8_dev_results.json) for run IDs, intervals,
breakdowns, resource metrics, model identity, and provenance hashes.

## Architecture

```text
BenchmarkExample + experiment config
                  |
             StrategyContext
                  |
       BudgetLedger / admission / reservations
                  |
       Provider -> logical calls + physical attempts
                  |
       SQLite + atomic example checkpoints
                  |
       deterministic evaluation + paired reports
```

Modules are under `src/collectiveeval/`; gated research launchers are under
`scripts/`. Providers include MockProvider, OpenAI-compatible endpoints, Ollama,
vLLM, and optional Hugging Face models. FastAPI exposes experiment submission
and run inspection.

## Setup

Package installation requires **Python 3.12 or newer**, as declared in pyproject.toml.

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
```

Optional Hugging Face dependencies:

```sh
python3 -m pip install -e '.[providers]'
```

The historical local pilot used direct source execution under Python 3.11.2;
that evidence does not change the package's declared installation requirement.

## Quick Start

Run a keyless mock experiment. This is a software smoke test, not scientific evidence:

```sh
PYTHONPATH=src python3 -m collectiveeval.cli run \
  --config configs/mock_single_agent.yaml \
  --db /tmp/collectiveeval.sqlite3 \
  --output-dir /tmp/collectiveeval-runs
```

Default mock fixtures do not inspect gold answers. Gold-aware fixture mode is
restricted to deliberate regression tests.

## Local Ollama

Local-provider pilot configurations recognize:

```sh
export COLLECTIVEEVAL_PROVIDER="ollama"
export COLLECTIVEEVAL_BASE_URL="http://localhost:11434/v1"
export COLLECTIVEEVAL_MODEL="gemma3:4b"
export COLLECTIVEEVAL_INPUT_COST_PER_1K="0"
export COLLECTIVEEVAL_OUTPUT_COST_PER_1K="0"
```

Local Ollama does not require a real OpenAI API key. Provider-reported usage is
preserved; unavailable returned usage uses an explicitly labeled estimation
fallback. A sent failure without usage is different: consumption is unknown,
not estimated as zero.

Historical Phase 8/9 launchers need original local evidence and the frozen protocol
bundle. They are not a one-command public-clone reproduction workflow.
No README command launches the held-out experiment.

## Scientific Accounting

- `logical_model_calls` counts strategy generations.
- `provider_attempts` counts dispatched invocations; `max_calls` constrains attempts.
- General-library retries retain separate rows and undergo budget re-admission.
- Unknown scientific consumption persists NULL tokens with `UNKNOWN_NOT_RETURNED`
  and stops under `UNKNOWN_USAGE_STOP`, without automatic retry or server restart.
- Returned usage survives parsing failures and post-call budget overruns.
- Completed overruns remain accounted; subsequent calls are blocked.
- Attempt, logical lifecycle, and strategy wall-clock latency are separate.
- Completed examples resume without replay; unknown/in-flight work requires review.

The clean corrective runs recorded **405 logical calls, 405 attempts, 261,437
provider-reported tokens, and zero retries or unknown-usage events**.

## Tests

The public DEV-safe suite excludes four structural suites requiring withheld TEST:

```sh
python3 -m pytest -q \
  --ignore=tests/test_benchmark_v2.py \
  --ignore=tests/test_benchmark_v3.py \
  --ignore=tests/test_benchmark_v3_1.py \
  --ignore=tests/test_phase6_benchmark.py
python3 -m ruff check .
python3 -m mypy src
```

Final local verification at protocol freeze: **146 passed, one FastAPI availability
skip**, Ruff clean, mypy clean across 35 source files. Full benchmark structural
checks need the private held-out bundle; missing TEST files are intentional.

A clean copy of the public upload was separately verified: **144 passed, three
skipped**, Ruff clean and mypy clean. The skips are FastAPI availability and two
checks that require withheld historical raw evidence. The router scope warning
is included; its trained model and training metrics are not.

## Reproducibility and Scope

DEV bytes and manifests retain frozen identities. The public export binds its local
source analysis by SHA-256 and includes model/run/protocol identities. Real API
replay is not guaranteed deterministic, even with fixed requested seeds.
Reused historical runs lack an independent provider-side completeness audit;
clean corrective runs have explicit attempt accounting.

Publication does not authorize TEST access, training on TEST, post-TEST tuning,
or changes to the frozen local protocol. Historical raw evidence and the blocked
run remain preserved outside the public Git history.
