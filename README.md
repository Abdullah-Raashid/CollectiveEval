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
- Frozen held-out TEST execution and post-run integrity audit **completed**:
  10 runs, 2,000 predictions, 4,510 calls/attempts, 3,012,623 reported tokens.
- The primary router is heuristic. A mock-trained learned router is not scientific evidence.

This showcase includes code, tests, DEV data, and aggregate
[held-out results](docs/heldout_results.json), alongside the historical
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

## Held-Out Results

200 frozen synthetic Japanese TEST examples: 50 per task family; 60 easy,
60 medium, 80 hard. Local Ollama `gemma3:4b` (Q4_K_M), concurrency one.
Natural allowance: 12,000 tokens/example; matched allowance: 4,000.
Each recipe has n=200, including invalid outputs. These are composite task
scores, not a single accuracy percentage.

| Recipe | Natural Mean | Matched Mean | Natural Delta vs SA [Paired 95% CI] | Matched Delta vs SA [Paired 95% CI] |
|---|---:|---:|---|---|
| SingleAgent | 0.6345 | 0.6345 | Reference | Reference |
| SelfConsistency K=2 | 0.6381 | 0.6381 | +0.0036 [-0.0013, +0.0091] | +0.0036 [-0.0013, +0.0091] |
| CriticReviser | 0.6271 | 0.6271 | -0.0074 [-0.0220, +0.0054] | -0.0074 [-0.0220, +0.0054] |
| Debate | 0.6164 | 0.6164 | -0.0181 [-0.0344, -0.0025] | -0.0181 [-0.0344, -0.0025] |
| HeuristicAdaptiveRouter | 0.6414 | 0.6334 | +0.0069 [-0.0108, +0.0263] | -0.0011 [-0.0204, +0.0190] |

Frozen bootstrap: 1,000 resamples, seed 20261003, 95% intervals. No automatic
significance claims. Full score intervals and subgroup tables are in the
[detailed report](docs/heldout_results.md).

| Recipe | Natural Calls / Tokens | Matched Calls / Tokens | Natural / Matched Attempt Seconds |
|---|---:|---:|---:|
| SingleAgent | 1.00 / 564 | 1.00 / 564 | 6.82 / 7.28 |
| SelfConsistency | 2.00 / 1,128 | 2.00 / 1,128 | 12.28 / 13.10 |
| CriticReviser | 2.58 / 1,904 | 2.57 / 1,894 | 21.20 / 25.83 |
| Debate | 3.99 / 2,801 | 3.94 / 2,720 | 35.27 / 49.79 |
| Router | 1.75 / 1,198 | 1.73 / 1,162 | 14.53 / 21.11 |

Values are means per example. Calls equal provider attempts here: zero retries.
Sequential laptop timing is confounded by run order, thermal state and load.

**Central findings:**

- Debate lost quality at approximately five times SA's natural tokens;
  CriticReviser's +0.030 DEV gain reversed to -0.0074 on TEST.
- Router accepted 147/200 initial answers, escalating 50 to critique and three
  to debate. It used 56% fewer natural calls than Debate, but more than SA.
  Its natural overall gain is uncertain; robustness gained +0.0485 (n=50),
  while extraction lost quality. These subgroup results are descriptive.
- SelfConsistency had 55/200 diverse candidate pairs, but selected sample zero
  every time. Its small gain versus SA is not evidence of aggregation helping;
  frozen temperatures differ (SC 0.6 versus SA 0.2).
- Six shortened matched Router revisions ended at their output caps and failed
  JSON parsing (-0.008875 full-cohort contribution); one valid summary improved
  (+0.000875), for a net -0.0080 versus natural. No outputs were repaired,
  retried or excluded.

**Budget caveat:** shared allowances are not equal realized spending or strict
hard caps. The matched condition retained 15 pre-call rejections and 19 overruns
(CR 2, Debate 7, Router 10); maximum actual usage reached 4,455 tokens.
On point-estimate token/latency Pareto fronts, natural SA/SC/Router are
nondominated; matched SA/SC are nondominated. This is not a universal ranking.

**Limits:** one quantized model, synthetic data, heuristic groundedness/failure
labels, differing recipe temperatures, unadjusted subgroup comparisons, and
private raw evidence. Local marginal API cost is zero; compute, energy and
elapsed time are not universally free. Learned routing and heterogeneous panels
were not part of this held-out experiment.

The [corrected DEV pilot](docs/phase8_dev_results.md) had only 32 examples.
Debate's negative direction replicated; CR reversed; Router's natural gain
weakened and matched gain reversed. DEV gain intervals included zero.

Protocol v3 SHA-256:
`0df6313de34514d713f6bc91f372ab12f4072849299b23ae76c5bdd372babf1a`.
Model digest:
`a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a`.
The canonical view preserves v1/v2 completions, the original schema failure,
and v3 continuation checkpoints. All cohort, identity, artifact and accounting
checks pass without replay. [Aggregate JSON](docs/heldout_results.json) binds
run IDs, model/protocol/benchmark hashes and offline analysis source hashes.

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

The historical local experiments used direct source execution under Python 3.11.2;
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

Historical research launchers need original local evidence and the frozen protocol
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

The final held-out canonical experiment recorded **4,510 logical calls/attempts,
3,012,623 reported tokens, zero transport failures/retries/unknown usage**, and
**eight returned parse/schema failures**. The frozen `failed_attempts` metric
counts transport failures; it does not imply all returned outputs were valid.

## Tests

The public DEV-safe suite excludes four structural suites requiring withheld TEST:

```sh
python3 -m pytest -q \
  --ignore=tests/test_benchmark_v2.py \
  --ignore=tests/test_benchmark_v3.py \
  --ignore=tests/test_benchmark_v3_1.py \
  --ignore=tests/test_phase6_benchmark.py
python3 -m ruff check .
python3 -m mypy src analysis
```

Final local DEV-safe verification: **254 passed, one FastAPI availability skip**;
the nested held-out reporting suite contributes **21 passing tests**. Ruff is
clean and mypy is clean across 41 source/analysis files. Full benchmark structural
checks need the private held-out bundle; missing TEST files are intentional.
A clean public-only copy passes **251 tests, four skips** (FastAPI availability
and three checks requiring withheld historical evidence), with Ruff/mypy clean.

Read-only local reproduction, requiring the preserved private bundle:

```sh
PYTHONPATH=src python3 -m analysis.heldout_final --check
```

This verifies evidence/protocol immutability and byte-identical analysis outputs;
it never calls a provider. Analysis code and reporting tests live outside the
frozen execution-source globs so the protocol's code identity remains intact.

## Reproducibility and Scope

DEV bytes and manifests retain frozen identities. The public export binds its local
source analysis by SHA-256 and includes model/run/protocol identities. Real API
replay is not guaranteed deterministic, even with fixed requested seeds.
Reused historical runs lack an independent provider-side completeness audit;
clean corrective runs have explicit attempt accounting.

TEST execution is finished. This publication pass is authorized read-only analysis,
not permission for more inference, training on TEST, post-TEST tuning, or changes
to the frozen local protocol. Historical raw evidence and blocked runs remain
preserved outside the public Git history.
