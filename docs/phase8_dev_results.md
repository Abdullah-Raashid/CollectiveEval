# Phase 8.2 DEV Evidence

This is a public aggregate showcase of completed local experiments, not a held-out
result or a replacement for the private raw evidence bundle. Numerical evidence,
run IDs, resource metrics, subgroup breakdowns, paired intervals and source hashes
are exported in [phase8_dev_results.json](phase8_dev_results.json).

## Scope

Benchmark: collectiveeval-benchmark-v3.1, synthetic Japanese enterprise tasks.
Pilot: 32 fixed DEV examples, eight each for grounded QA, structured extraction,
business summarization and robustness. Difficulty: 8 easy, 8 medium, 16 hard.
Provider: local Ollama; model: gemma3:4b, Q4_K_M; concurrency one.

The corrected view contains five approved historical observations and five new
corrective runs. Diagnostics are excluded. No TEST example or learned-router
result is included. Historical request completeness is independently unverifiable;
clean corrective efficiency evidence does not retroactively certify old calls.

## Methodology Repairs

The original admission estimate counted metadata that was not sent to the model,
causing some orchestration to stop early. Corrected admission uses the actual
provider-facing prompt, clips output allowance centrally, retains authoritative
returned usage, and distinguishes pre-call rejection from actual post-call overrun.

Retries previously collapsed failed attempts into the eventual success. New DBs
separate logical model calls and physical attempts. Every attempt consumes a call
slot. Unknown consumption is NULL, not zero, and stops scientific execution without
automatic retry. A client timeout does not imply server cancellation. Parse failure
after returned inference retains usage. Checkpoints preserve completed examples and
refuse silent replay of sent unknown/incomplete trajectories.

Five clean corrective runs: 405 logical calls, 405 attempts, 261437 reported tokens,
zero retries, failed attempts or unknown consumption. Two pre-call rejections and
one actual 55-token overrun are visible and retained.

## Findings

On this 32-example DEV pilot with local Gemma 3 4B, the CriticReviser/Router
matched-token regressions disappeared after admission repair. Their corrected
mean scores match their natural-condition scores. Debate remains lower quality
and more expensive than SingleAgent. Positive critique/router point estimates
have broad paired intervals; no automatic significance claims are made.

SelfConsistency requests distinct deterministic seeds but candidates are identical
on 23/32 examples and disagree on 9/32. The natural selected outputs did not change
relative to the same-seed historical run. Diversity does not guarantee a score gain.

Router accepted 20 initial answers and escalated 12 (11 critic, one debate). Its
matched mean is 2.0625 attempts/example, not five unconditional calls. All six
historically truncated router cases completed critique; five completed revision.
The sixth still has a legitimate pre-call token rejection.

## Limits

- The shared 4000-token ceiling was mostly nonbinding; it is not equal realized spending.
- Sampling temperatures differ across recipes, limiting causal orchestration claims.
- This is a small synthetic DEV sample on one quantized model, not an enterprise trial.
- Tiny reasoning-family subgroups and heuristic groundedness/failure labels need caution.
- Five reused historical runs do not have an independent physical-request audit.
- Raw databases and all TEST splits are intentionally withheld from the public repository.

## Held-out Status

Phase 9's local protocol is frozen, with five recipes across natural and matched-token
conditions, and no ablations, second model or learned router. TEST remains unexecuted.
The machine-specific protocol/raw evidence bundle is not included in this showcase.

Protocol SHA-256:
`e8fc45a7e772d24e6b8e422a1b51e09af07b74b58b950c06c08fb16e61571bce`.
