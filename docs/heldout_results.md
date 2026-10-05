# CollectiveEval: Frozen Held-Out Results

The experiment is complete. This is read-only post-run analysis, with no additional inference, TEST tuning, prediction changes or exclusions.

## Setup And Integrity

Integrity PASSED: ten canonical runs, exactly 200 unique TEST IDs per recipe, 2,000 predictions. Benchmark v3.1 is synthetic; 50 examples per task family, 60 easy / 60 medium / 80 hard. Local Ollama Gemma 3 4B (Q4_K_M), concurrency 1, no retries, 700 configured output tokens/call, up to 6 attempts/example. Requested temperatures: SA 0.2, SC 0.6, CR 0.2, Debate 0.5, Router 0.2 (nested debate retains 0.5); top_p=1; seed=20261003. Natural 12,000 and matched 4,000 total token allowances. No learned-router or heterogeneous-panel results.

Protocol v3 SHA-256: `0df6313de34514d713f6bc91f372ab12f4072849299b23ae76c5bdd372babf1a`.
Model digest: `a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a`.
TEST SHA-256: `0372cdb6389205a71db269327af5cbed6679ae2860235b806fbdcd1e3426fb7f`.
DEV SHA-256: `4c2779b9daae1a6e5c6d22677a9825f6ea7b7d6e00ac5b360e87e02e6547d622`.

Frozen percentile/paired bootstrap: 1,000 resamples, seed 20261003, 95% intervals. Primary n always 200, including invalid outputs. No automatic significance claims; subgroup intervals are descriptive and not multiplicity-adjusted. Scores are heterogeneous deterministic task metrics, not a single accuracy percentage.

### Canonical Provenance

| Condition | Recipe | Run ID | Continuation provenance |
| --- | --- | --- | --- |
| natural | SA | 5936fb81-ef1f-47ce-834f-7dab65771efc | v1_completed_reused |
| natural | SC | a3f7ad78-86be-4845-bc54-916bc93a0924 | v2_completed_carried |
| natural | CR | 7f4de625-d847-4ec8-ba2a-471af9e11728 | v2_completed_carried |
| natural | Debate | 774ba57e-0e09-403a-add9-59dd81d9c21c | v2_31_complete_plus_1_failure_and_v3_168_new |
| natural | Router | a15f0ac1-74e8-4e06-bf94-318b5b843dcb | v3_new |
| matched_tokens | SA | 1d48943f-cff4-402b-aea5-e8446becca10 | v3_new |
| matched_tokens | SC | 494eacbd-6212-4907-9df5-1e20496c267c | v3_new |
| matched_tokens | CR | 67d46d5e-dda5-4295-9fad-3a042eac625c | v3_new |
| matched_tokens | Debate | 44031ea9-0182-4e29-9e15-f58765206fa0 | v3_new |
| matched_tokens | Router | f8545d1e-e7a3-4483-9315-5ee765352c20 | v3_new |

V1 SA is carried read-only; v2 SC/CR checkpoints are retained; natural Debate is 31 completed v2 examples + one preserved returned failure + 168 v3 examples. Six other recipes are v3 executions. Certificates, model/config identities, artifact hashes and stored accounting reconcile; v1/v2/v3 evidence remains unchanged.

| Accounting measure | Canonical total |
| --- | --- |
| predictions | 2000 |
| logical_model_calls | 4510 |
| provider_attempts | 4510 |
| input_tokens | 2444384 |
| output_tokens | 568239 |
| total_tokens | 3012623 |
| failed_attempts | 0 |
| output_failure_attempts | 8 |
| retries | 0 |
| unknown_usage_attempts | 0 |
| parse_failures | 8 |
| pre_call_budget_rejections | 15 |
| post_call_overruns | 19 |
| attempt_latency_ms | 41442565.33776994 |
| logical_lifecycle_latency_ms | 41507741.27836013 |
| strategy_wall_clock_latency_ms | 41783152.4695692 |
| strategy_wall_clock_reconstructed_examples | 1 |
| marginal_api_cost_usd | 0.0 |

## Natural Results

| Recipe | n | Mean score | Score 95% CI | Delta vs SA | Paired 95% CI |
| --- | --- | --- | --- | --- | --- |
| SingleAgent | 200 | 0.6345 | [+0.5875, +0.6898] | +0.0000 | [+0.0000, +0.0000] |
| SelfConsistency K=2 | 200 | 0.6381 | [+0.5919, +0.6925] | +0.0036 | [-0.0013, +0.0091] |
| CriticReviser | 200 | 0.6271 | [+0.5812, +0.6829] | -0.0074 | [-0.0220, +0.0054] |
| Debate (2 agents, 1 revision) | 200 | 0.6164 | [+0.5700, +0.6677] | -0.0181 | [-0.0344, -0.0025] |
| HeuristicAdaptiveRouter | 200 | 0.6414 | [+0.5944, +0.6938] | +0.0069 | [-0.0108, +0.0263] |

| Recipe | Calls mean/median | Attempts mean/median | Tokens mean/median | Attempt s mean/median | Lifecycle s mean/median | Wall s mean/median | Quality/1k tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SA | 1.00 / 1.00 | 1.00 / 1.00 | 563.60 / 510.00 | 6.82 / 5.16 | 6.83 / 5.17 | 6.85 / 5.18 | 1.1258 |
| SC | 2.00 / 2.00 | 2.00 / 2.00 | 1128.40 / 1022.00 | 12.28 / 8.46 | 12.30 / 8.47 | 12.33 / 8.53 | 0.5655 |
| CR | 2.58 / 3.00 | 2.58 / 3.00 | 1903.50 / 1737.50 | 21.20 / 16.84 | 21.23 / 16.87 | 21.31 / 16.96 | 0.3295 |
| Debate | 3.99 / 4.00 | 3.99 / 4.00 | 2801.45 / 2518.00 | 35.27 / 28.56 | 35.32 / 28.59 | 35.43 / 28.66 | 0.2200 |
| Router | 1.75 / 1.00 | 1.75 / 1.00 | 1198.07 / 542.50 | 14.53 / 6.93 | 14.55 / 6.94 | 14.64 / 7.02 | 0.5354 |

## Matched-Token Results

| Recipe | n | Mean score | Score 95% CI | Delta vs SA | Paired 95% CI |
| --- | --- | --- | --- | --- | --- |
| SingleAgent | 200 | 0.6345 | [+0.5875, +0.6898] | +0.0000 | [+0.0000, +0.0000] |
| SelfConsistency K=2 | 200 | 0.6381 | [+0.5919, +0.6925] | +0.0036 | [-0.0013, +0.0091] |
| CriticReviser | 200 | 0.6271 | [+0.5812, +0.6829] | -0.0074 | [-0.0220, +0.0054] |
| Debate (2 agents, 1 revision) | 200 | 0.6164 | [+0.5700, +0.6677] | -0.0181 | [-0.0344, -0.0025] |
| HeuristicAdaptiveRouter | 200 | 0.6334 | [+0.5851, +0.6864] | -0.0011 | [-0.0204, +0.0190] |

| Recipe | Calls mean/median | Attempts mean/median | Tokens mean/median | Attempt s mean/median | Lifecycle s mean/median | Wall s mean/median | Quality/1k tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SA | 1.00 / 1.00 | 1.00 / 1.00 | 563.60 / 510.00 | 7.28 / 6.21 | 7.29 / 6.22 | 7.36 / 6.29 | 1.1258 |
| SC | 2.00 / 2.00 | 2.00 / 2.00 | 1128.40 / 1022.00 | 13.10 / 11.00 | 13.12 / 11.02 | 13.24 / 11.13 | 0.5655 |
| CR | 2.57 / 3.00 | 2.57 / 3.00 | 1893.93 / 1737.50 | 25.83 / 19.56 | 25.87 / 19.61 | 26.08 / 19.88 | 0.3311 |
| Debate | 3.94 / 4.00 | 3.94 / 4.00 | 2719.97 / 2518.00 | 49.79 / 41.42 | 49.89 / 41.51 | 50.24 / 41.96 | 0.2266 |
| Router | 1.73 / 1.00 | 1.73 / 1.00 | 1162.18 / 542.50 | 21.11 / 10.77 | 21.15 / 10.78 | 21.44 / 10.90 | 0.5450 |

Resource cells show mean / median per example. Attempt latency is the sum of recorded provider durations; lifecycle includes client work; wall latency includes orchestration. One historical failed wall duration is reconstructed. Quality/1k is a ratio of means, not a recomputation of the frozen task metric. Local marginal API cost is zero; quality/$ is not meaningful and is omitted.

## Matched Minus Natural

| Recipe | Score delta [paired CI] | Tokens/example delta | Calls/example delta | Attempt s/example delta | Rejection event delta | Overrun delta |
| --- | --- | --- | --- | --- | --- | --- |
| SA | +0.0000 [+0.0000, +0.0000] | +0.00 | +0.000 | +0.46 | 0 | 0 |
| SC | +0.0000 [+0.0000, +0.0000] | +0.00 | +0.000 | +0.82 | 0 | 0 |
| CR | +0.0000 [+0.0000, +0.0000] | -9.56 | -0.005 | +4.63 | 1 | 2 |
| Debate | +0.0000 [+0.0000, +0.0000] | -81.48 | -0.055 | +14.52 | 11 | 7 |
| Router | -0.0080 [-0.0165, -0.0015] | -35.88 | -0.020 | +6.58 | 3 | 10 |

The first four recipes have identical primary scores across conditions; resources need not be identical. Router loses 0.0080 aggregate score, entirely in business summarization (family delta -0.0320). Six matched Router revision responses terminate at their shortened output caps with finish_reason=length and fail JSON parsing. They remain in the denominator with the frozen failed-output score semantics, not repaired initial-answer fallbacks.

The six failures contribute -0.008875 to the full-cohort mean difference. One changed valid summary contributes +0.000875; the other 193 scores are unchanged. Net difference is -0.0080, not a loss attributable only to the six failures with no other output changes.

Shared ceilings are not identical realized spending or hard caps. The 4,000-token condition has 15 pre-call rejection events and 19 retained overrun examples: CR 2, Debate 7, Router 10. Maximum actual tokens are 4,455 / 4,173 / 4,417 respectively. Therefore this study supports a comparison under shared admission allowances, not a claim of perfectly enforced strict token parity.

Sequential laptop latency is confounded by run order, thermal state and load: even SA and SC use identical tokens but take longer in the matched condition. Do not attribute all latency differences to the budget policy.

### Central Question

Under these frozen 4,000-token allowances, no orchestration recipe has a higher overall mean than SC; CR and Debate are below SA, and Router is slightly below SA. Under natural allowances, Router has the highest overall point estimate at extra compute, but its paired interval versus SA spans zero. Router's robustness-family gain is +0.0485 (n=50) in both conditions, while extraction worsens. Debate's extraction delta is -0.0703; CR's QA delta is -0.0231. These are descriptive, unadjusted subgroup findings, not a general superiority claim or proof that orchestration caused the differences.


## Task Families

### natural

| Group | n | SA mean | SC mean; delta [CI] | CR mean; delta [CI] | Debate mean; delta [CI] | Router mean; delta [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| business_summarization | 50 | 0.2909 | 0.3019; +0.0110 [-0.0050, +0.0290] | 0.2909; +0.0000 [+0.0000, +0.0000] | 0.2999; +0.0090 [-0.0100, +0.0315] | 0.2914; +0.0005 [-0.0030, +0.0045] |
| grounded_qa | 50 | 0.7516 | 0.7512; -0.0004 [-0.0012, +0.0000] | 0.7285; -0.0231 [-0.0698, +0.0065] | 0.7316; -0.0200 [-0.0600, +0.0000] | 0.7447; -0.0069 [-0.0604, +0.0396] |
| robustness | 50 | 0.5172 | 0.5241; +0.0069 [+0.0000, +0.0190] | 0.5247; +0.0075 [-0.0171, +0.0396] | 0.5260; +0.0088 [-0.0105, +0.0417] | 0.5657; +0.0485 [+0.0067, +0.1032] |
| structured_extraction | 50 | 0.9783 | 0.9753; -0.0031 [-0.0092, +0.0000] | 0.9644; -0.0139 [-0.0294, -0.0031] | 0.9081; -0.0703 [-0.0961, -0.0458] | 0.9639; -0.0144 [-0.0303, -0.0031] |

### matched_tokens

| Group | n | SA mean | SC mean; delta [CI] | CR mean; delta [CI] | Debate mean; delta [CI] | Router mean; delta [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| business_summarization | 50 | 0.2909 | 0.3019; +0.0110 [-0.0050, +0.0290] | 0.2909; +0.0000 [+0.0000, +0.0000] | 0.2999; +0.0090 [-0.0100, +0.0315] | 0.2594; -0.0315 [-0.0650, -0.0055] |
| grounded_qa | 50 | 0.7516 | 0.7512; -0.0004 [-0.0012, +0.0000] | 0.7285; -0.0231 [-0.0698, +0.0065] | 0.7316; -0.0200 [-0.0600, +0.0000] | 0.7447; -0.0069 [-0.0604, +0.0396] |
| robustness | 50 | 0.5172 | 0.5241; +0.0069 [+0.0000, +0.0190] | 0.5247; +0.0075 [-0.0171, +0.0396] | 0.5260; +0.0088 [-0.0105, +0.0417] | 0.5657; +0.0485 [+0.0067, +0.1032] |
| structured_extraction | 50 | 0.9783 | 0.9753; -0.0031 [-0.0092, +0.0000] | 0.9644; -0.0139 [-0.0294, -0.0031] | 0.9081; -0.0703 [-0.0961, -0.0458] | 0.9639; -0.0144 [-0.0303, -0.0031] |


## Difficulty

### natural

| Group | n | SA mean | SC mean; delta [CI] | CR mean; delta [CI] | Debate mean; delta [CI] | Router mean; delta [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 60 | 0.7080 | 0.7099; +0.0020 [-0.0094, +0.0174] | 0.7032; -0.0047 [-0.0142, +0.0000] | 0.6775; -0.0305 [-0.0508, -0.0136] | 0.7104; +0.0025 [-0.0052, +0.0100] |
| hard | 80 | 0.6283 | 0.6308; +0.0026 [-0.0016, +0.0082] | 0.6233; -0.0050 [-0.0144, +0.0026] | 0.6145; -0.0138 [-0.0481, +0.0149] | 0.6443; +0.0161 [-0.0299, +0.0582] |
| medium | 60 | 0.5693 | 0.5760; +0.0067 [-0.0008, +0.0192] | 0.5561; -0.0132 [-0.0575, +0.0255] | 0.5578; -0.0115 [-0.0274, +0.0016] | 0.5684; -0.0009 [-0.0130, +0.0112] |

### matched_tokens

| Group | n | SA mean | SC mean; delta [CI] | CR mean; delta [CI] | Debate mean; delta [CI] | Router mean; delta [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 60 | 0.7080 | 0.7099; +0.0020 [-0.0094, +0.0174] | 0.7032; -0.0047 [-0.0142, +0.0000] | 0.6775; -0.0305 [-0.0508, -0.0136] | 0.7104; +0.0025 [-0.0052, +0.0100] |
| hard | 80 | 0.6283 | 0.6308; +0.0026 [-0.0016, +0.0082] | 0.6233; -0.0050 [-0.0144, +0.0026] | 0.6145; -0.0138 [-0.0481, +0.0149] | 0.6243; -0.0039 [-0.0527, +0.0460] |
| medium | 60 | 0.5693 | 0.5760; +0.0067 [-0.0008, +0.0192] | 0.5561; -0.0132 [-0.0575, +0.0255] | 0.5578; -0.0115 [-0.0274, +0.0016] | 0.5684; -0.0009 [-0.0130, +0.0112] |


## Frozen Reasoning Families

Reasoning labels come from unchanged v3.1 metadata. TINY cells (n<5) cannot support reliable interpretation; referential ambiguity has only n=1. Pooled family means can mix task metrics and composition effects; no post-hoc hard-case superiority claim.

### natural

| Group | n | SA mean | SC mean; delta [CI] | CR mean; delta [CI] | Debate mean; delta [CI] | Router mean; delta [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| conflict_current_version | 20 | 0.6146 | 0.6183; +0.0038 [+0.0000, +0.0113] | 0.6104; -0.0042 [-0.0125, +0.0000] | 0.6171; +0.0025 [+0.0000, +0.0075] | 0.6482; +0.0337 [+0.0000, +0.1010] |
| cross_sentence_composition | 6 | 0.2682 | 0.2648; -0.0034 [-0.0101, +0.0000] | 0.3060; +0.0378 [+0.0101, +0.0694] | 0.2675; -0.0007 [-0.0022, +0.0000] | 0.2790; +0.0108 [-0.0101, +0.0426] |
| direct_lookup | 60 | 0.7080 | 0.7099; +0.0020 [-0.0094, +0.0174] | 0.7032; -0.0047 [-0.0142, +0.0000] | 0.6775; -0.0305 [-0.0508, -0.0136] | 0.7104; +0.0025 [-0.0052, +0.0100] |
| exception_rule | 20 | 0.7223 | 0.7223; +0.0000 [+0.0000, +0.0000] | 0.7029; -0.0194 [-0.0507, +0.0000] | 0.7342; +0.0119 [-0.0458, +0.0968] | 0.8074; +0.0850 [-0.0194, +0.2089] |
| insufficient_evidence | 10 | 0.7000 | 0.7000; +0.0000 [+0.0000, +0.0000] | 0.7000; +0.0000 [+0.0000, +0.0000] | 0.6000; -0.1000 [-0.3000, +0.0000] | 0.6000; -0.1000 [-0.3000, +0.0000] |
| long_context_same_topic | 16 | 0.5112 | 0.5206; +0.0094 [-0.0078, +0.0344] | 0.5017; -0.0095 [-0.0286, +0.0000] | 0.5060; -0.0052 [-0.0571, +0.0575] | 0.5017; -0.0095 [-0.0286, +0.0000] |
| nullable_missing_field | 20 | 0.3071 | 0.3071; +0.0000 [+0.0000, +0.0000] | 0.3401; +0.0330 [+0.0000, +0.0990] | 0.3071; +0.0000 [+0.0000, +0.0000] | 0.3058; -0.0014 [-0.0042, +0.0000] |
| numeric_normalization | 7 | 0.9246 | 0.9246; +0.0000 [+0.0000, +0.0000] | 0.9246; +0.0000 [+0.0000, +0.0000] | 0.8810; -0.0437 [-0.1091, +0.0000] | 0.9246; +0.0000 [+0.0000, +0.0000] |
| paraphrase_normalization | 10 | 0.7607 | 0.7857; +0.0250 [+0.0000, +0.0750] | 0.7607; +0.0000 [+0.0000, +0.0000] | 0.7607; +0.0000 [+0.0000, +0.0000] | 0.7857; +0.0250 [+0.0000, +0.0750] |
| referential_ambiguity | 1 (TINY) | 0.2619 | 0.2619; +0.0000 [+0.0000, +0.0000] | 0.2619; +0.0000 [+0.0000, +0.0000] | 0.2619; +0.0000 [+0.0000, +0.0000] | 0.2619; +0.0000 [+0.0000, +0.0000] |
| same_entity_distractor | 30 | 0.6803 | 0.6853; +0.0050 [-0.0025, +0.0175] | 0.6319; -0.0484 [-0.1333, +0.0000] | 0.6573; -0.0230 [-0.0515, +0.0014] | 0.6711; -0.0092 [-0.0269, +0.0000] |

### matched_tokens

| Group | n | SA mean | SC mean; delta [CI] | CR mean; delta [CI] | Debate mean; delta [CI] | Router mean; delta [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| conflict_current_version | 20 | 0.6146 | 0.6183; +0.0038 [+0.0000, +0.0113] | 0.6104; -0.0042 [-0.0125, +0.0000] | 0.6171; +0.0025 [+0.0000, +0.0075] | 0.6482; +0.0337 [+0.0000, +0.1010] |
| cross_sentence_composition | 6 | 0.2682 | 0.2648; -0.0034 [-0.0101, +0.0000] | 0.3060; +0.0378 [+0.0101, +0.0694] | 0.2675; -0.0007 [-0.0022, +0.0000] | 0.2790; +0.0108 [-0.0101, +0.0426] |
| direct_lookup | 60 | 0.7080 | 0.7099; +0.0020 [-0.0094, +0.0174] | 0.7032; -0.0047 [-0.0142, +0.0000] | 0.6775; -0.0305 [-0.0508, -0.0136] | 0.7104; +0.0025 [-0.0052, +0.0100] |
| exception_rule | 20 | 0.7223 | 0.7223; +0.0000 [+0.0000, +0.0000] | 0.7029; -0.0194 [-0.0507, +0.0000] | 0.7342; +0.0119 [-0.0458, +0.0968] | 0.8074; +0.0850 [-0.0194, +0.2089] |
| insufficient_evidence | 10 | 0.7000 | 0.7000; +0.0000 [+0.0000, +0.0000] | 0.7000; +0.0000 [+0.0000, +0.0000] | 0.6000; -0.1000 [-0.3000, +0.0000] | 0.6000; -0.1000 [-0.3000, +0.0000] |
| long_context_same_topic | 16 | 0.5112 | 0.5206; +0.0094 [-0.0078, +0.0344] | 0.5017; -0.0095 [-0.0286, +0.0000] | 0.5060; -0.0052 [-0.0571, +0.0575] | 0.4017; -0.1095 [-0.1925, -0.0316] |
| nullable_missing_field | 20 | 0.3071 | 0.3071; +0.0000 [+0.0000, +0.0000] | 0.3401; +0.0330 [+0.0000, +0.0990] | 0.3071; +0.0000 [+0.0000, +0.0000] | 0.3058; -0.0014 [-0.0042, +0.0000] |
| numeric_normalization | 7 | 0.9246 | 0.9246; +0.0000 [+0.0000, +0.0000] | 0.9246; +0.0000 [+0.0000, +0.0000] | 0.8810; -0.0437 [-0.1091, +0.0000] | 0.9246; +0.0000 [+0.0000, +0.0000] |
| paraphrase_normalization | 10 | 0.7607 | 0.7857; +0.0250 [+0.0000, +0.0750] | 0.7607; +0.0000 [+0.0000, +0.0000] | 0.7607; +0.0000 [+0.0000, +0.0000] | 0.7857; +0.0250 [+0.0000, +0.0750] |
| referential_ambiguity | 1 (TINY) | 0.2619 | 0.2619; +0.0000 [+0.0000, +0.0000] | 0.2619; +0.0000 [+0.0000, +0.0000] | 0.2619; +0.0000 [+0.0000, +0.0000] | 0.2619; +0.0000 [+0.0000, +0.0000] |
| same_entity_distractor | 30 | 0.6803 | 0.6853; +0.0050 [-0.0025, +0.0175] | 0.6319; -0.0484 [-0.1333, +0.0000] | 0.6573; -0.0230 [-0.0515, +0.0014] | 0.6711; -0.0092 [-0.0269, +0.0000] |


# Mechanism Analysis

Initial/final comparisons use the unchanged evaluator offline. No gold-derived value enters routing inference. Coverage is explicit; missing candidate histories are not fabricated.

## natural

### CriticReviser

Revisions: 115/200 (57.5%); explicit no-revision: 85/200 (42.5%); budget-stopped examples: 0; returned output failures: 0.
Revised mean 0.4725; unrevised mean 0.8363. These selected groups are not randomized. Against the generator: 4 improved / 8 regressed / 188 unchanged; coverage 200.

### SelfConsistency

Two candidates per example: {2: 200}. Canonical diversity 55/200 (27.5%); full-object diversity 55/200. There are 55 K=2 ties and 145 agreement cases. Frozen tie resolution selects sample zero on all 200 examples: aggregation helps 0 / hurts 0 relative to that first sample. The small gain versus SA cannot be attributed to aggregation; recipe temperatures differ. Requested distinct seeds do not ensure independent draws.

### Debate

Completed-round distribution: {2: 199, 0: 1} (round means two successful responses; initial plus one revision). Full trajectories 199/200; initial disagreement 44; revision disagreement 30; selected structured prediction changes 109/199.
Against initial majority: 2 improve / 20 regress / 177 unchanged, coverage 199. Budget-stopped examples 0; schema failures 1. Stored majority-wrong labels: 10; peer-error-propagation heuristic labels: 64. The latter is convergence on a wrong answer, not causal evidence of influence.

### Router

Accept 147/200 (73.5%); escalate 53/200 (26.5%): {'accept': 147, 'critic': 50, 'debate': 3}. Reconstructed routes 0; none remain unknown.
| Route group | n | Mean score | Mean calls | Mean tokens |
| --- | --- | --- | --- | --- |
| accepted | 147 | 0.6624 | 1.000 | 531.28 |
| escalated | 53 | 0.5832 | 3.830 | 3047.47 |
Actual calls/example distribution: {1.0: 147, 3.0: 12, 4.0: 38, 5.0: 3}. Initial-to-final: 2 improved / 5 regressed / 193 unchanged. 91 accepted predictions are imperfect; 51/53 escalations do not improve the initial score. These retrospective heuristics do not establish whether an unexecuted alternative route would have helped.

## matched_tokens

### CriticReviser

Revisions: 114/200 (57.0%); explicit no-revision: 85/200 (42.5%); budget-stopped examples: 1; returned output failures: 0.
Revised mean 0.4740; unrevised mean 0.8301. These selected groups are not randomized. Against the generator: 4 improved / 8 regressed / 188 unchanged; coverage 200.

### SelfConsistency

Two candidates per example: {2: 200}. Canonical diversity 55/200 (27.5%); full-object diversity 55/200. There are 55 K=2 ties and 145 agreement cases. Frozen tie resolution selects sample zero on all 200 examples: aggregation helps 0 / hurts 0 relative to that first sample. The small gain versus SA cannot be attributed to aggregation; recipe temperatures differ. Requested distinct seeds do not ensure independent draws.

### Debate

Completed-round distribution: {2: 188, 0: 1, 1: 11} (round means two successful responses; initial plus one revision). Full trajectories 188/200; initial disagreement 44; revision disagreement 26; selected structured prediction changes 109/199.
Against initial majority: 2 improve / 20 regress / 177 unchanged, coverage 199. Budget-stopped examples 11; schema failures 1. Stored majority-wrong labels: 10; peer-error-propagation heuristic labels: 64. The latter is convergence on a wrong answer, not causal evidence of influence.

### Router

Accept 147/200 (73.5%); escalate 53/200 (26.5%): {'accept': 147, 'critic': 50, 'debate': 3}. Reconstructed routes 6; none remain unknown.
| Route group | n | Mean score | Mean calls | Mean tokens |
| --- | --- | --- | --- | --- |
| accepted | 147 | 0.6624 | 1.000 | 531.28 |
| escalated | 53 | 0.5530 | 3.755 | 2912.06 |
Actual calls/example distribution: {1.0: 147, 3.0: 16, 4.0: 34, 5.0: 3}. Initial-to-final: 1 improved / 10 regressed / 189 unchanged. 91 accepted predictions are imperfect; 52/53 escalations do not improve the initial score. These retrospective heuristics do not establish whether an unexecuted alternative route would have helped.

Natural Router uses 1.75 calls and 1,198.07 tokens/example versus Debate's 3.99 calls and 2,801.45 tokens: approximately 56% fewer calls and 57% fewer tokens. Router still costs more than SA (1 call, 563.60 tokens).


# Failure Analysis

Only persisted frozen taxonomy annotations are counted. Labels overlap and are not an exhaustive error partition or independently adjudicated diagnoses.

## natural

| Stored label | SA | SC | CR | Debate | Router |
| --- | --- | --- | --- | --- | --- |
| UNKNOWN_PROVIDER_USAGE | 0 | 0 | 0 | 0 | 0 |
| INTERRUPTED_TRAJECTORY | 0 | 0 | 0 | 0 | 0 |
| HALLUCINATION | 43 | 43 | 46 | 42 | 39 |
| MISSED_EVIDENCE | 5 | 5 | 2 | 5 | 6 |
| WRONG_CITATION | 12 | 12 | 11 | 13 | 11 |
| OVER_ABSTENTION | 10 | 10 | 10 | 10 | 10 |
| UNDER_ABSTENTION | 2 | 3 | 2 | 4 | 4 |
| SCHEMA_FAILURE | 0 | 0 | 0 | 1 | 0 |
| NUMERIC_ERROR | 25 | 25 | 25 | 25 | 25 |
| DATE_NORMALIZATION_ERROR | 29 | 30 | 30 | 36 | 31 |
| NEGATION_ERROR | 0 | 0 | 0 | 0 | 0 |
| PEER_ERROR_PROPAGATION | 0 | 0 | 0 | 64 | 0 |
| MAJORITY_WRONG | 0 | 0 | 0 | 10 | 0 |
| CRITIC_REGRESSION | 0 | 0 | 8 | 0 | 0 |
| ROUTER_UNDER_ESCALATION | 0 | 0 | 0 | 0 | 91 |
| ROUTER_OVER_ESCALATION | 0 | 0 | 0 | 0 | 51 |

### Component Coverage

NULL unavailable metrics remain NULL, not zeros; non-applicable metrics do not enter component denominators. All primary task scores have n=200.

- SA: failed outputs 0; unavailable applicable components: none.
- SC: failed outputs 0; unavailable applicable components: none.
- CR: failed outputs 0; unavailable applicable components: none.
- Debate: failed outputs 1; unavailable applicable components: none.
- Router: failed outputs 0; unavailable applicable components: none.

### Returned Failure Operations

| Recipe | Parse/schema failures | Task | Role | Length stops | Failed output caps |
| --- | --- | --- | --- | --- | --- |
| SA | 0 | {} | {} | 0 | [] |
| SC | 0 | {} | {} | 0 | [] |
| CR | 0 | {} | {} | 0 | [] |
| Debate | 1 | {'structured_extraction': 1} | {'agent_1': 1} | 0 | [700] |
| Router | 0 | {} | {} | 0 | [] |

## matched_tokens

| Stored label | SA | SC | CR | Debate | Router |
| --- | --- | --- | --- | --- | --- |
| UNKNOWN_PROVIDER_USAGE | 0 | 0 | 0 | 0 | 0 |
| INTERRUPTED_TRAJECTORY | 0 | 0 | 0 | 0 | 0 |
| HALLUCINATION | 43 | 43 | 46 | 42 | 39 |
| MISSED_EVIDENCE | 5 | 5 | 2 | 5 | 6 |
| WRONG_CITATION | 12 | 12 | 11 | 13 | 11 |
| OVER_ABSTENTION | 10 | 10 | 10 | 10 | 10 |
| UNDER_ABSTENTION | 2 | 3 | 2 | 4 | 4 |
| SCHEMA_FAILURE | 0 | 0 | 0 | 1 | 6 |
| NUMERIC_ERROR | 25 | 25 | 25 | 25 | 25 |
| DATE_NORMALIZATION_ERROR | 29 | 30 | 30 | 36 | 31 |
| NEGATION_ERROR | 0 | 0 | 0 | 0 | 0 |
| PEER_ERROR_PROPAGATION | 0 | 0 | 0 | 64 | 0 |
| MAJORITY_WRONG | 0 | 0 | 0 | 10 | 0 |
| CRITIC_REGRESSION | 0 | 0 | 8 | 0 | 0 |
| ROUTER_UNDER_ESCALATION | 0 | 0 | 0 | 0 | 91 |
| ROUTER_OVER_ESCALATION | 0 | 0 | 0 | 0 | 46 |

### Component Coverage

NULL unavailable metrics remain NULL, not zeros; non-applicable metrics do not enter component denominators. All primary task scores have n=200.

- SA: failed outputs 0; unavailable applicable components: none.
- SC: failed outputs 0; unavailable applicable components: none.
- CR: failed outputs 0; unavailable applicable components: none.
- Debate: failed outputs 1; unavailable applicable components: none.
- Router: failed outputs 6; unavailable applicable components: {'action_item_correctness': 6, 'deadline_correctness': 6, 'decision_extraction_correctness': 6, 'exact_match': 6, 'owner_correctness': 6, 'risk_extraction_correctness': 6, 'supported_fact_coverage': 6, 'token_f1': 6}.

### Returned Failure Operations

| Recipe | Parse/schema failures | Task | Role | Length stops | Failed output caps |
| --- | --- | --- | --- | --- | --- |
| SA | 0 | {} | {} | 0 | [] |
| SC | 0 | {} | {} | 0 | [] |
| CR | 0 | {} | {} | 0 | [] |
| Debate | 1 | {'structured_extraction': 1} | {'agent_1': 1} | 0 | [700] |
| Router | 6 | {'business_summarization': 6} | {'reviser': 6} | 6 | [42, 108, 181, 192, 201, 253] |

Two Debate extraction failures are returned objects with a schema-invalid field; their provider usage is known. Six matched Router failures are unparseable length-terminated summaries at the reduced caps listed in the evidence table.

Frozen failed_attempts=0 counts transport failures; eight returned output failures are separately retained as SCHEMA_FAILURE / PARSE_ERROR. No retries, output coercion, replay or favorable fallback. Case A uses existing safe partial scoring; unparseable Case B uses primary zero and unavailable task components. The frozen failed-output contract in the JSON specifies all cases.

HALLUCINATION is imperfect reference-supported QA scoring, not a human finding. DATE_NORMALIZATION_ERROR and NEGATION_ERROR use frozen tag heuristics; zero labels do not prove no such error. PEER_ERROR_PROPAGATION means wrong convergence. NUMERIC_ERROR uses the frozen numeric-field mismatch heuristic (Python booleans also count as numeric); it is not a pure count of arithmetic mistakes. The reporting pass does not repair these frozen annotations.

Router's stored over-escalation labels are 51 natural / 46 matched. The read-only trajectory diagnostic finds 51 / 52 non-improving escalations because failed predictions lack candidate-based taxonomy annotations; these counts are explicitly different diagnostics, not retroactively rewritten failure labels.


# DEV Versus TEST

Corrected 32-example DEV pilot versus 200-example TEST. Different disjoint cohorts, not paired to one another. Both use their existing within-cohort paired statistics. No new weighting, tuning or exclusions. DEV source hashes and completed corrective attempt evidence were verified before analysis.

| Condition | Recipe | DEV delta [CI] | TEST delta [CI] | Direction | DEV CI includes zero |
| --- | --- | --- | --- | --- | --- |
| natural | SC | +0.0048 [+0.0000, +0.0135] | +0.0036 [-0.0013, +0.0091] | weakened | True |
| natural | CR | +0.0300 [-0.0174, +0.0984] | -0.0074 [-0.0220, +0.0054] | reversed | True |
| natural | Debate | -0.0200 [-0.0391, -0.0048] | -0.0181 [-0.0344, -0.0025] | weakened | False |
| natural | Router | +0.0157 [-0.0168, +0.0640] | +0.0069 [-0.0108, +0.0263] | weakened | True |
| matched_tokens | SC | +0.0048 [+0.0000, +0.0135] | +0.0036 [-0.0013, +0.0091] | weakened | True |
| matched_tokens | CR | +0.0300 [-0.0174, +0.0984] | -0.0074 [-0.0220, +0.0054] | reversed | True |
| matched_tokens | Debate | -0.0200 [-0.0391, -0.0048] | -0.0181 [-0.0344, -0.0025] | weakened | False |
| matched_tokens | Router | +0.0157 [-0.0168, +0.0640] | -0.0011 [-0.0204, +0.0190] | reversed | True |

SC's small positive direction weakens; Debate's negative direction replicates. CR changes from +0.0300 to -0.0074; natural Router's +0.0157 weakens to +0.0069, matched Router reverses to -0.0011. All DEV positive-gain intervals included zero, including SC's boundary at zero. This is generalization evidence, not a reason to tune TEST.

## Task Families

| Condition | Recipe | Task | DEV n | TEST n | DEV delta | TEST delta |
| --- | --- | --- | --- | --- | --- | --- |
| natural | SC | business_summarization | 8 | 50 | +0.0156 | +0.0110 |
| natural | SC | grounded_qa | 8 | 50 | +0.0000 | -0.0004 |
| natural | SC | robustness | 8 | 50 | +0.0036 | +0.0069 |
| natural | SC | structured_extraction | 8 | 50 | +0.0000 | -0.0031 |
| natural | CR | business_summarization | 8 | 50 | -0.0156 | +0.0000 |
| natural | CR | grounded_qa | 8 | 50 | -0.0104 | -0.0231 |
| natural | CR | robustness | 8 | 50 | +0.1650 | +0.0075 |
| natural | CR | structured_extraction | 8 | 50 | -0.0191 | -0.0139 |
| natural | Debate | business_summarization | 8 | 50 | +0.0000 | +0.0090 |
| natural | Debate | grounded_qa | 8 | 50 | +0.0000 | -0.0200 |
| natural | Debate | robustness | 8 | 50 | +0.0000 | +0.0088 |
| natural | Debate | structured_extraction | 8 | 50 | -0.0799 | -0.0703 |
| natural | Router | business_summarization | 8 | 50 | -0.0156 | +0.0005 |
| natural | Router | grounded_qa | 8 | 50 | +0.0000 | -0.0069 |
| natural | Router | robustness | 8 | 50 | +0.0977 | +0.0485 |
| natural | Router | structured_extraction | 8 | 50 | -0.0191 | -0.0144 |
| matched_tokens | SC | business_summarization | 8 | 50 | +0.0156 | +0.0110 |
| matched_tokens | SC | grounded_qa | 8 | 50 | +0.0000 | -0.0004 |
| matched_tokens | SC | robustness | 8 | 50 | +0.0036 | +0.0069 |
| matched_tokens | SC | structured_extraction | 8 | 50 | +0.0000 | -0.0031 |
| matched_tokens | CR | business_summarization | 8 | 50 | -0.0156 | +0.0000 |
| matched_tokens | CR | grounded_qa | 8 | 50 | -0.0104 | -0.0231 |
| matched_tokens | CR | robustness | 8 | 50 | +0.1650 | +0.0075 |
| matched_tokens | CR | structured_extraction | 8 | 50 | -0.0191 | -0.0139 |
| matched_tokens | Debate | business_summarization | 8 | 50 | +0.0000 | +0.0090 |
| matched_tokens | Debate | grounded_qa | 8 | 50 | +0.0000 | -0.0200 |
| matched_tokens | Debate | robustness | 8 | 50 | +0.0000 | +0.0088 |
| matched_tokens | Debate | structured_extraction | 8 | 50 | -0.0799 | -0.0703 |
| matched_tokens | Router | business_summarization | 8 | 50 | -0.0156 | -0.0315 |
| matched_tokens | Router | grounded_qa | 8 | 50 | +0.0000 | -0.0069 |
| matched_tokens | Router | robustness | 8 | 50 | +0.0977 | +0.0485 |
| matched_tokens | Router | structured_extraction | 8 | 50 | -0.0191 | -0.0144 |

DEV had 8 examples per task versus TEST's 50. CR's robustness advantage shrank from +0.1650 to +0.0075; Router's from +0.0977 to +0.0485. Extraction was already weak for Debate and remains below SA on TEST (-0.0703 versus DEV -0.0799). The 32-example pilot was not precise enough to establish general orchestration improvements.

Historical reused DEV runs do not have an independent provider-side completeness audit; clean corrective runs have explicit physical-attempt accounting. TEST's canonical client ledger is reconciled across all ten recipes.


# Pareto And Efficiency

Dominance uses mean primary score and one mean resource at a time, within each condition: at least as much quality and no more resource, with one strict inequality. Point-estimate dominance is not statistical dominance or a universal ranking.

| Condition | Resource | Nondominated | Dominated |
| --- | --- | --- | --- |
| matched_tokens | total_tokens | SA, SC | CR by SA/SC/Router; Debate by SA/SC/CR/Router; Router by SA/SC |
| matched_tokens | provider_attempts | SA, SC | CR by SA/SC/Router; Debate by SA/SC/CR/Router; Router by SA |
| matched_tokens | latency_ms | SA, SC | CR by SA/SC/Router; Debate by SA/SC/CR/Router; Router by SA/SC |
| matched_tokens | wall_clock_strategy_latency_ms | SA, SC | CR by SA/SC/Router; Debate by SA/SC/CR/Router; Router by SA/SC |
| natural | total_tokens | SA, SC, Router | CR by SA/SC/Router; Debate by SA/SC/CR/Router |
| natural | provider_attempts | SA, Router | SC by Router; CR by SA/SC/Router; Debate by SA/SC/CR/Router |
| natural | latency_ms | SA, SC, Router | CR by SA/SC/Router; Debate by SA/SC/CR/Router |
| natural | wall_clock_strategy_latency_ms | SA, SC, Router | CR by SA/SC/Router; Debate by SA/SC/CR/Router |

Natural Router's higher point-estimate quality comes at greater compute than SA. SC and Router trade tokens/latency against quality, but Router dominates SC on calls. CR and Debate are dominated on every listed resource axis. Under matched allowances, only SA and SC are nondominated on all axes. This is not evidence that SC aggregation helped: the frozen selector always chose its first sample.

Marginal local API price is zero for every run, so quality/$ is suppressed. Compute, energy and elapsed time still matter. Quality/1k tokens is the ratio of aggregate mean score to aggregate mean tokens; the JSON separately retains the existing mean of per-example ratios, which is a different quantity.


## Limitations

- One quantized local Gemma 3 4B model on a frozen synthetic Japanese enterprise benchmark; not evidence of performance on private enterprise data or other models.
- Recipe temperatures differ, and routing also uses its existing nested debate settings; orchestration is not causally isolated from sampling.
- Shared 12,000/4,000-token allowances are not identical realized spending or guaranteed hard token caps: provider-reported post-call overruns are retained and disclosed.
- Bootstrap intervals are descriptive, unadjusted for multiple subgroup comparisons; cells below five are too small for reliable interpretation.
- Groundedness and failure taxonomy are deterministic reference heuristics, not human entailment judgments. Convergence labels do not prove causal peer-error propagation.
- Router under/over-escalation labels are retrospective heuristics, not counterfactual proof of the optimal route. Accepted-versus-escalated groups are selected, not randomized.
- Protocol v3 was an operational/evaluation amendment after a returned schema failure. Valid metrics and scientific recipes stayed unchanged; all failures remain in the cohort.
- The historical failed trajectory's strategy wall-time is reconstructed from recorded logical lifecycles; attempt latency is measured. No end-to-end remeasurement occurred.
- Conditions ran sequentially on one laptop. Thermal state, model residency, and background load were not randomized; cross-condition latency differences are not causal budget effects.
- Client attempt completeness and pinned identities are audited, not independent server-side wire completeness. Seeds request reproducibility but do not guarantee independent or deterministic real-API samples.
- Local marginal API pricing is zero; compute, energy, and elapsed time are not universally free. Quality per dollar is not a meaningful discriminator here.
- HeterogeneousPanel, learned routing, other models, and ablations are outside this frozen held-out experiment. TEST was not used for routing features, training, or post-TEST tuning.
- Raw databases, private TEST examples, and local protocol bundles remain withheld. The public aggregate export is not a full raw-bundle reproduction release.

## Reproduction And Publication

Public aggregate export: docs/heldout_results.json. Private raw protocols, DBs, TEST data and per-example appendix are excluded from Git. Public clones can inspect aggregate results but cannot fully reproduce this run without the withheld bundle.

With the original local evidence, reproduce without model calls:

```sh
PYTHONPATH=src python3 -m analysis.heldout_final --check
```

The analysis is outside frozen source globs, preserving protocol code identity. The aggregate export binds analysis source hashes and the full raw evidence inventory hash; local integrity_report.json contains the full immutable evidence hash inventory.
