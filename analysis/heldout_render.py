"""Deterministic report rendering and aggregate-only public export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from analysis.heldout_final import FINAL, LABELS, ROOT, STRATEGIES, successful_outputs

CONDITIONS = ("natural", "matched_tokens")
SHORT = {
    "single_agent": "SA",
    "self_consistency": "SC",
    "critic_reviser": "CR",
    "multi_agent_debate": "Debate",
    "adaptive_router": "Router",
}
PITCH = (
    "I built CollectiveEval to measure whether multi-agent LLM orchestration improves Japanese "
    "enterprise-task quality under shared inference budgets. Its frozen 200-example held-out "
    "study reconciled 4,510 real local-model calls and found that added orchestration often "
    "increased cost without improving quality, while selective routing exposed useful "
    "task-specific gains and budget-induced output failures."
)
RESUME = [
    "Built a six-strategy LLM evaluation framework with frozen synthetic Japanese benchmarks, "
    "deterministic evaluators, and paired bootstrap comparisons under shared inference allowances.",
    "Completed a five-recipe, two-condition held-out study: 2,000 predictions, 4,510 audited "
    "Ollama attempts, and 3.01M provider-reported tokens, with no retries or unknown usage.",
    "Engineered immutable continuation certificates, per-example checkpoints, and usage-preserving "
    "parse-failure accounting to resume interrupted experiments without replaying TEST work.",
    "Analyzed routing and orchestration mechanisms: heuristic routing accepted 73.5% of examples "
    "and used 56% fewer natural-condition calls than Debate; documented regressions, truncation "
    "failures, and uncertainty rather than claiming universal multi-agent gains.",
]


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(str(x) for x in row) + " |" for row in rows],
        ]
    )


def interval(data: dict[str, Any]) -> str:
    return f"[{data['ci_lower']:+.4f}, {data['ci_upper']:+.4f}]"


def score_table(result: dict[str, Any], condition: str) -> str:
    rows = []
    for strategy in STRATEGIES:
        score = result["strategy_results"][condition][strategy]["metrics"]["task_score"]
        delta = result["paired_deltas"][condition][strategy]["task_score"]
        rows.append(
            [
                LABELS[strategy],
                score["n"],
                f"{score['mean']:.4f}",
                interval(score),
                f"{delta['mean_difference']:+.4f}",
                interval(delta),
            ]
        )
    return table(
        ["Recipe", "n", "Mean score", "Score 95% CI", "Delta vs SA", "Paired 95% CI"], rows
    )


def resource_table(result: dict[str, Any], condition: str) -> str:
    def pair(data: dict[str, Any], scale: float = 1) -> str:
        return f"{data['mean'] / scale:.2f} / {data['median'] / scale:.2f}"

    rows = []
    for strategy in STRATEGIES:
        data = result["strategy_results"][condition][strategy]
        m = data["metrics"]
        rows.append(
            [
                SHORT[strategy],
                pair(m["logical_model_calls"]),
                pair(m["provider_attempts"]),
                pair(m["total_tokens"]),
                pair(m["latency_ms"], 1000),
                pair(m["logical_call_latency_ms"], 1000),
                pair(m["wall_clock_strategy_latency_ms"], 1000),
                f"{data['quality_per_1k_tokens']:.4f}",
            ]
        )
    return table(
        [
            "Recipe",
            "Calls mean/median",
            "Attempts mean/median",
            "Tokens mean/median",
            "Attempt s mean/median",
            "Lifecycle s mean/median",
            "Wall s mean/median",
            "Quality/1k tokens",
        ],
        rows,
    )


def breakdown(result: dict[str, Any], dimension: str) -> str:
    sections = []
    for condition in CONDITIONS:
        data = result["strategy_results"][condition]
        rows = []
        for group, base in data["single_agent"]["breakdown"][dimension].items():
            cells = []
            for strategy in STRATEGIES[1:]:
                item = data[strategy]["breakdown"][dimension][group]
                delta = item["paired_vs_single_agent"]
                cells.append(
                    f"{item['mean']:.4f}; {delta['mean_difference']:+.4f} " + interval(delta)
                )
            rows.append(
                [
                    group,
                    str(base["n"]) + (" (TINY)" if base["tiny_cell"] else ""),
                    f"{base['mean']:.4f}",
                    *cells,
                ]
            )
        sections += [
            f"### {condition}",
            "",
            table(
                [
                    "Group",
                    "n",
                    "SA mean",
                    "SC mean; delta [CI]",
                    "CR mean; delta [CI]",
                    "Debate mean; delta [CI]",
                    "Router mean; delta [CI]",
                ],
                rows,
            ),
            "",
        ]
    return "\n".join(sections)


def condition_report(result: dict[str, Any]) -> str:
    rows = []
    for strategy, data in result["condition_comparison"].items():
        p = data["paired"]
        rows.append(
            [
                SHORT[strategy],
                f"{p['task_score']['mean_difference']:+.4f} " + interval(p["task_score"]),
                f"{p['total_tokens']['mean_difference']:+.2f}",
                f"{p['provider_attempts']['mean_difference']:+.3f}",
                f"{p['latency_ms']['mean_difference'] / 1000:+.2f}",
                data["budget_rejection_delta"],
                data["post_call_overrun_delta"],
            ]
        )
    return "\n".join(
        [
            "## Matched Minus Natural",
            "",
            table(
                [
                    "Recipe",
                    "Score delta [paired CI]",
                    "Tokens/example delta",
                    "Calls/example delta",
                    "Attempt s/example delta",
                    "Rejection event delta",
                    "Overrun delta",
                ],
                rows,
            ),
            "",
            "The first four recipes have identical primary scores across conditions; "
            "resources need not be identical. Router loses 0.0080 aggregate score, "
            "entirely in business summarization "
            "(family delta -0.0320). Six matched Router revision responses terminate at their "
            "shortened output caps with finish_reason=length and fail JSON parsing. "
            "They remain in the denominator "
            "with the frozen failed-output score semantics, not repaired initial-answer fallbacks.",
            "",
            "The six failures contribute -0.008875 to the full-cohort mean difference. "
            "One changed valid summary contributes +0.000875; the other 193 scores "
            "are unchanged. Net difference is -0.0080, not a loss attributable only "
            "to the six failures with no other output changes.",
            "",
            "Shared ceilings are not identical realized spending or hard caps. The 4,000-token "
            "condition has 15 pre-call rejection events and 19 retained overrun examples: CR 2, "
            "Debate 7, Router 10. Maximum actual tokens are 4,455 / 4,173 / 4,417 respectively. "
            "Therefore this study supports a comparison under shared admission allowances, not a "
            "claim of perfectly enforced strict token parity.",
            "",
            "Sequential laptop latency is confounded by run order, thermal state and load: even SA "
            "and SC use identical tokens but take longer in the matched condition. "
            "Do not attribute "
            "all latency differences to the budget policy.",
            "",
            "### Central Question",
            "",
            "Under these frozen 4,000-token allowances, no orchestration recipe has a "
            "higher overall mean than SC; CR and Debate are below SA, "
            "and Router is slightly below SA. Under natural "
            "allowances, Router has the highest overall point estimate at extra compute, but its "
            "paired interval versus SA spans zero. Router's robustness-family gain is +0.0485 "
            "(n=50) in both conditions, while extraction worsens. Debate's extraction delta is "
            "-0.0703; CR's QA delta is -0.0231. These are descriptive, "
            "unadjusted subgroup findings, "
            "not a general superiority claim or proof that orchestration caused the differences.",
            "",
        ]
    )


def mechanism_report(result: dict[str, Any]) -> str:
    lines = [
        "# Mechanism Analysis",
        "",
        "Initial/final comparisons use the unchanged evaluator offline. No gold-derived "
        "value enters routing inference. Coverage is explicit; missing candidate histories "
        "are not fabricated.",
        "",
    ]
    for condition in CONDITIONS:
        data = result["strategy_results"][condition]
        sc, cr, debate, router = [data[s]["mechanism"] for s in STRATEGIES[1:]]
        lines += [
            f"## {condition}",
            "",
            "### CriticReviser",
            "",
            f"Revisions: {cr['revision_count']}/200 ({cr['revision_rate']:.1%}); explicit "
            f"no-revision: {cr['explicit_no_revision_count']}/200 "
            f"({cr['no_revision_rate']:.1%}); budget-stopped examples: "
            f"{cr['budget_stopped_examples']}; returned output failures: "
            f"{cr['failed_outputs']}.",
            f"Revised mean {cr['score_revised']['task_score']:.4f}; unrevised mean "
            f"{cr['score_unrevised']['task_score']:.4f}. These selected groups are not "
            "randomized. Against the generator: "
            f"{cr['improved']} improved / {cr['regressed']} regressed / "
            f"{cr['unchanged']} unchanged; coverage {cr['initial_comparison_coverage']}.",
            "",
            "### SelfConsistency",
            "",
            f"Two candidates per example: {sc['candidate_count_distribution']}. Canonical "
            f"diversity {sc['answer_diverse']}/200 ({sc['diversity_rate']:.1%}); full-object "
            f"diversity {sc['full_output_diverse']}/200. There are {sc['ties']} K=2 ties "
            "and 145 agreement cases. Frozen tie resolution selects sample zero on all "
            "200 examples: aggregation helps 0 / hurts 0 relative to that first sample. "
            "The small gain versus SA cannot be attributed to aggregation; recipe "
            "temperatures differ. Requested distinct seeds do not ensure independent draws.",
            "",
            "### Debate",
            "",
            f"Completed-round distribution: {debate['completed_rounds_distribution']} "
            "(round means two successful responses; initial plus one revision). "
            f"Full trajectories {debate['full_two_round_trajectories']}/200; initial "
            f"disagreement {debate['initial_disagreements']}; revision disagreement "
            f"{debate['revision_disagreements']}; selected structured prediction changes "
            f"{debate['changed_selected_prediction']}/{debate['round_comparison_coverage']}.",
            f"Against initial majority: {debate['improved']} improve / "
            f"{debate['regressed']} regress / {debate['unchanged']} unchanged, coverage "
            f"{debate['initial_comparison_coverage']}. Budget-stopped examples "
            f"{debate['budget_stopped_examples']}; schema failures "
            f"{debate['failed_outputs']}. "
            "Stored majority-wrong labels: 10; peer-error-propagation heuristic labels: 64. "
            "The latter is convergence on a wrong answer, not causal evidence of influence.",
            "",
            "### Router",
            "",
            f"Accept {router['accepted']}/200 (73.5%); escalate {router['escalated']}/200 "
            f"(26.5%): {router['route_counts']}. Reconstructed routes "
            f"{router['route_reconstructed']}; none remain unknown.",
            table(
                ["Route group", "n", "Mean score", "Mean calls", "Mean tokens"],
                [
                    [
                        name,
                        group["n"],
                        f"{group['task_score']:.4f}",
                        f"{group['provider_attempts']:.3f}",
                        f"{group['total_tokens']:.2f}",
                    ]
                    for name, group in router["accepted_vs_escalated"].items()
                ],
            ),
            f"Actual calls/example distribution: {router['actual_calls_distribution']}. "
            f"Initial-to-final: {router['improved']} improved / {router['regressed']} "
            f"regressed / {router['unchanged']} unchanged. "
            f"{router['accepted_imperfect']} accepted predictions are imperfect; "
            f"{router['retrospective_nonimproving_escalations']}/53 escalations do not "
            "improve the initial score. These retrospective heuristics do not establish "
            "whether an unexecuted alternative route would have helped.",
            "",
        ]
    lines += [
        "Natural Router uses 1.75 calls and 1,198.07 tokens/example versus Debate's "
        "3.99 calls and 2,801.45 tokens: approximately 56% fewer calls and 57% fewer tokens. "
        "Router still costs more than SA (1 call, 563.60 tokens).",
        "",
    ]
    return "\n".join(lines)


def failure_report(result: dict[str, Any]) -> str:
    lines = [
        "# Failure Analysis",
        "",
        "Only persisted frozen taxonomy annotations are counted. Labels overlap and are "
        "not an exhaustive error partition or independently adjudicated diagnoses.",
        "",
    ]
    for condition in CONDITIONS:
        data = result["strategy_results"][condition]
        labels = list(data["single_agent"]["failure_counts"])
        lines += [
            f"## {condition}",
            "",
            table(
                ["Stored label", *[SHORT[s] for s in STRATEGIES]],
                [
                    [label, *[data[s]["failure_counts"][label] for s in STRATEGIES]]
                    for label in labels
                ],
            ),
            "",
            "### Component Coverage",
            "",
            "NULL unavailable metrics remain NULL, not zeros; non-applicable metrics "
            "do not enter component denominators. All primary task scores have n=200.",
            "",
        ]
        for strategy in STRATEGIES:
            cov = data[strategy]["component_coverage"]
            missing = {
                k: v["n_unavailable"] for k, v in cov["components"].items() if v["n_unavailable"]
            }
            lines.append(
                f"- {SHORT[strategy]}: failed outputs {cov['failed_output_count']}; "
                f"unavailable applicable components: {missing or 'none'}."
            )
        lines += [
            "",
            "### Returned Failure Operations",
            "",
            table(
                [
                    "Recipe",
                    "Parse/schema failures",
                    "Task",
                    "Role",
                    "Length stops",
                    "Failed output caps",
                ],
                [
                    [
                        SHORT[s],
                        data[s]["accounting"]["parse_failures"],
                        data[s]["returned_output_failures"]["by_task"],
                        data[s]["returned_output_failures"]["by_role"],
                        data[s]["returned_output_failures"]["length_terminated"],
                        data[s]["returned_output_failures"]["failed_output_caps"],
                    ]
                    for s in STRATEGIES
                ],
            ),
            "",
        ]
    lines += [
        "Two Debate extraction failures are returned objects with a schema-invalid field; "
        "their provider usage is known. Six matched Router failures are unparseable "
        "length-terminated summaries at the reduced caps listed in the evidence table.",
        "",
        "Frozen failed_attempts=0 counts transport failures; eight returned output failures "
        "are separately retained as SCHEMA_FAILURE / PARSE_ERROR. No retries, output "
        "coercion, replay or favorable fallback. Case A uses existing safe partial scoring; "
        "unparseable Case B uses primary zero and unavailable task components. The frozen "
        "failed-output contract in the JSON specifies all cases.",
        "",
        "HALLUCINATION is imperfect reference-supported QA scoring, not a human finding. "
        "DATE_NORMALIZATION_ERROR and NEGATION_ERROR use frozen tag heuristics; zero "
        "labels do not prove no such error. PEER_ERROR_PROPAGATION means wrong convergence. "
        "NUMERIC_ERROR uses the frozen numeric-field mismatch heuristic (Python booleans "
        "also count as numeric); it is not a pure count of arithmetic mistakes. "
        "The reporting pass does not repair these frozen annotations.",
        "",
        "Router's stored over-escalation labels are 51 natural / 46 matched. The read-only "
        "trajectory diagnostic finds 51 / 52 non-improving escalations because failed "
        "predictions lack candidate-based taxonomy annotations; these counts are explicitly "
        "different diagnostics, not retroactively rewritten failure labels.",
        "",
    ]
    return "\n".join(lines)


def dev_report(result: dict[str, Any]) -> str:
    rows, families = [], []
    dev = result["dev_vs_test"]
    for condition in CONDITIONS:
        for s in STRATEGIES[1:]:
            r = dev["comparisons"][condition][s]
            rows.append(
                [
                    condition,
                    SHORT[s],
                    f"{r['dev']['mean_difference']:+.4f} " + interval(r["dev"]),
                    f"{r['test']['mean_difference']:+.4f} " + interval(r["test"]),
                    r["direction"],
                    r["dev_interval_includes_zero"],
                ]
            )
            for task, value in r["task_family_deltas"].items():
                families.append(
                    [
                        condition,
                        SHORT[s],
                        task,
                        value["dev_n"],
                        value["test_n"],
                        f"{value['dev_delta']:+.4f}",
                        f"{value['test_delta']:+.4f}",
                    ]
                )
    return "\n".join(
        [
            "# DEV Versus TEST",
            "",
            "Corrected 32-example DEV pilot versus 200-example TEST. Different disjoint cohorts, "
            "not paired to one another. Both use their existing within-cohort paired statistics. "
            "No new weighting, tuning or exclusions. DEV source hashes and completed corrective "
            "attempt evidence were verified before analysis.",
            "",
            table(
                [
                    "Condition",
                    "Recipe",
                    "DEV delta [CI]",
                    "TEST delta [CI]",
                    "Direction",
                    "DEV CI includes zero",
                ],
                rows,
            ),
            "",
            "SC's small positive direction weakens; Debate's negative direction replicates. "
            "CR changes from +0.0300 to -0.0074; natural Router's +0.0157 weakens to +0.0069, "
            "matched Router reverses to -0.0011. All DEV positive-gain intervals included zero, "
            "including SC's boundary at zero. This is generalization evidence, "
            "not a reason to tune TEST.",
            "",
            "## Task Families",
            "",
            table(
                ["Condition", "Recipe", "Task", "DEV n", "TEST n", "DEV delta", "TEST delta"],
                families,
            ),
            "",
            "DEV had 8 examples per task versus TEST's 50. CR's robustness advantage shrank from "
            "+0.1650 to +0.0075; Router's from +0.0977 to +0.0485. Extraction was already weak for "
            "Debate and remains below SA on TEST (-0.0703 versus DEV -0.0799). "
            "The 32-example pilot was not precise "
            "enough to establish general orchestration improvements.",
            "",
            "Historical reused DEV runs do not have an independent provider-side completeness "
            "audit; clean corrective runs have explicit physical-attempt accounting. TEST's "
            "canonical client ledger is reconciled across all ten recipes.",
            "",
        ]
    )


def pareto_report(result: dict[str, Any]) -> str:
    rows = []
    for condition, resources in result["pareto"].items():
        for resource, strategies in resources.items():
            rows.append(
                [
                    condition,
                    resource,
                    ", ".join(SHORT[s] for s, v in strategies.items() if v["nondominated"]),
                    "; ".join(
                        SHORT[s] + " by " + "/".join(SHORT[x] for x in v["dominated_by"])
                        for s, v in strategies.items()
                        if v["dominated_by"]
                    ),
                ]
            )
    return "\n".join(
        [
            "# Pareto And Efficiency",
            "",
            "Dominance uses mean primary score and one mean resource at a time, within each "
            "condition: at least as much quality and no more resource, with one strict inequality. "
            "Point-estimate dominance is not statistical dominance or a universal ranking.",
            "",
            table(["Condition", "Resource", "Nondominated", "Dominated"], rows),
            "",
            "Natural Router's higher point-estimate quality comes at greater compute than SA. "
            "SC and Router trade tokens/latency against quality, but Router dominates SC on calls. "
            "CR and Debate are dominated on every listed resource axis. Under matched allowances, "
            "only SA and SC are nondominated on all axes. This is not evidence that SC aggregation "
            "helped: the frozen selector always chose its first sample.",
            "",
            "Marginal local API price is zero for every run, so quality/$ is suppressed. Compute, "
            "energy and elapsed time still matter. Quality/1k tokens is the ratio of aggregate "
            "mean score to aggregate mean tokens; the JSON separately retains the existing mean "
            "of per-example ratios, which is a different quantity.",
            "",
        ]
    )


def final_summary(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# CollectiveEval: Final Research Summary",
            "",
            PITCH,
            "",
            "## Study",
            "",
            "Five frozen recipes, two token-allowance conditions (12,000 natural / 4,000 matched), "
            "200 synthetic Japanese TEST examples across QA, extraction, business summaries and "
            "robustness. Local quantized Gemma 3 4B, concurrency one, zero API pricing. "
            "Six-strategy framework; heterogeneous panels and learned routing were not evaluated "
            "in this held-out study.",
            "",
            score_table(result, "natural"),
            "",
            "Matched means are identical except Router, which falls from 0.6414 to 0.6334; "
            "matched SA / SC / CR / Debate are 0.6345 / 0.6381 / 0.6271 / 0.6164.",
            "",
            "## Findings",
            "",
            "- More orchestration did not imply more quality: CR reversed its small DEV advantage, "
            "and Debate underperformed SA at roughly five times its natural tokens.",
            "- Selective routing accepted 73.5% of examples, used 56% fewer calls than Debate, "
            "and improved robustness by +0.0485. Its overall natural gain has an interval spanning "
            "zero, and six truncated matched revisions remove that overall advantage.",
            "- SC's 27.5% candidate diversity yielded no aggregation improvement: "
            "all 200 selections "
            "were the first sample. Frozen recipes also use different temperatures.",
            "",
            "## Engineering Lesson",
            "",
            "Scientific reliability depends on immutable cohorts, real attempt/usage accounting, "
            "resume provenance and explicit invalid-output denominators. The audit reconciled "
            "2,000 predictions, 4,510 calls/attempts and 3,012,623 reported tokens, "
            "preserving eight "
            "output failures, 15 admission rejections and 19 overruns "
            "without replay or repair. "
            "A shared admission allowance is not proof of a strict realized token cap.",
            "",
            "## Limits",
            "",
            *["- " + item for item in result["limitations"]],
            "",
        ]
    )


def public_view(result: dict[str, Any]) -> dict[str, Any]:
    # Publish only aggregate structures; never raw rows, example IDs, prompts, or predictions.
    top = (
        "artifact_kind",
        "protocol_sha256",
        "statistics",
        "benchmark_version",
        "test_sha256",
        "dev_sha256",
        "test_n",
        "task_counts",
        "difficulty_counts",
        "paired_deltas",
        "condition_comparison",
        "pareto",
        "model",
        "conditions",
        "accounting_totals",
        "raw_evidence_inventory_sha256",
        "analysis_source_hashes",
        "failure_contract",
        "limitations",
        "no_model_calls",
        "dev_vs_test",
    )
    allowed = (
        "run_id",
        "metrics",
        "breakdown",
        "quality_per_1k_tokens",
        "quality_per_1k_definition",
        "mean_example_quality_per_1k_tokens",
        "marginal_api_cost_usd",
        "quality_per_dollar",
        "failure_counts",
        "failure_by_task",
        "component_coverage",
        "accounting",
        "token_budget",
        "returned_output_failures",
    )
    mechanism_keys = (
        "n",
        "failed_outputs",
        "budget_stopped_examples",
        "initial_comparison_coverage",
        "improved",
        "regressed",
        "unchanged",
        "revision_count",
        "revision_rate",
        "explicit_no_revision_count",
        "no_revision_rate",
        "score_revised",
        "score_unrevised",
        "candidate_count_distribution",
        "full_output_diverse",
        "answer_diverse",
        "diversity_rate",
        "ties",
        "selected_sample_distribution",
        "interpretation",
        "completed_rounds_distribution",
        "full_two_round_trajectories",
        "initial_disagreements",
        "revision_disagreements",
        "changed_selected_prediction",
        "round_comparison_coverage",
        "route_counts",
        "accepted",
        "escalated",
        "accepted_vs_escalated",
        "retrospective_nonimproving_escalations",
        "accepted_imperfect",
        "route_reconstructed",
        "actual_calls_distribution",
    )
    view = {key: result[key] for key in top}
    view["provenance"] = [
        {key: row[key] for key in ("run_id", "strategy", "condition", "provenance")}
        for row in result["provenance"]
    ]
    view["strategy_results"] = {}
    for condition, strategies in result["strategy_results"].items():
        view["strategy_results"][condition] = {}
        for strategy, data in strategies.items():
            view["strategy_results"][condition][strategy] = {
                **{key: data[key] for key in allowed},
                "mechanism": {
                    key: data["mechanism"][key]
                    for key in mechanism_keys
                    if key in data["mechanism"]
                },
            }
    view["public_export_policy"] = (
        "Aggregate only; private TEST examples, per-example predictions, raw responses, "
        "databases, prompts and machine paths are withheld."
    )
    return cast(dict[str, Any], json.loads(json_text(view)))


def select_representatives(
    result: dict[str, Any], all_rows: dict[str, dict[str, list[dict[str, Any]]]]
) -> list[dict[str, Any]]:
    categories = [
        "orchestration helps",
        "orchestration hurts",
        "critic fixes error",
        "critic regression",
        "debate helps",
        "debate hurts",
        "router correctly accepts",
        "router correctly escalates",
        "router under-escalates",
        "router over-escalates",
        "citation/grounding failure",
        "schema failure",
    ]
    eligible: dict[str, list[dict[str, Any]]] = {name: [] for name in categories}
    for condition in CONDITIONS:
        base = {r["example_id"]: r for r in all_rows[condition]["single_agent"]}
        for strategy in STRATEGIES:
            mechanism_rows = result["strategy_results"][condition][strategy]["mechanism"]["details"]
            details = {d["example_id"]: d for d in mechanism_rows}
            for row in all_rows[condition][strategy]:
                d = details[row["example_id"]]
                delta = row["task_score"] - base[row["example_id"]]["task_score"]
                initial_delta = d.get("change_from_initial")
                tests = {
                    "orchestration helps": strategy in STRATEGIES[2:] and delta > 1e-12,
                    "orchestration hurts": strategy in STRATEGIES[2:] and delta < -1e-12,
                    "critic fixes error": strategy == "critic_reviser"
                    and initial_delta is not None
                    and initial_delta > 1e-12,
                    "critic regression": strategy == "critic_reviser"
                    and initial_delta is not None
                    and initial_delta < -1e-12,
                    "debate helps": strategy == "multi_agent_debate" and delta > 1e-12,
                    "debate hurts": strategy == "multi_agent_debate" and delta < -1e-12,
                    "router correctly accepts": d.get("route") == "accept"
                    and row["task_score"] == 1,
                    "router correctly escalates": d.get("route") in {"critic", "debate"}
                    and initial_delta is not None
                    and initial_delta > 1e-12,
                    "router under-escalates": "ROUTER_UNDER_ESCALATION" in row["failure_types"],
                    "router over-escalates": "ROUTER_OVER_ESCALATION" in row["failure_types"],
                    "citation/grounding failure": "WRONG_CITATION" in row["failure_types"],
                    "schema failure": row.get("kind") == "failed_prediction",
                }
                for category, passed in tests.items():
                    if passed:
                        eligible[category].append(
                            {
                                "condition": condition,
                                "strategy": strategy,
                                "row": row,
                                "baseline": base[row["example_id"]],
                                "decision": d,
                            }
                        )
    selected = []
    for category, choices in eligible.items():
        ordered = sorted(
            choices,
            key=lambda x: (
                CONDITIONS.index(x["condition"]),
                x["row"]["example_id"],
                STRATEGIES.index(x["strategy"]),
            ),
        )
        selected.append(
            {
                "category": category,
                "eligible_count": len(ordered),
                "selection": ordered[0] if ordered else None,
            }
        )
    return selected


def safe_prediction(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: safe_prediction(v)
            for k, v in value.items()
            if k
            not in {
                "raw_content",
                "raw_output",
                "raw_response",
                "reasoning",
                "analysis",
                "chain_of_thought",
                "thoughts",
                "rationale",
            }
        }
    if isinstance(value, list):
        return [safe_prediction(v) for v in value]
    return value


def representative_report(result: dict[str, Any], rows: dict[str, Any]) -> str:
    lines = [
        "# Representative Cases (Private; Do Not Publish)",
        "",
        "Selection is deterministic: for each fixed category, choose the first eligible "
        "trajectory by condition (natural first), lexicographic example ID, then frozen "
        "recipe order. Counts include both conditions. No ranking by favorable effect size. "
        "Categories may reuse cases. Correct routing means this observed scoring criterion, "
        "not an optimal counterfactual. No raw responses, critic arguments, or hidden "
        "chain-of-thought are printed. Synthetic inputs and final structured outputs are "
        "private because the frozen TEST cohort is withheld.",
        "",
    ]
    for case in select_representatives(result, rows):
        lines += [
            f"## {case['category']}",
            "",
            f"Eligible trajectories: {case['eligible_count']}.",
            "",
        ]
        chosen = case["selection"]
        if chosen is None:
            lines += ["No reconstructable case.", ""]
            continue
        row, base, decision = chosen["row"], chosen["baseline"], chosen["decision"]
        example = row["example"]
        payload = {
            "condition": chosen["condition"],
            "strategy": chosen["strategy"],
            "example_id": row["example_id"],
            "task": row["task"],
            "input_excerpt": json.dumps(example.input, ensure_ascii=False)[:1200],
            "baseline_final_prediction": safe_prediction(base["output"]),
            "baseline_task_score": base["task_score"],
            "final_task_score": row["task_score"],
            "final_prediction": safe_prediction(row["output"])
            if not decision["failed_output"]
            else {
                "kind": "failed_prediction",
                "role": row["output"]["role"],
                "schema_valid": row["output"]["schema_valid"],
                "parsed_json": safe_prediction(row["output"]["parsed_json"]),
            },
            "stored_failure_labels": row["failure_types"],
            "decision": decision,
            "logical_calls": row["logical_model_calls"],
            "tokens": row["total_tokens"],
        }
        initial = successful_outputs(row, {"generator", "cheap_single_agent"})
        if initial:
            payload["initial_structured_prediction"] = safe_prediction(initial[0]["output"])
        lines += ["```json", json_text(payload).rstrip(), "```", ""]
    return "\n".join(lines)


def application_report() -> str:
    portfolio = (
        "CollectiveEval evaluates whether multi-agent LLM orchestration earns its inference "
        "overhead "
        "on synthetic Japanese enterprise tasks. I implemented six strategies, frozen benchmarks, "
        "deterministic evaluators, and budget-aware execution with checkpointed provider "
        "accounting. "
        "The completed held-out study compared five recipes across two token allowances using "
        "200 examples, 2,000 predictions, and 4,510 local Gemma calls. Debate lost quality while "
        "spending more tokens; selective routing accepted most examples and exposed robustness "
        "gains alongside budget-induced truncation failures. Immutable continuation certificates "
        "preserved interrupted runs without replay. Reports disclose paired uncertainty, realized "
        "spending, invalid-output denominators, and limitations rather than presenting added "
        "orchestration as automatically beneficial or universally superior."
    )
    technical = (
        "CollectiveEval is a framework for evaluating single-agent, self-consistency, "
        "critic-reviser, debate, heterogeneous-panel, and adaptive-router inference through "
        "a shared "
        "task interface. I developed frozen synthetic Japanese benchmarks covering grounded QA, "
        "structured extraction, business summarization, and robustness, with schema validation, "
        "citation checks, abstention metrics, and failure annotations. The scientific runner "
        "separates strategy calls from physical provider attempts, preserves returned token usage "
        "even when parsing fails, and checkpoints examples for interruption-safe continuation. "
        "Hash-bound protocols and reuse certificates protect completed evidence without replaying "
        "held-out work. The final experiment used local quantized Gemma 3 4B, five frozen recipes, "
        "two token allowances, and 200 TEST examples per recipe. All 2,000 predictions and 4,510 "
        "attempts reconciled with 3,012,623 reported tokens and no retries or unknown usage. "
        "Paired bootstrap analysis found Debate below SingleAgent, CriticReviser reversing its "
        "DEV gain, and selective routing producing robustness gains without a clear aggregate "
        "advantage. Mechanism analysis revealed that self-consistency always selected its first "
        "sample and six budget-limited router revisions produced truncated JSON. Reports "
        "distinguish shared admission allowances from strict realized ceilings, disclose "
        "overruns, and preserve failed-output denominators. The work demonstrates "
        "evaluation engineering, resource-aware orchestration, reproducible analysis, "
        "and honest interpretation; limitations "
        "include one model, synthetic data, heuristic groundedness, differing temperatures, and "
        "unadjusted subgroup comparisons."
    )
    return "\n".join(
        [
            "# Sakana Application Snapshot",
            "",
            "## Two-Sentence Pitch",
            "",
            PITCH,
            "",
            "## Resume Bullets",
            "",
            *["- " + item for item in RESUME],
            "",
            f"## Portfolio Description ({len(portfolio.split())} Words)",
            "",
            portfolio,
            "",
            f"## Technical Description ({len(technical.split())} Words)",
            "",
            technical,
            "",
            "## Three Held-Out Findings",
            "",
            "1. Debate lost 0.0181 mean score versus SA, "
            "at 2,801 versus 564 natural tokens/example.",
            "2. Router accepted 73.5%, used 56% fewer natural calls than Debate, and gained 0.0485 "
            "on robustness; its overall natural paired interval spans zero.",
            "3. SC's 27.5% candidate diversity did not improve aggregation; six shortened matched "
            "Router revisions failed parsing and one valid summary improved, "
            "for a net mean difference of -0.0080 versus natural.",
            "",
            "## Lesson",
            "",
            "An evaluation is only as credible as its cohort integrity, attempt accounting, "
            "invalid-output semantics and resume evidence; shared budget labels alone do not "
            "establish fair realized compute or better decisions.",
            "",
            "## Product Role Fit",
            "",
            "This project fits Applied Research Engineer (Product) by connecting "
            "research questions to reliable execution, measurable quality-resource "
            "tradeoffs and decision-ready reports.",
            "",
            "## Honest Limits",
            "",
            "One quantized model and synthetic benchmark; descriptive unadjusted intervals, "
            "heuristic groundedness, differing recipe temperatures, retained token overruns, "
            "sequential laptop "
            "latency, and private raw evidence. No universal winner or confirmatory causal claim.",
            "",
            "Repository: https://github.com/Abdullah-Raashid/CollectiveEval",
            "",
        ]
    )


def full_report(result: dict[str, Any]) -> str:
    totals = result["accounting_totals"]
    return "\n".join(
        [
            "# CollectiveEval: Frozen Held-Out Results",
            "",
            "The experiment is complete. This is read-only post-run analysis, with no additional "
            "inference, TEST tuning, prediction changes or exclusions.",
            "",
            "## Setup And Integrity",
            "",
            "Integrity PASSED: ten canonical runs, exactly 200 unique TEST IDs per recipe, "
            "2,000 predictions. Benchmark v3.1 is synthetic; 50 examples per task family, "
            "60 easy / 60 medium / 80 hard. Local Ollama Gemma 3 4B (Q4_K_M), concurrency 1, "
            "no retries, 700 configured output tokens/call, up to 6 attempts/example. "
            "Requested temperatures: SA 0.2, SC 0.6, CR 0.2, Debate 0.5, Router 0.2 "
            "(nested debate retains 0.5); top_p=1; seed=20261003. Natural 12,000 and matched "
            "4,000 total token allowances. No learned-router or heterogeneous-panel results.",
            "",
            f"Protocol v3 SHA-256: `{result['protocol_sha256']}`.",
            f"Model digest: `{result['model']['digest']}`.",
            f"TEST SHA-256: `{result['test_sha256']}`.",
            f"DEV SHA-256: `{result['dev_sha256']}`.",
            "",
            "Frozen percentile/paired bootstrap: 1,000 resamples, seed 20261003, 95% intervals. "
            "Primary n always 200, including invalid outputs. No automatic significance claims; "
            "subgroup intervals are descriptive and not multiplicity-adjusted. Scores are "
            "heterogeneous deterministic task metrics, not a single accuracy percentage.",
            "",
            "### Canonical Provenance",
            "",
            table(
                ["Condition", "Recipe", "Run ID", "Continuation provenance"],
                [
                    [r["condition"], SHORT[r["strategy"]], r["run_id"], r["provenance"]]
                    for r in result["provenance"]
                ],
            ),
            "",
            "V1 SA is carried read-only; v2 SC/CR checkpoints are retained; natural Debate is "
            "31 completed v2 examples + one preserved returned failure + 168 v3 examples. "
            "Six other recipes are v3 executions. Certificates, model/config identities, artifact "
            "hashes and stored accounting reconcile; v1/v2/v3 evidence remains unchanged.",
            "",
            table(["Accounting measure", "Canonical total"], [[k, v] for k, v in totals.items()]),
            "",
            "## Natural Results",
            "",
            score_table(result, "natural"),
            "",
            resource_table(result, "natural"),
            "",
            "## Matched-Token Results",
            "",
            score_table(result, "matched_tokens"),
            "",
            resource_table(result, "matched_tokens"),
            "",
            "Resource cells show mean / median per example. Attempt latency is the sum of recorded "
            "provider durations; lifecycle includes client work; "
            "wall latency includes orchestration. "
            "One historical failed wall duration is reconstructed. Quality/1k is a ratio of means, "
            "not a recomputation of the frozen task metric. Local marginal API cost is zero; "
            "quality/$ is not meaningful and is omitted.",
            "",
            condition_report(result),
            "",
            "## Task Families",
            "",
            breakdown(result, "task"),
            "",
            "## Difficulty",
            "",
            breakdown(result, "difficulty"),
            "",
            "## Frozen Reasoning Families",
            "",
            "Reasoning labels come from unchanged v3.1 metadata. TINY cells (n<5) cannot support "
            "reliable interpretation; referential ambiguity has only n=1. Pooled family means "
            "can mix task metrics and composition effects; "
            "no post-hoc hard-case superiority claim.",
            "",
            breakdown(result, "reasoning_family"),
            "",
            mechanism_report(result),
            "",
            failure_report(result),
            "",
            dev_report(result),
            "",
            pareto_report(result),
            "",
            "## Limitations",
            "",
            *["- " + x for x in result["limitations"]],
            "",
            "## Reproduction And Publication",
            "",
            "Public aggregate export: docs/heldout_results.json. Private raw protocols, DBs, "
            "TEST data and per-example appendix are excluded from Git. Public clones can inspect "
            "aggregate results but cannot fully reproduce this run without the withheld bundle.",
            "",
            "With the original local evidence, reproduce without model calls:",
            "",
            "```sh",
            "PYTHONPATH=src python3 -m analysis.heldout_final --check",
            "```",
            "",
            "The analysis is outside frozen source globs, preserving protocol code identity. "
            "The aggregate export binds analysis source hashes and the full raw evidence inventory "
            "hash; local integrity_report.json contains the full immutable evidence hash "
            "inventory.",
            "",
        ]
    )


def render_outputs(result: dict[str, Any], rows: dict[str, Any]) -> dict[Path, str]:
    report = full_report(result)
    return {
        FINAL / "heldout_analysis.json": json_text(result),
        FINAL / "heldout_analysis.md": report,
        FINAL / "dev_vs_test.md": dev_report(result),
        FINAL / "mechanism_analysis.md": mechanism_report(result),
        FINAL / "failure_analysis.md": failure_report(result),
        FINAL / "pareto_analysis.md": pareto_report(result),
        FINAL / "representative_cases.md": representative_report(result, rows),
        FINAL / "final_research_summary.md": final_summary(result),
        FINAL / "sakana_application_summary.md": application_report(),
        ROOT / "docs/heldout_results.md": report,
        ROOT / "docs/heldout_results.json": json_text(public_view(result)),
    }
