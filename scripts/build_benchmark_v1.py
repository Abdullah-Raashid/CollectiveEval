"""Build the frozen synthetic CollectiveEval benchmark v1 artifacts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from collectiveeval.core import BenchmarkExample, TaskType
from collectiveeval.datasets import BENCHMARK_VERSION, freeze_benchmark, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data" / "benchmark_v1"
REPORT_DIR = ROOT / "reports"

COMPANIES = ["株式会社青空", "北斗商事", "みなと物流", "桜テック", "東都食品", "西山医療"]
COUNTERPARTIES = ["藤田電機", "アーバン印刷", "光和システム", "緑川産業", "日本橋データ"]
PEOPLE = ["佐藤", "田中", "鈴木", "高橋", "伊藤", "中村"]
PRODUCTS = ["保守契約", "広告出稿", "クラウド移行", "什器購入", "研修運営", "監査対応"]
DIFFICULTIES = ["easy", "medium", "hard"]


def main() -> None:
    examples = [
        *[grounded_qa_example(index) for index in range(120)],
        *[structured_extraction_example(index) for index in range(120)],
        *[summarization_example(index) for index in range(120)],
        *[robustness_example(index) for index in range(120)],
    ]
    dev = [example for example in examples if example.metadata["split"] == "dev"]
    test = [example for example in examples if example.metadata["split"] == "test"]

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(BENCHMARK_DIR / "dev.jsonl", dev)
    write_jsonl(BENCHMARK_DIR / "test.jsonl", test)
    manifest = freeze_benchmark(BENCHMARK_DIR, version=BENCHMARK_VERSION)
    write_benchmark_card(examples, manifest)
    write_review_sample(examples)


def split_for(index: int) -> str:
    return "test" if index % 5 in {3, 4} else "dev"


def metadata(
    *,
    source: str,
    index: int,
    tags: list[str],
    difficulty: str | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "difficulty": difficulty or DIFFICULTIES[index % len(DIFFICULTIES)],
        "tags": tags,
        "split": split_for(index),
        "provenance": "synthetic_business_style_v1",
    }


def grounded_qa_example(index: int) -> BenchmarkExample:
    company = COMPANIES[index % len(COMPANIES)]
    counterparty = COUNTERPARTIES[index % len(COUNTERPARTIES)]
    days = [7, 10, 14, 30, 45, 60][index % 6]
    amount = (index % 9 + 1) * 110000
    month = index % 12 + 1
    day = index % 24 + 1
    date = f"2026-{month:02d}-{day:02d}"
    pattern = index % 8
    base_id = f"v1-gqa-{index:03d}"

    if pattern == 0:
        evidence = [
            {
                "id": f"{base_id}-a",
                "text": f"{company}の解約通知は{days}日前までに書面で提出する。",
            },
            {
                "id": f"{base_id}-b",
                "text": f"{counterparty}の更新通知は10日前までにメールで提出する。",
            },
        ]
        question = f"{company}の解約通知は何日前までに必要ですか。"
        answer, gold_ids, answerable = f"{days}日前", [f"{base_id}-a"], True
        tags = ["grounded_qa", "numeric_fact", "distractor_passage"]
    elif pattern == 1:
        evidence = [
            {"id": f"{base_id}-a", "text": f"{company}の支払期日は{date}である。"},
            {"id": f"{base_id}-b", "text": f"{company}の請求番号はINV-{index:04d}である。"},
        ]
        question = f"{company}の支払期日はいつですか。"
        answer, gold_ids, answerable = date, [f"{base_id}-a"], True
        tags = ["grounded_qa", "date", "distractor_passage"]
    elif pattern == 2:
        evidence = [
            {"id": f"{base_id}-a", "text": f"{company}は追加費用{amount:,}円を承認した。"},
            {"id": f"{base_id}-b", "text": "交通費は本件の追加費用に含めない。"},
        ]
        question = "承認された追加費用はいくらですか。"
        answer, gold_ids, answerable = f"{amount}円", [f"{base_id}-a"], True
        tags = ["grounded_qa", "currency", "negation"]
    elif pattern == 3:
        evidence = [
            {"id": f"{base_id}-a", "text": f"{company}の旧規程では申請期限は10日前だった。"},
            {
                "id": f"{base_id}-b",
                "text": f"2026年版の新規程では申請期限は{days}日前に変更された。",
            },
        ]
        question = "最新の規程では申請期限は何日前ですか。"
        answer, gold_ids, answerable = f"{days}日前", [f"{base_id}-b"], True
        tags = ["grounded_qa", "conflicting_evidence", "date"]
    elif pattern == 4:
        evidence = [
            {"id": f"{base_id}-a", "text": f"{company}は{counterparty}と覚書を締結した。"},
            {"id": f"{base_id}-b", "text": "覚書には違約金の金額は記載されていない。"},
        ]
        question = "違約金はいくらですか。"
        answer, gold_ids, answerable = "", [], False
        tags = ["grounded_qa", "insufficient_evidence", "unanswerable"]
    elif pattern == 5:
        evidence = [
            {"id": f"{base_id}-a", "text": f"{company}の自動更新は有効ではない。"},
            {"id": f"{base_id}-b", "text": "ただし手動更新は双方の書面合意により可能である。"},
        ]
        question = f"{company}の契約は自動更新されますか。"
        answer, gold_ids, answerable = "自動更新されない", [f"{base_id}-a"], True
        tags = ["grounded_qa", "negation", "boolean_condition"]
    elif pattern == 6:
        long_context = " ".join(
            [
                f"参考{i}: {COUNTERPARTIES[(index + i) % len(COUNTERPARTIES)]}の一般条項は対象外。"
                for i in range(12)
            ]
        )
        evidence = [
            {"id": f"{base_id}-a", "text": long_context},
            {
                "id": f"{base_id}-b",
                "text": f"対象案件では{company}の検収期限は納品後{days}営業日である。",
            },
        ]
        question = "対象案件の検収期限は納品後何営業日ですか。"
        answer, gold_ids, answerable = f"{days}営業日", [f"{base_id}-b"], True
        tags = ["grounded_qa", "long_context", "numeric_fact"]
    else:
        evidence = [
            {"id": f"{base_id}-a", "text": f"{company}の担当部署は営業企画部である。"},
            {"id": f"{base_id}-b", "text": "承認者名は資料からは確認できない。"},
        ]
        question = "承認者は誰ですか。"
        answer, gold_ids, answerable = "", [], False
        tags = ["grounded_qa", "insufficient_evidence", "unanswerable"]

    input_payload = {
        "source_text": "\n".join(unit["text"] for unit in evidence),
        "question": question,
        "evidence_units": evidence,
    }
    gold = {"answerable": answerable, "answer": answer, "evidence": gold_ids}
    if answer:
        gold["acceptable_answers"] = [answer.replace("日前", "日")]
    return BenchmarkExample(
        id=base_id,
        task_type=TaskType.GROUNDED_QA,
        input=input_payload,
        gold=gold,
        metadata=metadata(
            source="synthetic_grounded_qa_template_v1",
            index=index,
            tags=tags,
        ),
    )


def structured_extraction_example(index: int) -> BenchmarkExample:
    kind = index % 6
    company = COMPANIES[index % len(COMPANIES)]
    counterparty = COUNTERPARTIES[(index + 1) % len(COUNTERPARTIES)]
    owner = PEOPLE[index % len(PEOPLE)]
    amount = (index % 13 + 2) * 100000
    date = f"2026-{index % 12 + 1:02d}-{index % 24 + 1:02d}"
    due = f"2026-{(index + 1) % 12 + 1:02d}-{(index + 7) % 24 + 1:02d}"
    schema = {
        "type": "object",
        "required": ["document_type", "company", "amount_jpy", "issue_date"],
        "additionalProperties": False,
        "properties": {
            "document_type": {"type": "string"},
            "company": {"type": "string"},
            "counterparty": {"type": ["string", "null"]},
            "amount_jpy": {"type": "integer"},
            "issue_date": {"type": "string", "format": "date"},
            "due_date": {"type": ["string", "null"], "format": "date"},
            "owner": {"type": ["string", "null"]},
            "auto_renewal": {"type": ["boolean", "null"]},
            "priority": {"type": ["string", "null"]},
        },
    }
    labels = ["契約書", "請求書", "規程通知", "案件更新", "発注書", "サポート票"]
    auto_renewal: bool | None = None if kind in {1, 3, 4, 5} else kind == 0
    priority = ["低", "中", "高", None][index % 4]
    expected = {
        "document_type": labels[kind],
        "company": company,
        "counterparty": counterparty,
        "amount_jpy": amount,
        "issue_date": date,
        "due_date": due if kind != 2 else None,
        "owner": owner,
        "auto_renewal": auto_renewal,
        "priority": priority,
    }
    text = (
        f"{labels[kind]}: {company}は{counterparty}向けに{date}付で"
        f"{amount:,}円の{PRODUCTS[index % len(PRODUCTS)]}を登録した。"
        f"支払または対応期限は{due}、担当者は{owner}。"
        f"自動更新は{'有効' if auto_renewal else '無効' if auto_renewal is False else '対象外'}。"
        f"優先度は{priority or '未設定'}。"
    )
    return BenchmarkExample(
        id=f"v1-ext-{index:03d}",
        task_type=TaskType.STRUCTURED_EXTRACTION,
        input={"text": text},
        gold={
            "expected": expected,
            "json_schema": schema,
            "schema_required": schema["required"],
        },
        metadata=metadata(
            source="synthetic_structured_extraction_template_v1",
            index=index,
            tags=["structured_extraction", labels[kind], "date", "currency", "boolean"],
        ),
    )


def summarization_example(index: int) -> BenchmarkExample:
    company = COMPANIES[index % len(COMPANIES)]
    owner = PEOPLE[index % len(PEOPLE)]
    reviewer = PEOPLE[(index + 2) % len(PEOPLE)]
    deadline = f"2026-{(index + 2) % 12 + 1:02d}-{(index + 10) % 24 + 1:02d}"
    budget = (index % 8 + 3) * 500000
    decision = f"{company}は第2フェーズを{budget:,}円以内で進める"
    risk = "旧システムの権限棚卸しが遅れる可能性"
    text = (
        f"会議メモ: {company}のクラウド移行について議論した。"
        f"決定事項は「{decision}」。{owner}は{deadline}までに移行計画を更新する。"
        f"{reviewer}は監査観点の確認を担当する。リスクは{risk}。"
        "営業部からの別件要望は今回の判断対象外とした。"
    )
    gold = {
        "summary": f"{company}はクラウド移行の第2フェーズを予算内で進める。",
        "decisions": [decision],
        "action_items": [
            {"owner": owner, "action": "移行計画を更新する", "deadline": deadline},
            {"owner": reviewer, "action": "監査観点を確認する", "deadline": deadline},
        ],
        "risks": [risk],
        "supported_facts": [company, "第2フェーズ", str(budget), owner, deadline, risk],
    }
    return BenchmarkExample(
        id=f"v1-sum-{index:03d}",
        task_type=TaskType.BUSINESS_SUMMARIZATION,
        input={"text": text},
        gold=gold,
        metadata=metadata(
            source="synthetic_business_summarization_template_v1",
            index=index,
            tags=["business_summarization", "meeting_minutes", "action_items", "risks"],
        ),
    )


def robustness_example(index: int) -> BenchmarkExample:
    phenomenon = [
        "negation",
        "distractor",
        "mixed_language",
        "japanese_era",
        "gregorian_date",
        "full_width_numeral",
        "conflicting_statements",
        "insufficient_evidence",
        "long_context",
        "ambiguous_reference",
        "technical_business_term",
        "keigo_variation",
    ][index % 12]
    base_id = f"v1-rob-{index:03d}"
    company = COMPANIES[index % len(COMPANIES)]
    days = [5, 7, 14, 30, 45][index % 5]
    answerable = phenomenon != "insufficient_evidence"
    answer = f"{days}日前" if answerable else ""
    evidence = [
        {"id": f"{base_id}-main", "text": f"{company}の承認依頼は{days}日前までに提出します。"},
        {"id": f"{base_id}-noise", "text": "この段落は別会社の旧手順であり、対象外です。"},
    ]
    question = "承認依頼は何日前までに提出しますか。"

    if phenomenon == "negation":
        evidence[0]["text"] = f"{company}では即日申請は認めず、{days}日前までの申請のみ受理する。"
    elif phenomenon == "mixed_language":
        evidence[0]["text"] = f"{company}のSLA reviewは{days} business days前までに開始する。"
        answer = f"{days} business days前"
    elif phenomenon == "japanese_era":
        evidence[0]["text"] = "令和8年4月1日に新料金を適用する。"
        question = "新料金の適用日はいつですか。"
        answer = "2026-04-01"
    elif phenomenon == "gregorian_date":
        evidence[0]["text"] = "棚卸しの締切は2026-09-30である。"
        question = "棚卸しの締切日はいつですか。"
        answer = "2026-09-30"
    elif phenomenon == "full_width_numeral":
        full_width_days = str(days).translate(str.maketrans("0123456789", "０１２３４５６７８９"))
        evidence[0]["text"] = f"支払猶予は全角表記で{full_width_days}日と記載する。"
        question = "支払猶予は何日ですか。"
        answer = f"{days}日"
    elif phenomenon == "conflicting_statements":
        evidence = [
            {"id": f"{base_id}-old", "text": "旧手順では5日前までに提出する。"},
            {"id": f"{base_id}-main", "text": f"最新版では{days}日前までに提出する。"},
        ]
    elif phenomenon == "insufficient_evidence":
        evidence = [{"id": f"{base_id}-main", "text": f"{company}の申請窓口は総務部です。"}]
        question = "承認依頼は何日前までに提出しますか。"
    elif phenomenon == "long_context":
        filler = " ".join([f"参考情報{i}は対象外です。" for i in range(30)])
        evidence[0]["text"] = f"{filler} 対象規程では承認依頼は{days}日前までに提出する。"
    elif phenomenon == "ambiguous_reference":
        evidence[0]["text"] = (
            f"A案件は{days}日前、B案件は10日前に申請する。"
            "本問の対象はA案件である。"
        )
    elif phenomenon == "technical_business_term":
        evidence[0]["text"] = f"{company}のDPAレビューは{days}日前までに法務へ起票する。"
    elif phenomenon == "keigo_variation":
        evidence[0]["text"] = f"恐れ入りますが、承認依頼は{days}日前までにご提出ください。"

    return BenchmarkExample(
        id=base_id,
        task_type=TaskType.ROBUSTNESS,
        input={
            "source_text": "\n".join(unit["text"] for unit in evidence),
            "question": question,
            "evidence_units": evidence,
        },
        gold={
            "answerable": answerable,
            "answer": answer,
            "evidence": [f"{base_id}-main"] if answerable else [],
            "acceptable_answers": [answer.replace("日前", "日")] if answer else [],
        },
        metadata=metadata(
            source="synthetic_robustness_template_v1",
            index=index,
            tags=["robustness", phenomenon],
        ),
    )


def write_benchmark_card(examples: list[BenchmarkExample], manifest: dict[str, Any]) -> None:
    counts = count_examples(examples)
    text = f"""# CollectiveEval Benchmark v1 Card

## Purpose

CollectiveEval Benchmark v1 is a frozen synthetic Japanese business benchmark for
testing budget-aware inference strategies across grounded QA, structured
extraction, summarization, and robustness/reasoning.

## Task Definitions

- Grounded QA: answer questions from cited evidence units or abstain.
- Structured extraction: extract schema-bound business fields from Japanese text.
- Business summarization: recover factual decisions, actions, deadlines, and risks.
- Robustness/reasoning: controlled QA examples targeting Japanese language and
  business-text failure modes.

## Construction Methodology

The benchmark is generated from deterministic, hand-designed synthetic templates.
Templates vary companies, dates, amounts, people, document type, distractors,
negation, conflicts, insufficient evidence, and business terminology. Examples
are validated through the package schema before freezing.

## Source And Provenance

All examples are synthetic business-style documents. They are not confidential
records and should not be described as real enterprise data.

## Counts

- Total examples: {len(examples)}
- Dev examples: {manifest["files"]["dev.jsonl"]["examples"]}
- Test examples: {manifest["files"]["test.jsonl"]["examples"]}
- By task: {json.dumps(counts["task_type"], ensure_ascii=False, sort_keys=True)}
- By difficulty: {json.dumps(counts["difficulty"], ensure_ascii=False, sort_keys=True)}

## Hashes

- dev.jsonl: `{manifest["files"]["dev.jsonl"]["sha256"]}`
- test.jsonl: `{manifest["files"]["test.jsonl"]["sha256"]}`
- manifest payload: `{manifest["manifest_payload_sha256"]}`
- manifest file: `{manifest["manifest_file_sha256"]}`

## Limitations

The benchmark is synthetic and template-generated, so it cannot measure all
distributional properties of real deployments. It is useful for controlled
correctness, accounting, and evaluator regression, not for broad product claims.

## Contamination Risks

The dataset lives in-repository and may be visible to agents during development.
The test split is held out for evaluation and must not be used for tuning,
router training, prompt iteration, or ablation selection.

## Intended Use

Use this benchmark to compare inference strategies under matched per-example
token/call/cost/latency budgets and to test evaluator behavior.

## Inappropriate Use

Do not claim performance on real enterprise documents, confidential data, or
general Japanese reasoning from this benchmark alone.

## Held-Out Evaluation Policy

The `test.jsonl` split is held out. Training and tuning code, including learned
router fitting, must reject test examples.
"""
    (REPORT_DIR / "benchmark_card.md").write_text(text, encoding="utf-8")


def write_review_sample(examples: list[BenchmarkExample]) -> None:
    buckets: dict[tuple[str, str], list[BenchmarkExample]] = defaultdict(list)
    for example in examples:
        buckets[(str(example.task_type), str(example.metadata["difficulty"]))].append(example)
    selected = [items[0] for _, items in sorted(buckets.items()) if items]
    tag_seen: set[str] = set()
    for example in examples:
        for tag in example.metadata.get("tags", []):
            if tag not in tag_seen:
                selected.append(example)
                tag_seen.add(str(tag))
            if len(selected) >= 32:
                break
        if len(selected) >= 32:
            break
    unique: dict[str, BenchmarkExample] = {example.id: example for example in selected}
    lines = [
        "# Benchmark v1 Manual Review Sample",
        "",
        "This deterministic sample is for human inspection only. It is not an approval record.",
        "",
    ]
    for example in unique.values():
        lines.extend(
            [
                f"## {example.id}",
                "",
                f"- Task: `{example.task_type}`",
                f"- Split: `{example.metadata['split']}`",
                f"- Difficulty: `{example.metadata['difficulty']}`",
                f"- Tags: {', '.join(example.metadata['tags'])}",
                f"- Source: `{example.metadata['source']}`",
                "",
                "Input:",
                "```json",
                json.dumps(example.input, ensure_ascii=False, indent=2, sort_keys=True),
                "```",
                "",
                "Gold:",
                "```json",
                json.dumps(example.gold, ensure_ascii=False, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    (REPORT_DIR / "benchmark_review_sample.md").write_text("\n".join(lines), encoding="utf-8")


def count_examples(examples: list[BenchmarkExample]) -> dict[str, dict[str, int]]:
    return {
        "task_type": dict(sorted(Counter(str(example.task_type) for example in examples).items())),
        "difficulty": dict(
            sorted(Counter(str(example.metadata["difficulty"]) for example in examples).items())
        ),
        "split": dict(
            sorted(Counter(str(example.metadata["split"]) for example in examples).items())
        ),
    }


if __name__ == "__main__":
    main()
