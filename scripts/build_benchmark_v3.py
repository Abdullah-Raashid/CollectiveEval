"""Build CollectiveEval Benchmark v3.1 operation-integrity revision.

Benchmark v3.1 preserves the v3 benchmark design while repairing human review
findings around reasoning-family and generator-operation truthfulness.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from collectiveeval.core import BenchmarkExample, TaskType
from collectiveeval.datasets import (
    BENCHMARK_V3_1_VERSION,
    freeze_benchmark,
    load_jsonl,
    near_duplicate_leakage_report,
    operation_witness_report,
    validate_benchmark_dir,
    validate_examples,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data" / "benchmark_v3_1"
REPORT_DIR = ROOT / "reports"
GENERATOR_VERSION = BENCHMARK_V3_1_VERSION

COMPANIES = [
    "株式会社青空",
    "北斗商事",
    "みなと物流",
    "桜テック",
    "東都食品",
    "西山医療",
    "南港リテール",
    "白河精密",
    "大手町ラボ",
    "緑丘ファーム",
    "光南出版",
    "朝霧ケア",
]
COUNTERPARTIES = [
    "藤田電機",
    "アーバン印刷",
    "光和システム",
    "緑川産業",
    "日本橋データ",
    "東海監査法人",
    "晴海倉庫",
    "青葉リーガル",
]
PEOPLE = ["佐藤", "田中", "鈴木", "高橋", "伊藤", "中村", "小林", "山本", "森", "井上"]
TOPICS = ["監査対応", "配送改善", "クラウド移行", "保守契約", "広告審査", "研修運営"]
SURFACE_FORMS = [
    "prose_paragraph",
    "email_thread",
    "bullet_memo",
    "table_plaintext",
    "amendment_original",
    "policy_with_exceptions",
    "meeting_interleaved",
    "invoice_field_layout",
    "support_ticket_chronology",
    "contract_clause",
    "status_update_superseded",
    "fragmented_multiparagraph",
]
REASONING_FAMILIES = [
    "direct_lookup",
    "same_entity_distractor",
    "paraphrase_normalization",
    "nullable_missing_field",
    "simple_temporal_resolution",
    "conflict_current_version",
    "exception_rule",
    "cross_sentence_composition",
    "insufficient_evidence",
    "numeric_normalization",
    "referential_ambiguity",
    "long_context_same_topic",
]
V3_1_TEST_FAMILY_INDICES = {1, 4, 6, 9, 11}


def split_for_family(family_index: int) -> str:
    return "test" if family_index in V3_1_TEST_FAMILY_INDICES else "dev"


def main() -> None:
    examples = [
        *grounded_qa_examples(),
        *structured_extraction_examples(),
        *summarization_examples(),
        *robustness_examples(),
    ]
    examples.sort(key=lambda example: example.id)
    dev = [example for example in examples if example.metadata["split"] == "dev"]
    test = [example for example in examples if example.metadata["split"] == "test"]

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(BENCHMARK_DIR / "dev.jsonl", dev)
    write_jsonl(BENCHMARK_DIR / "test.jsonl", test)
    validate_benchmark_dir(BENCHMARK_DIR)
    manifest = freeze_benchmark(BENCHMARK_DIR, version=BENCHMARK_V3_1_VERSION)
    frozen_examples = load_jsonl(BENCHMARK_DIR / "dev.jsonl") + load_jsonl(
        BENCHMARK_DIR / "test.jsonl"
    )
    write_benchmark_card(frozen_examples, manifest)
    write_review_sample(dev)
    write_quality_audit(dev, test, frozen_examples)
    write_operation_witness_audit(dev, test, frozen_examples)
    write_v2_defect_report()
    write_archive_notes()


def grounded_qa_examples() -> list[BenchmarkExample]:
    examples = []
    for family_index, surface in enumerate(SURFACE_FORMS):
        split = split_for_family(family_index)
        for case_index in range(10):
            examples.append(grounded_qa_example(family_index, case_index, surface, split))
    return examples


def grounded_qa_example(
    family_index: int,
    case_index: int,
    surface: str,
    split: str,
) -> BenchmarkExample:
    base_id = f"v3_1-gqa-{family_index:02d}-{case_index:02d}"
    company = pick(COMPANIES, family_index, case_index)
    other = pick(COUNTERPARTIES, family_index, case_index)
    issue_date = iso_date(family_index, case_index, 0)
    current_date = iso_date(family_index, case_index, 8)
    days = [7, 10, 14, 21, 30, 45][(family_index + case_index) % 6]
    profile = qa_profile(case_index)
    evidence: list[dict[str, str]]
    question = f"{company}の現在の提出期限は何日前ですか。"
    answer = f"{days}日前"
    gold_ids = [f"{base_id}-main"]
    answerable = True

    if profile.reasoning_family == "direct_lookup":
        evidence = [{"id": f"{base_id}-main", "text": f"{company}の提出期限は{days}日前である。"}]
    elif profile.reasoning_family == "same_entity_distractor":
        evidence = [
            {"id": f"{base_id}-dist", "text": f"{company}の支払期限は{days + 5}日前である。"},
            {"id": f"{base_id}-main", "text": f"{company}の提出期限は{days}日前である。"},
        ]
    elif profile.reasoning_family == "paraphrase_normalization":
        evidence = [
            {"id": f"{base_id}-main", "text": f"{company}では締切の{days}日前までに申請を出す。"},
            {"id": f"{base_id}-dist", "text": f"{other}では提出期限を20日前と呼ぶ。"},
        ]
    elif profile.reasoning_family == "simple_temporal_resolution":
        evidence = [
            {"id": f"{base_id}-old", "text": f"{issue_date}時点の{company}提出期限は5日前だった。"},
            {
                "id": f"{base_id}-main",
                "text": f"{current_date}時点の{company}提出期限は{days}日前である。",
            },
        ]
    elif profile.reasoning_family == "nullable_missing_field":
        evidence = [
            {
                "id": f"{base_id}-main",
                "text": f"{company}の提出期限は{days}日前だが、代理承認者は明記されていない。",
            },
            {
                "id": f"{base_id}-dist",
                "text": f"管理番号GQA-{family_index:02d}-{case_index:02d}は照合用である。",
            },
        ]
        question = f"{company}の代理承認者は誰ですか。"
        answer = "明記されていない"
    elif profile.reasoning_family == "conflict_current_version":
        old_days = days + 7
        evidence = [
            {
                "id": f"{base_id}-old",
                "text": f"旧規程では{company}の提出期限は{old_days}日前である。",
            },
            {"id": f"{base_id}-main", "text": f"改訂版では{company}の提出期限は{days}日前である。"},
        ]
    elif profile.reasoning_family == "exception_rule":
        answer = "7日前"
        evidence = [
            {"id": f"{base_id}-rule", "text": f"{company}の通常提出期限は14日前である。"},
            {"id": f"{base_id}-main", "text": f"{company}の緊急区分では提出期限は7日前である。"},
        ]
        question = f"{company}の緊急区分での提出期限は何日前ですか。"
        gold_ids = [f"{base_id}-main"]
    elif profile.reasoning_family == "cross_sentence_composition":
        answer = "法務部"
        evidence = [
            {"id": f"{base_id}-route", "text": f"{company}の高リスク案件は専門部門が確認する。"},
            {"id": f"{base_id}-main", "text": f"{company}の専門部門は法務部である。"},
        ]
        question = f"{company}の高リスク案件を確認する部門はどこですか。"
        gold_ids = [f"{base_id}-route", f"{base_id}-main"]
    elif profile.reasoning_family == "insufficient_evidence":
        evidence = [
            {
                "id": f"{base_id}-main",
                "text": f"{company}の提出窓口は営業企画部であり、最終承認者は記載されていない。",
            }
        ]
        question = f"{company}の最終承認者は誰ですか。"
        answer = ""
        gold_ids = []
        answerable = False
    elif profile.reasoning_family == "numeric_normalization":
        amount = (family_index + case_index + 9) * 100000
        answer = f"{amount}円"
        evidence = [
            {
                "id": f"{base_id}-main",
                "text": f"{company}の承認上限は{amount // 10000}万円である。",
            },
            {
                "id": f"{base_id}-dist",
                "text": f"{company}の月額費用は{amount // 20000}万円である。",
            },
        ]
        question = f"{company}の承認上限はいくらですか。"
    elif profile.reasoning_family == "referential_ambiguity":
        evidence = [
            {"id": f"{base_id}-case", "text": f"A案件は{days}日前、B案件は10日前に提出する。"},
            {"id": f"{base_id}-main", "text": f"{company}の対象案件はA案件である。"},
        ]
        gold_ids = [f"{base_id}-case", f"{base_id}-main"]
    else:
        filler = " ".join(f"{company}の参考記録{i}は旧案件の情報である。" for i in range(18))
        evidence = [
            {"id": f"{base_id}-dist", "text": filler},
            {"id": f"{base_id}-main", "text": f"{company}の現行提出期限は{days}日前である。"},
        ]

    evidence.append(
        {"id": f"{base_id}-context", "text": f"{company}のv3.1管理番号はGQA-{family_index:02d}。"}
    )
    gold: dict[str, Any] = {"answerable": answerable, "answer": answer, "evidence": gold_ids}
    if answer:
        gold["acceptable_answers"] = acceptable_variants(answer)
    return make_example(
        base_id,
        TaskType.GROUNDED_QA,
        {
            "source_text": render_surface(surface, [unit["text"] for unit in evidence]),
            "question": question,
            "evidence_units": evidence,
        },
        gold,
        "synthetic_grounded_qa_v3_1",
        split,
        profile,
        family_index,
        surface,
        company,
        ["grounded_qa", profile.reasoning_family],
        chronology={
            "ordered_pairs": [
                {"label": "revision_date", "earlier": issue_date, "later": current_date}
            ]
        },
    )


def structured_extraction_examples() -> list[BenchmarkExample]:
    examples = []
    for family_index, surface in enumerate(SURFACE_FORMS):
        split = split_for_family(family_index)
        for case_index in range(10):
            examples.append(extraction_example(family_index, case_index, surface, split))
    return examples


def extraction_example(
    family_index: int,
    case_index: int,
    surface: str,
    split: str,
) -> BenchmarkExample:
    base_id = f"v3_1-ext-{family_index:02d}-{case_index:02d}"
    company = pick(COMPANIES, family_index, case_index)
    counterparty = pick(COUNTERPARTIES, case_index, family_index)
    owner = pick(PEOPLE, family_index, case_index)
    issue_date = iso_date(family_index, case_index, 0)
    due_date = iso_date(family_index, case_index, 24)
    amount = (family_index + case_index + 8) * 100000
    profile = extraction_profile(case_index)
    has_due = profile.reasoning_family not in {"nullable_missing_field", "insufficient_evidence"}
    auto_renewal: bool | None = (family_index + case_index) % 2 == 0
    priority: str | None = ["低", "中", "高", None][(family_index + case_index) % 4]
    document_type = {
        "invoice_field_layout": "請求書",
        "support_ticket_chronology": "サポート票",
        "contract_clause": "契約条項",
    }.get(surface, "業務文書")
    body = [
        f"文書種別: {document_type}",
        f"会社: {company}",
        f"取引先: {counterparty}",
        f"発行日: {issue_date}",
        f"金額: {amount:,}円",
        f"担当者: {owner}",
        f"自動更新: {'有効' if auto_renewal else '無効'}",
        f"優先度: {priority if priority else '未設定'}",
    ]
    if has_due:
        body.append(f"対応期限: {due_date}")
    else:
        body.append("対応期限は本文に明記されていない。")

    if profile.reasoning_family == "same_entity_distractor":
        body.insert(4, f"参考: {company}の旧見積金額は{amount + 40000:,}円である。")
    elif profile.reasoning_family == "conflict_current_version":
        old_due_date = iso_date(family_index, case_index, 12)
        body.insert(
            0,
            f"旧版では{company}の金額は{amount - 50000:,}円、期限は{old_due_date}だった。",
        )
        body.append(f"最新版では金額{amount:,}円、対応期限{due_date}を採用する。")
    elif profile.reasoning_family == "exception_rule":
        body.append(f"通常更新は自動更新有効だが、{company}の今回区分では自動更新は無効。")
        auto_renewal = False
    elif profile.reasoning_family == "numeric_normalization":
        full_width = str(amount).translate(str.maketrans("0123456789", "０１２３４５６７８９"))
        body = [line for line in body if not line.startswith("金額:")]
        body.insert(4, f"金額: {full_width}円")
    elif profile.reasoning_family == "simple_temporal_resolution":
        body.append(
            f"改訂前の期限は{iso_date(family_index, case_index, 11)}、改訂後の期限は{due_date}。"
        )
    elif profile.reasoning_family == "long_context_same_topic":
        body.extend(f"{company}の過去案件{i}は今回の抽出対象外。" for i in range(14))
    elif profile.reasoning_family == "referential_ambiguity":
        body.insert(
            1, f"A案件は{company}、B案件は{pick(COMPANIES, case_index, family_index)}を指す。"
        )
        body.append("本文中の「前者」が今回の抽出対象である。")

    expected = {
        "document_type": document_type,
        "company": company,
        "counterparty": counterparty,
        "amount_jpy": amount,
        "issue_date": issue_date,
        "due_date": due_date if has_due else None,
        "owner": owner,
        "auto_renewal": auto_renewal,
        "priority": priority,
    }
    return make_example(
        base_id,
        TaskType.STRUCTURED_EXTRACTION,
        {"text": render_surface(surface, body)},
        {
            "expected": expected,
            "json_schema": extraction_schema(),
            "schema_required": ["document_type", "company", "amount_jpy", "issue_date"],
        },
        "synthetic_structured_extraction_v3_1",
        split,
        profile,
        family_index,
        surface,
        company,
        ["structured_extraction", "date", "currency", profile.reasoning_family],
        document_type=document_type,
        chronology={
            "ordered_pairs": [{"label": "issue_due", "earlier": issue_date, "later": due_date}]
            if has_due
            else []
        },
    )


def summarization_examples() -> list[BenchmarkExample]:
    examples = []
    for family_index, surface in enumerate(SURFACE_FORMS):
        split = split_for_family(family_index)
        for case_index in range(10):
            examples.append(summarization_example(family_index, case_index, surface, split))
    return examples


def summarization_example(
    family_index: int,
    case_index: int,
    surface: str,
    split: str,
) -> BenchmarkExample:
    base_id = f"v3_1-sum-{family_index:02d}-{case_index:02d}"
    company = pick(COMPANIES, family_index, case_index)
    owner = pick(PEOPLE, family_index, case_index)
    reviewer = pick(PEOPLE, family_index + 3, case_index)
    topic = pick(TOPICS, family_index, case_index)
    document_date = iso_date(family_index, case_index, 0)
    deadline = iso_date(family_index, case_index, 18)
    reviewer_deadline = iso_date(family_index, case_index, 26)
    budget = (family_index + case_index + 6) * 500000
    profile = summarization_profile(case_index)
    secondary_has_deadline = profile.reasoning_family != "nullable_missing_field"
    decision = f"{company}は{topic}を{budget:,}円以内で進める"
    risk = f"{topic}の関係者確認が遅れる可能性"
    action_items = [
        {"owner": owner, "action": "実行計画を更新する", "deadline": deadline},
        {
            "owner": reviewer,
            "action": "監査観点を確認する",
            "deadline": reviewer_deadline if secondary_has_deadline else None,
        },
    ]
    lines = [
        f"v3.1レビュー管理番号: {base_id} / surface={surface} / family={family_index}",
        f"作成日: {document_date}",
        f"議題: {company}の{topic}",
        f"決定事項: {decision}。",
        f"{owner}は{deadline}までに実行計画を更新する。",
        (
            f"{reviewer}は{reviewer_deadline}までに監査観点を確認する。"
            if secondary_has_deadline
            else f"{reviewer}は監査観点を確認する。"
        ),
        f"リスク: {risk}。",
    ]
    if profile.reasoning_family == "conflict_current_version":
        lines.insert(2, f"前回案では上限を{budget + 300000:,}円としていた。")
        lines.append(f"今回の決定では{budget:,}円を現行上限とする。")
    elif profile.reasoning_family == "same_entity_distractor":
        lines.append(f"参考: {company}の別件{topic}レビューは今回の判断対象外。")
    elif profile.reasoning_family == "exception_rule":
        lines.append(f"通常は{reviewer}の確認は翌月でよい。")
        lines.append(f"ただし緊急監査の場合は{reviewer}の確認を優先する。")
    elif profile.reasoning_family == "cross_sentence_composition":
        lines.insert(3, f"{company}の高リスク論点は法務確認を要する。")
    elif profile.reasoning_family == "long_context_same_topic":
        lines.extend(
            f"過去会議{i}: {company}の{topic}旧論点であり今回の決定対象外。" for i in range(12)
        )
    supported_facts: list[Any] = [decision, risk]
    for action in action_items:
        supported_facts.extend([action["owner"], action["action"]])
        if action["deadline"] is not None:
            supported_facts.append(action["deadline"])
    gold = {
        "summary": f"{company}は{topic}を予算内で進める。",
        "decisions": [decision],
        "action_items": action_items,
        "risks": [risk],
        "supported_facts": supported_facts,
        "json_schema": summary_schema(),
    }
    return make_example(
        base_id,
        TaskType.BUSINESS_SUMMARIZATION,
        {"text": render_surface(surface, lines)},
        gold,
        "synthetic_business_summarization_v3_1",
        split,
        profile,
        family_index,
        surface,
        company,
        ["business_summarization", "action_items", "risks", profile.reasoning_family],
        document_date=document_date,
        chronology={
            "ordered_pairs": [
                {"label": "document_action_deadline", "earlier": document_date, "later": deadline}
            ]
        },
    )


def robustness_examples() -> list[BenchmarkExample]:
    examples = []
    for family_index, surface in enumerate(SURFACE_FORMS):
        split = split_for_family(family_index)
        for case_index in range(10):
            examples.append(robustness_example(family_index, case_index, surface, split))
    return examples


def robustness_example(
    family_index: int,
    case_index: int,
    surface: str,
    split: str,
) -> BenchmarkExample:
    base_id = f"v3_1-rob-{family_index:02d}-{case_index:02d}"
    company = pick(COMPANIES, family_index, case_index)
    days = [5, 7, 14, 30, 45][(family_index + case_index) % 5]
    profile = robustness_profile(family_index, case_index)
    question = f"{company}の承認依頼は何日前までに提出しますか。"
    answer = f"{days}日前"
    answerable = True
    evidence = [
        {"id": f"{base_id}-main", "text": f"{company}の承認依頼は{days}日前までに提出する。"}
    ]
    gold_ids = [f"{base_id}-main"]
    extra_tags = [profile.reasoning_family]

    if profile.reasoning_family == "same_entity_distractor":
        evidence.append(
            {"id": f"{base_id}-dist", "text": f"{company}の支払猶予は{days + 3}日である。"}
        )
    elif profile.reasoning_family == "paraphrase_normalization":
        question = f"{company}のSLA reviewはいつまでに開始しますか。"
        answer = f"{days} business days前"
        evidence[0]["text"] = f"{company}のSLA reviewは{days} business days前までに開始する。"
        extra_tags.append("mixed_japanese_english")
    elif profile.reasoning_family == "nullable_missing_field":
        question = f"{company}の代替承認ルートはどれですか。"
        answer = "明記されていない"
        evidence = [
            {
                "id": f"{base_id}-main",
                "text": (
                    f"{company}の通常承認依頼は{days}日前までに提出するが、"
                    "代替承認ルートは明記されていない。"
                ),
            }
        ]
    elif profile.reasoning_family == "conflict_current_version":
        old_days = max(1, days - 2)
        evidence = [
            {
                "id": f"{base_id}-old",
                "text": f"旧手順では{company}の承認依頼は{old_days}日前までに提出する。",
            },
            {
                "id": f"{base_id}-main",
                "text": f"最新版では{company}の承認依頼は{days}日前までに提出する。",
            },
        ]
    elif profile.reasoning_family == "exception_rule":
        answer = "7日前"
        evidence = [
            {"id": f"{base_id}-rule", "text": f"{company}の通常カテゴリは14日前までに提出する。"},
            {"id": f"{base_id}-main", "text": f"{company}の緊急カテゴリは7日前までに提出する。"},
        ]
        question = f"{company}の緊急カテゴリでは承認依頼を何日前までに提出しますか。"
    elif profile.reasoning_family == "cross_sentence_composition":
        answer = "2026-04-01"
        evidence = [
            {
                "id": f"{base_id}-rule",
                "text": f"{company}の新料金は改訂施行日から適用する。",
            },
            {
                "id": f"{base_id}-date",
                "text": f"{company}の改訂施行日は令和8年4月1日（2026-04-01）である。",
            },
        ]
        question = f"{company}の新料金の適用日はいつですか。"
        gold_ids = [f"{base_id}-rule", f"{base_id}-date"]
        extra_tags.append("japanese_era_date")
    elif profile.reasoning_family == "insufficient_evidence":
        evidence = [
            {
                "id": f"{base_id}-main",
                "text": f"{company}の申請窓口は総務部であり、承認依頼期限は記載されていない。",
            }
        ]
        question = f"{company}の承認依頼期限は何日前ですか。"
        answer = ""
        answerable = False
        gold_ids = []
    elif profile.reasoning_family == "numeric_normalization":
        full_width = str(days).translate(str.maketrans("0123456789", "０１２３４５６７８９"))
        question = f"{company}の支払猶予は何日ですか。"
        answer = f"{days}日"
        evidence[0]["text"] = f"{company}の支払猶予は全角表記で{full_width}日と記載する。"
        extra_tags.append("full_width_numeral")
    elif profile.reasoning_family == "referential_ambiguity":
        evidence = [
            {
                "id": f"{base_id}-case",
                "text": f"A案件は{days}日前、B案件は10日前に承認依頼を提出する。",
            },
            {"id": f"{base_id}-main", "text": f"{company}の今回対象は前者、つまりA案件である。"},
        ]
        gold_ids = [f"{base_id}-case", f"{base_id}-main"]
    elif profile.reasoning_family == "long_context_same_topic":
        filler = " ".join(f"{company}の旧申請記録{i}は対象外。" for i in range(24))
        evidence = [
            {"id": f"{base_id}-dist", "text": filler},
            {
                "id": f"{base_id}-main",
                "text": f"{company}の現行承認依頼は{days}日前までに提出する。",
            },
        ]
    elif profile.reasoning_family == "direct_lookup" and case_index == 1:
        evidence[0]["text"] = (
            f"{company}では即日申請を認めないわけではないが、通常は{days}日前提出を求める。"
        )
        extra_tags.append("nested_negation")

    evidence.append(
        {
            "id": f"{base_id}-context",
            "text": (
                f"{company}のv3.1管理番号はROB-{family_index:02d}-{surface}。"
                f"この管理行は{base_id}専用で、回答根拠ではない。"
            ),
        }
    )
    gold: dict[str, Any] = {"answerable": answerable, "answer": answer, "evidence": gold_ids}
    if answer:
        gold["acceptable_answers"] = acceptable_variants(answer)
    return make_example(
        base_id,
        TaskType.ROBUSTNESS,
        {
            "source_text": render_surface(surface, [unit["text"] for unit in evidence]),
            "question": question,
            "evidence_units": evidence,
        },
        gold,
        "synthetic_robustness_v3_1",
        split,
        profile,
        family_index,
        surface,
        company,
        ["robustness", *extra_tags],
    )


class DifficultyProfile:
    def __init__(
        self,
        difficulty: str,
        reasoning_family: str,
        operations: list[str],
        factors: list[str],
    ) -> None:
        self.difficulty = difficulty
        self.reasoning_family = reasoning_family
        self.operations = operations
        self.factors = factors


def qa_profile(case_index: int) -> DifficultyProfile:
    profiles = [
        DifficultyProfile(
            "easy",
            "direct_lookup",
            ["direct_lookup", "one_relevant_fact", "minimal_noise"],
            ["direct_lookup", "one_relevant_fact"],
        ),
        DifficultyProfile(
            "easy",
            "direct_lookup",
            ["direct_lookup", "one_relevant_fact", "minimal_noise"],
            ["direct_lookup", "one_relevant_fact"],
        ),
        DifficultyProfile(
            "easy",
            "direct_lookup",
            ["direct_lookup", "one_relevant_fact", "minimal_noise"],
            ["direct_lookup", "minimal_noise"],
        ),
        DifficultyProfile(
            "medium",
            "same_entity_distractor",
            ["same_entity_distractor", "relevant_distractor_same_entity"],
            ["same_entity_distractor"],
        ),
        DifficultyProfile(
            "medium",
            "paraphrase_normalization",
            ["paraphrase_normalization", "normalization"],
            ["paraphrase_normalization"],
        ),
        DifficultyProfile(
            "medium",
            "nullable_missing_field",
            ["nullable_missing_field", "missing_field_handling"],
            ["nullable_missing_field"],
        ),
        DifficultyProfile(
            "hard",
            "conflict_current_version",
            ["conflict_resolution_same_entity", "current_version_resolution"],
            ["conflict_resolution_same_entity", "current_version_resolution"],
        ),
        DifficultyProfile(
            "hard",
            "exception_rule",
            ["exception_rule_resolution", "conditional_logic"],
            ["exception_rule_resolution", "conditional_logic"],
        ),
        DifficultyProfile(
            "hard",
            "cross_sentence_composition",
            ["cross_sentence_composition", "multi_constraint_resolution"],
            ["cross_sentence_composition", "multi_constraint_resolution"],
        ),
        DifficultyProfile(
            "hard",
            "insufficient_evidence",
            ["insufficient_evidence_abstention"],
            ["insufficient_evidence_abstention"],
        ),
    ]
    return profiles[case_index]


def extraction_profile(case_index: int) -> DifficultyProfile:
    profiles = [
        DifficultyProfile(
            "easy",
            "direct_lookup",
            ["direct_lookup", "one_relevant_fact", "minimal_noise"],
            ["direct_lookup", "one_relevant_fact"],
        ),
        DifficultyProfile(
            "easy",
            "direct_lookup",
            ["direct_lookup", "one_relevant_fact", "minimal_noise"],
            ["direct_lookup", "one_relevant_fact"],
        ),
        DifficultyProfile(
            "easy",
            "direct_lookup",
            ["direct_lookup", "one_relevant_fact", "minimal_noise"],
            ["direct_lookup", "minimal_noise"],
        ),
        DifficultyProfile(
            "medium",
            "same_entity_distractor",
            ["same_entity_distractor", "relevant_distractor_same_entity"],
            ["same_entity_distractor"],
        ),
        DifficultyProfile(
            "medium",
            "nullable_missing_field",
            ["nullable_missing_field", "missing_field_handling"],
            ["nullable_missing_field"],
        ),
        DifficultyProfile(
            "medium",
            "same_entity_distractor",
            ["same_entity_distractor", "relevant_distractor_same_entity"],
            ["same_entity_distractor"],
        ),
        DifficultyProfile(
            "hard",
            "conflict_current_version",
            ["conflict_resolution_same_entity", "current_version_resolution"],
            ["conflict_resolution_same_entity", "current_version_resolution"],
        ),
        DifficultyProfile(
            "hard",
            "exception_rule",
            ["exception_rule_resolution", "conditional_logic"],
            ["exception_rule_resolution", "conditional_logic"],
        ),
        DifficultyProfile(
            "hard",
            "numeric_normalization",
            ["arithmetic_or_normalization"],
            ["arithmetic_or_normalization"],
        ),
        DifficultyProfile(
            "hard",
            "long_context_same_topic",
            ["long_context_same_entity_retrieval"],
            ["long_context_same_entity_retrieval"],
        ),
    ]
    return profiles[case_index]


def summarization_profile(case_index: int) -> DifficultyProfile:
    profiles = [
        DifficultyProfile(
            "easy",
            "direct_lookup",
            ["direct_lookup", "one_relevant_fact", "minimal_noise"],
            ["direct_lookup", "one_relevant_fact"],
        ),
        DifficultyProfile(
            "easy",
            "direct_lookup",
            ["direct_lookup", "one_relevant_fact", "minimal_noise"],
            ["direct_lookup", "one_relevant_fact"],
        ),
        DifficultyProfile(
            "easy",
            "direct_lookup",
            ["direct_lookup", "one_relevant_fact", "minimal_noise"],
            ["direct_lookup", "minimal_noise"],
        ),
        DifficultyProfile(
            "medium",
            "same_entity_distractor",
            ["same_entity_distractor", "relevant_distractor_same_entity"],
            ["same_entity_distractor"],
        ),
        DifficultyProfile(
            "medium",
            "nullable_missing_field",
            ["nullable_missing_field", "missing_field_handling"],
            ["nullable_missing_field"],
        ),
        DifficultyProfile(
            "medium",
            "same_entity_distractor",
            ["same_entity_distractor", "relevant_distractor_same_entity"],
            ["same_entity_distractor"],
        ),
        DifficultyProfile(
            "hard",
            "conflict_current_version",
            ["conflict_resolution_same_entity", "current_version_resolution"],
            ["conflict_resolution_same_entity", "current_version_resolution"],
        ),
        DifficultyProfile(
            "hard",
            "exception_rule",
            ["exception_rule_resolution", "conditional_logic"],
            ["exception_rule_resolution", "conditional_logic"],
        ),
        DifficultyProfile(
            "hard",
            "long_context_same_topic",
            ["long_context_same_entity_retrieval"],
            ["long_context_same_entity_retrieval"],
        ),
        DifficultyProfile(
            "hard",
            "long_context_same_topic",
            ["long_context_same_entity_retrieval"],
            ["long_context_same_entity_retrieval"],
        ),
    ]
    return profiles[case_index]


def robustness_profile(family_index: int, case_index: int) -> DifficultyProfile:
    if case_index != 8:
        return qa_profile(case_index)
    variants = [
        DifficultyProfile(
            "hard",
            "cross_sentence_composition",
            [
                "cross_sentence_composition",
                "multi_constraint_resolution",
                "japanese_era_date_conversion",
            ],
            ["cross_sentence_composition", "multi_constraint_resolution"],
        ),
        DifficultyProfile(
            "hard",
            "numeric_normalization",
            ["arithmetic_or_normalization"],
            ["arithmetic_or_normalization"],
        ),
        DifficultyProfile(
            "hard",
            "referential_ambiguity",
            ["referential_ambiguity_resolution", "multi_constraint_resolution"],
            ["referential_ambiguity_resolution"],
        ),
        DifficultyProfile(
            "hard",
            "long_context_same_topic",
            ["long_context_same_entity_retrieval"],
            ["long_context_same_entity_retrieval"],
        ),
    ]
    return variants[family_index % len(variants)]


def make_example(
    example_id: str,
    task_type: TaskType,
    input_payload: dict[str, Any],
    gold: dict[str, Any],
    source: str,
    split: str,
    profile: DifficultyProfile,
    family_index: int,
    surface: str,
    target_entity: str,
    tags: list[str],
    *,
    document_type: str | None = None,
    document_date: str | None = None,
    chronology: dict[str, Any] | None = None,
) -> BenchmarkExample:
    return BenchmarkExample(
        id=example_id,
        task_type=task_type,
        input=input_payload,
        gold=gold,
        metadata={
            "source": source,
            "difficulty": profile.difficulty,
            "difficulty_factors": profile.factors,
            "tags": canonical_tags(tags),
            "split": split,
            "template_family": (
                f"{task_type.value}_{surface}_{profile.reasoning_family}_{family_index:02d}"
            ),
            "surface_form_family": surface,
            "reasoning_family": profile.reasoning_family,
            "scenario_family": f"{task_type.value}_scenario_{family_index:02d}",
            "generator_version": GENERATOR_VERSION,
            "generator_operations": profile.operations,
            "split_policy": "heldout_template_family",
            "challenge": profile.difficulty == "hard",
            "provenance": "synthetic_business_style_v3_1",
            "structural_signature": f"{surface}:{profile.reasoning_family}",
            "target_entity": target_entity,
            "document_type": document_type,
            "document_date": document_date,
            "chronology": chronology or {"ordered_pairs": []},
        },
    )


def render_surface(surface: str, lines: list[str]) -> str:
    if surface == "email_thread":
        return "\n".join(f"From: reviewer{i}@example.jp\n{line}" for i, line in enumerate(lines, 1))
    if surface == "bullet_memo":
        return "\n".join(f"- {line}" for line in lines)
    if surface == "table_plaintext":
        return "\n".join(f"| row{i} | {line} |" for i, line in enumerate(lines, 1))
    if surface == "amendment_original":
        return "原文:\n" + "\n".join(lines[:1]) + "\n改訂:\n" + "\n".join(lines[1:])
    if surface == "policy_with_exceptions":
        return "規程本文\n" + "\n".join(lines) + "\n例外条項は本文に優先する。"
    if surface == "meeting_interleaved":
        return "\n".join(f"発言{i}: {line}" for i, line in enumerate(lines, 1))
    if surface == "invoice_field_layout":
        return "\n".join(f"{line}" for line in lines)
    if surface == "support_ticket_chronology":
        return "\n".join(f"T+{i}: {line}" for i, line in enumerate(lines, 1))
    if surface == "contract_clause":
        return "\n".join(f"第{i}条 {line}" for i, line in enumerate(lines, 1))
    if surface == "status_update_superseded":
        return "\n".join(f"ステータス{i}: {line}" for i, line in enumerate(lines, 1))
    if surface == "fragmented_multiparagraph":
        return "\n\n".join(lines)
    return " ".join(lines)


def extraction_schema() -> dict[str, Any]:
    return {
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


def summary_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["summary", "decisions", "action_items", "risks"],
        "properties": {
            "summary": {"type": "string"},
            "decisions": {"type": "array", "items": {"type": "string"}},
            "action_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["owner", "action", "deadline"],
                    "properties": {
                        "owner": {"type": "string"},
                        "action": {"type": "string"},
                        "deadline": {"type": ["string", "null"], "format": "date"},
                    },
                },
            },
            "risks": {"type": "array", "items": {"type": "string"}},
        },
    }


def write_benchmark_card(examples: list[BenchmarkExample], manifest: dict[str, Any]) -> None:
    counts = count_examples(examples)
    text = f"""# CollectiveEval Benchmark v3.1 Card

Benchmark v3.1 is a surgical integrity patch over v3. It archives v1/v2/v3
for scientific use and repairs operation-witness metadata before Phase 8.

## Counts

- Total examples: {len(examples)}
- Dev examples: {manifest["files"]["dev.jsonl"]["examples"]}
- Test examples: {manifest["files"]["test.jsonl"]["examples"]}
- By task: {json.dumps(counts["task_type"], ensure_ascii=False, sort_keys=True)}
- By difficulty: {json.dumps(counts["difficulty"], ensure_ascii=False, sort_keys=True)}
- Surface forms: {json.dumps(counts["surface_form_family"], ensure_ascii=False, sort_keys=True)}
- Reasoning families: {json.dumps(counts["reasoning_family"], ensure_ascii=False, sort_keys=True)}

## Hashes

- dev.jsonl: `{manifest["files"]["dev.jsonl"]["sha256"]}`
- test.jsonl: `{manifest["files"]["test.jsonl"]["sha256"]}`
- manifest payload: `{manifest["manifest_payload_sha256"]}`
- manifest file: `{manifest["manifest_file_sha256"]}`

## Held-Out Policy

The test split is blind for Phase 8 and later. Do not use it for prompt
iteration, router training, ablation design, or manual review samples.
"""
    (REPORT_DIR / "benchmark_v3_1_card.md").write_text(text, encoding="utf-8")


def write_review_sample(dev_examples: list[BenchmarkExample]) -> None:
    selected: dict[str, BenchmarkExample] = {}
    for bucket_key in ("reasoning_family", "surface_form_family"):
        buckets: dict[str, list[BenchmarkExample]] = defaultdict(list)
        for example in dev_examples:
            buckets[str(example.metadata[bucket_key])].append(example)
        for examples in buckets.values():
            hard = [example for example in examples if example.metadata["difficulty"] == "hard"]
            pool = hard or examples
            selected[sorted(pool, key=lambda item: item.id)[0].id] = sorted(
                pool, key=lambda item: item.id
            )[0]
    for example in sorted(
        dev_examples, key=lambda item: (item.metadata["difficulty"] != "hard", item.id)
    ):
        selected[example.id] = example
        if len(selected) >= 48:
            break
    lines = [
        "# Benchmark v3.1 Manual Review Sample",
        "",
        "DEV ONLY. This sample emphasizes hard cases plus every reasoning and surface-form family.",
        "",
    ]
    for example in list(selected.values())[:48]:
        lines.extend(
            [
                f"## {example.id}",
                "",
                f"- Task: `{example.task_type}`",
                f"- Split: `{example.metadata['split']}`",
                f"- Difficulty: `{example.metadata['difficulty']}`",
                f"- Surface form: `{example.metadata['surface_form_family']}`",
                f"- Reasoning family: `{example.metadata['reasoning_family']}`",
                f"- Generator operations: {', '.join(example.metadata['generator_operations'])}",
                f"- Tags: {', '.join(example.metadata['tags'])}",
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
    (REPORT_DIR / "benchmark_v3_1_review_sample.md").write_text("\n".join(lines), encoding="utf-8")


def write_quality_audit(
    dev: list[BenchmarkExample],
    test: list[BenchmarkExample],
    examples: list[BenchmarkExample],
) -> None:
    validation_issues = validate_examples(examples)
    leakage = near_duplicate_leakage_report(dev, test, threshold=0.97)
    counts = count_examples(examples)
    chronology_issues = [
        issue for issue in validation_issues if "date" in issue or "chronology" in issue
    ]
    factor_issues = [
        issue for issue in validation_issues if "difficulty" in issue or "hard example" in issue
    ]
    tag_issues = [issue for issue in validation_issues if "tag" in issue]
    source_gold_issues = [
        issue
        for issue in validation_issues
        if "unsupported" in issue or "mismatch" in issue or "supported_facts" in issue
    ]
    grouped_split_overlap = leakage["template_family_overlap"] or leakage["scenario_family_overlap"]
    audit = {
        "chronology_audit": chronology_issues or "clean",
        "difficulty_factor_audit": factor_issues or "clean",
        "structural_diversity_counts": {
            "surface_form_family": counts["surface_form_family"],
            "reasoning_family": counts["reasoning_family"],
            "template_families": counts["template_family"],
        },
        "tag_audit": tag_issues or "clean",
        "source_gold_consistency": source_gold_issues or "clean",
        "grouped_split_audit": grouped_split_overlap or "clean",
        "near_duplicate_audit": leakage["high_similarity_pairs"] or "clean",
    }
    lines = [
        "# Benchmark v3.1 Quality Audit",
        "",
        "```json",
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
    ]
    (REPORT_DIR / "benchmark_v3_1_quality_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_operation_witness_audit(
    dev: list[BenchmarkExample],
    test: list[BenchmarkExample],
    examples: list[BenchmarkExample],
) -> None:
    counts = count_examples(examples)
    report = operation_witness_report(examples)
    payload = {
        **report,
        "dev_examples": len(dev),
        "test_examples": len(test),
        "regenerated": {
            "scope": "all v3.1 examples were freshly generated from patched rules",
            "examples": len(examples),
        },
        "relabeled": {
            "scope": "generator profiles were corrected so labels match rendered witnesses",
            "examples": "see violations_by_operation from pre-patch v3 audit in final report",
        },
        "final_counts": {
            "difficulty": counts["difficulty"],
            "reasoning_family": counts["reasoning_family"],
            "surface_form_family": counts["surface_form_family"],
        },
        "dev_only_review_sample": True,
    }
    lines = [
        "# Benchmark v3.1 Operation Witness Audit",
        "",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
    ]
    (REPORT_DIR / "benchmark_v3_1_operation_witness_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_v2_defect_report() -> None:
    v2_dir = ROOT / "data" / "benchmark_v2"
    if not v2_dir.exists():
        return
    examples = load_jsonl(v2_dir / "dev.jsonl") + load_jsonl(v2_dir / "test.jsonl")
    defects = audit_v2_defects(examples)
    lines = [
        "# Benchmark v2 Defects Found During v3 Review",
        "",
        "```json",
        json.dumps(defects, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
    ]
    (REPORT_DIR / "benchmark_v3_v2_defects_found.md").write_text("\n".join(lines), encoding="utf-8")


def audit_v2_defects(examples: list[BenchmarkExample]) -> dict[str, Any]:
    duplicate_tags = []
    hard_without_hard_operation = []
    hard_without_operation_trace = []
    robustness_multi_constraint_without_multiple_evidence = []
    chronology = []
    unsupported_summary_facts = []
    nominal_templates = []
    missing_v3_structural_metadata = []
    for example in examples:
        tags = example.metadata.get("tags", [])
        if isinstance(tags, list) and len(tags) != len(set(tags)):
            duplicate_tags.append(example.id)
        if example.metadata.get("difficulty") == "hard":
            if "generator_operations" not in example.metadata:
                hard_without_operation_trace.append(example.id)
            factors = set(example.metadata.get("difficulty_factors", []))
            if not factors.intersection(
                {"conflict_or_negation", "cross_sentence_or_temporal_reasoning"}
            ):
                hard_without_hard_operation.append(example.id)
        if (
            example.task_type == TaskType.ROBUSTNESS
            and "multi_constraint" in tags
            and len(example.gold.get("evidence", [])) <= 1
        ):
            robustness_multi_constraint_without_multiple_evidence.append(example.id)
        if example.task_type == TaskType.STRUCTURED_EXTRACTION:
            expected = example.gold.get("expected", {})
            if isinstance(expected, dict):
                issue = parse_date(expected.get("issue_date"))
                due = parse_date(expected.get("due_date"))
                if issue and due and due < issue:
                    chronology.append(example.id)
        if example.task_type == TaskType.BUSINESS_SUMMARIZATION:
            supported = {str(item) for item in example.gold.get("supported_facts", [])}
            for action in example.gold.get("action_items", []):
                if isinstance(action, dict):
                    required = [action.get("owner"), action.get("action"), action.get("deadline")]
                    if any(item is not None and str(item) not in supported for item in required):
                        unsupported_summary_facts.append(example.id)
                        break
        if example.metadata.get("template_family") == example.metadata.get("document_type"):
            nominal_templates.append(example.id)
        if not {
            "surface_form_family",
            "reasoning_family",
            "generator_operations",
        }.issubset(example.metadata):
            missing_v3_structural_metadata.append(example.id)
    return {
        "due_date_before_issue_date": {"count": len(chronology), "example_ids": chronology[:20]},
        "duplicate_tags": {"count": len(duplicate_tags), "example_ids": duplicate_tags[:20]},
        "hard_difficulty_without_operation_trace": {
            "count": len(hard_without_operation_trace),
            "example_ids": hard_without_operation_trace[:20],
        },
        "hard_without_verified_hard_operation": {
            "count": len(hard_without_hard_operation),
            "example_ids": hard_without_hard_operation[:20],
        },
        "robustness_multi_constraint_without_multiple_gold_evidence": {
            "count": len(robustness_multi_constraint_without_multiple_evidence),
            "example_ids": robustness_multi_constraint_without_multiple_evidence[:20],
        },
        "supported_facts_not_exhaustive_for_summary_actions": {
            "count": len(unsupported_summary_facts),
            "example_ids": unsupported_summary_facts[:20],
        },
        "missing_v3_structural_metadata": {
            "count": len(missing_v3_structural_metadata),
            "example_ids": missing_v3_structural_metadata[:20],
        },
        "template_family_nominal_only": {
            "count": len(nominal_templates),
            "example_ids": nominal_templates[:20],
        },
    }


def write_archive_notes() -> None:
    for version in ("benchmark_v1", "benchmark_v2", "benchmark_v3"):
        path = ROOT / "data" / version / "ARCHIVED.md"
        path.write_text(
            (
                f"# {version} Archived\n\n"
                "This benchmark version is retained for reproducibility only. "
                "Do not use it for v3.1 scientific claims, router training, "
                "or Phase 8 real-model pilots.\n"
            ),
            encoding="utf-8",
        )


def count_examples(examples: list[BenchmarkExample]) -> dict[str, dict[str, int]]:
    fields = (
        "task_type",
        "difficulty",
        "split",
        "surface_form_family",
        "reasoning_family",
        "template_family",
    )
    counts: dict[str, dict[str, int]] = {}
    for field in fields:
        if field == "task_type":
            counter = Counter(str(example.task_type) for example in examples)
        else:
            counter = Counter(str(example.metadata.get(field)) for example in examples)
        counts[field] = dict(sorted(counter.items()))
    return counts


def canonical_tags(tags: list[str]) -> list[str]:
    canonical = []
    for tag in tags:
        normalized = "".join(
            character if character.isalnum() else "_" for character in tag.lower()
        ).strip("_")
        if normalized and normalized not in canonical:
            canonical.append(normalized)
    return canonical


def acceptable_variants(answer: str) -> list[str]:
    variants = [answer]
    if "日前" in answer:
        variants.append(answer.replace("日前", "日"))
    if answer.endswith("円") and answer[:-1].isdigit():
        variants.append(f"{int(answer[:-1]) // 10000}万円")
    if answer == "2026-04-01":
        variants.append("令和8年4月1日")
    return sorted(set(variants))


def iso_date(seed_a: int, seed_b: int, offset_days: int) -> str:
    base = date(2026, 1, 10) + timedelta(days=seed_a * 19 + seed_b * 3 + offset_days)
    return base.isoformat()


def parse_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def pick(values: list[str], seed_a: int, seed_b: int) -> str:
    return values[(seed_a * 3 + seed_b) % len(values)]


if __name__ == "__main__":
    main()
