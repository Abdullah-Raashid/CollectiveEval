"""Build CollectiveEval Benchmark v2 with stricter validity controls."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from collectiveeval.core import BenchmarkExample, TaskType
from collectiveeval.datasets import (
    BENCHMARK_V2_VERSION,
    freeze_benchmark,
    load_jsonl,
    near_duplicate_leakage_report,
    validate_benchmark_dir,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data" / "benchmark_v2"
REPORT_DIR = ROOT / "reports"
GENERATOR_VERSION = BENCHMARK_V2_VERSION

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
PRODUCTS = [
    "保守契約",
    "広告出稿",
    "クラウド移行",
    "什器購入",
    "研修運営",
    "監査対応",
    "データ移行",
    "配送改善",
]
PRIORITIES = ["低", "中", "高", None]


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
    manifest = freeze_benchmark(BENCHMARK_DIR, version=BENCHMARK_V2_VERSION)
    all_examples = load_jsonl(BENCHMARK_DIR / "dev.jsonl") + load_jsonl(
        BENCHMARK_DIR / "test.jsonl"
    )
    write_benchmark_card(all_examples, manifest)
    write_review_sample(dev)
    write_leakage_report(dev, test)
    write_defect_repair_log(all_examples)


def grounded_qa_examples() -> list[BenchmarkExample]:
    examples = []
    families = [
        "gqa_notice_deadline",
        "gqa_payment_due",
        "gqa_approved_cost",
        "gqa_latest_policy",
        "gqa_penalty_absent",
        "gqa_auto_renewal",
        "gqa_acceptance_long_context",
        "gqa_approver_missing",
        "gqa_sla_mixed_language",
        "gqa_temporal_version",
    ]
    for family_index, family in enumerate(families):
        split = "dev" if family_index < 6 else "test"
        for case_index in range(12):
            examples.append(grounded_qa_example(family, family_index, case_index, split))
    return examples


def grounded_qa_example(
    family: str,
    family_index: int,
    case_index: int,
    split: str,
) -> BenchmarkExample:
    base_id = f"v2-gqa-{family_index:02d}-{case_index:02d}"
    company = COMPANIES[(family_index + case_index) % len(COMPANIES)]
    other = COUNTERPARTIES[(family_index * 2 + case_index) % len(COUNTERPARTIES)]
    days = [5, 7, 10, 14, 30, 45, 60][(family_index + case_index) % 7]
    amount = (case_index + family_index + 3) * 100000
    issue_date = date_for(family_index, case_index)
    difficulty, factors = difficulty_for(case_index)
    evidence: list[dict[str, str]]
    question: str
    answer: str
    gold_ids: list[str]
    answerable = True
    tags = ["grounded_qa", family]

    if family == "gqa_notice_deadline":
        evidence = [
            {
                "id": f"{base_id}-main",
                "text": f"{company}の解約通知は{days}日前までに書面で提出する。",
            },
            {
                "id": f"{base_id}-dist",
                "text": f"{other}の更新通知は20日前までにメールで提出する。",
            },
        ]
        question = f"{company}の解約通知は何日前までに必要ですか。"
        answer, gold_ids = f"{days}日前", [f"{base_id}-main"]
        tags.append("numeric_fact")
    elif family == "gqa_payment_due":
        evidence = [
            {"id": f"{base_id}-main", "text": f"{company}の支払期日は{issue_date}である。"},
            {
                "id": f"{base_id}-dist",
                "text": f"請求番号はINV-{family_index}{case_index:03d}である。",
            },
        ]
        question = f"{company}の支払期日はいつですか。"
        answer, gold_ids = issue_date, [f"{base_id}-main"]
        tags.append("date")
    elif family == "gqa_approved_cost":
        evidence = [
            {"id": f"{base_id}-main", "text": f"{company}は追加費用{amount:,}円を承認した。"},
            {"id": f"{base_id}-dist", "text": "交通費は本件の追加費用に含めない。"},
        ]
        question = f"{company}で承認された追加費用はいくらですか。"
        answer, gold_ids = f"{amount}円", [f"{base_id}-main"]
        tags.extend(["currency", "negation"])
    elif family == "gqa_latest_policy":
        old_days = max(3, days - 2)
        evidence = [
            {
                "id": f"{base_id}-old",
                "text": f"旧規程では{company}の申請期限は{old_days}日前だった。",
            },
            {
                "id": f"{base_id}-main",
                "text": f"2026年版では{company}の申請期限は{days}日前に変更された。",
            },
        ]
        question = f"{company}の最新規程での申請期限は何日前ですか。"
        answer, gold_ids = f"{days}日前", [f"{base_id}-main"]
        tags.extend(["conflicting_evidence", "temporal_version"])
    elif family == "gqa_penalty_absent":
        evidence = [
            {
                "id": f"{base_id}-main",
                "text": f"{company}と{other}の覚書には違約金額は記載されていない。",
            },
            {"id": f"{base_id}-dist", "text": "支払条件は別紙Aを参照する。"},
        ]
        question = f"{company}と{other}の覚書の違約金はいくらですか。"
        answer, gold_ids, answerable = "", [], False
        tags.extend(["insufficient_evidence", "unanswerable"])
    elif family == "gqa_auto_renewal":
        evidence = [
            {"id": f"{base_id}-main", "text": f"{company}の自動更新は有効ではない。"},
            {"id": f"{base_id}-dist", "text": "手動更新は双方の書面合意により可能である。"},
        ]
        question = f"{company}の契約は自動更新されますか。"
        answer, gold_ids = "自動更新されない", [f"{base_id}-main"]
        tags.extend(["negation", "boolean_condition"])
    elif family == "gqa_acceptance_long_context":
        filler = " ".join(
            f"参考{i}: {COUNTERPARTIES[(case_index + i) % len(COUNTERPARTIES)]}は対象外。"
            for i in range(16)
        )
        evidence = [
            {"id": f"{base_id}-dist", "text": filler},
            {
                "id": f"{base_id}-main",
                "text": f"対象案件では{company}の検収期限は納品後{days}営業日である。",
            },
        ]
        question = f"対象案件の{company}の検収期限は納品後何営業日ですか。"
        answer, gold_ids = f"{days}営業日", [f"{base_id}-main"]
        tags.extend(["long_context", "numeric_fact"])
    elif family == "gqa_approver_missing":
        evidence = [
            {"id": f"{base_id}-main", "text": f"{company}の担当部署は営業企画部である。"},
            {"id": f"{base_id}-missing", "text": "承認者名は資料からは確認できない。"},
        ]
        question = f"{company}の承認者は誰ですか。"
        answer, gold_ids, answerable = "", [], False
        tags.extend(["insufficient_evidence", "unanswerable"])
    elif family == "gqa_sla_mixed_language":
        evidence = [
            {
                "id": f"{base_id}-main",
                "text": f"{company}のSLA reviewは{days} business days前までに開始する。",
            },
            {"id": f"{base_id}-dist", "text": "契約更新レビューは10日前に開始する。"},
        ]
        question = f"{company}のSLA reviewはいつまでに開始しますか。"
        answer, gold_ids = f"{days} business days前", [f"{base_id}-main"]
        tags.extend(["mixed_japanese_english", "business_term"])
    else:
        previous = date_for(family_index, max(0, case_index - 1))
        evidence = [
            {
                "id": f"{base_id}-old",
                "text": f"旧資料では{company}の開始日は{previous}とされていた。",
            },
            {"id": f"{base_id}-main", "text": f"最新版では{company}の開始日は{issue_date}である。"},
        ]
        question = f"{company}の最新版での開始日はいつですか。"
        answer, gold_ids = issue_date, [f"{base_id}-main"]
        tags.extend(["temporal_version", "date"])

    if difficulty == "hard":
        evidence.append(
            {
                "id": f"{base_id}-hard-dist",
                "text": f"なお、{other}の別案件では{max(1, days - 1)}日前という記録がある。",
            }
        )
    gold: dict[str, Any] = {"answerable": answerable, "answer": answer, "evidence": gold_ids}
    if answer:
        gold["acceptable_answers"] = acceptable_variants(answer)
        if family == "gqa_auto_renewal":
            gold["acceptable_answers"].append("有効ではない")
    return BenchmarkExample(
        id=base_id,
        task_type=TaskType.GROUNDED_QA,
        input={
            "source_text": "\n".join(unit["text"] for unit in evidence),
            "question": question,
            "evidence_units": evidence,
        },
        gold=gold,
        metadata=metadata(
            source="synthetic_grounded_qa_v2",
            template_family=family,
            scenario_family=f"{family}_scenario_{family_index}",
            split=split,
            difficulty=difficulty,
            difficulty_factors=factors,
            tags=tags,
            challenge=bool({"hard", "unanswerable", "temporal_version"} & {difficulty, *tags}),
        ),
    )


def structured_extraction_examples() -> list[BenchmarkExample]:
    examples = []
    families = [
        "ext_contract_notice",
        "ext_invoice_payment",
        "ext_policy_notification",
        "ext_project_update",
        "ext_purchase_order",
        "ext_support_ticket",
        "ext_nda_record",
        "ext_incident_report",
        "ext_renewal_memo",
        "ext_procurement_request",
    ]
    for family_index, family in enumerate(families):
        split = "dev" if family_index < 6 else "test"
        for case_index in range(12):
            examples.append(extraction_example(family, family_index, case_index, split))
    return examples


def extraction_example(
    family: str,
    family_index: int,
    case_index: int,
    split: str,
) -> BenchmarkExample:
    base_id = f"v2-ext-{family_index:02d}-{case_index:02d}"
    company = COMPANIES[(family_index + case_index) % len(COMPANIES)]
    counterparty = COUNTERPARTIES[(family_index + case_index * 2) % len(COUNTERPARTIES)]
    owner = PEOPLE[(family_index + case_index) % len(PEOPLE)]
    amount = (family_index + case_index + 4) * 100000
    issue_date = date_for(family_index, case_index)
    due_date = date_for(family_index + 1, case_index + 6)
    priority = PRIORITIES[(family_index + case_index) % len(PRIORITIES)]
    auto_renewal = auto_value(family_index + case_index)
    has_due = (family_index + case_index) % 4 != 2
    document_type = {
        "ext_contract_notice": "契約通知",
        "ext_invoice_payment": "請求書",
        "ext_policy_notification": "規程通知",
        "ext_project_update": "案件更新",
        "ext_purchase_order": "発注書",
        "ext_support_ticket": "サポート票",
        "ext_nda_record": "NDA管理票",
        "ext_incident_report": "障害報告",
        "ext_renewal_memo": "更新メモ",
        "ext_procurement_request": "購買申請",
    }[family]
    difficulty, factors = difficulty_for(case_index)
    due_sentence = (
        f"支払または対応期限は{due_date}。"
        if has_due
        else "支払または対応期限は本文に明記されていない。"
    )
    priority_sentence = f"優先度は{priority}。" if priority else "優先度は未設定。"
    auto_sentence = {
        True: "自動更新は有効。",
        False: "自動更新は無効。",
        None: "自動更新は対象外。",
    }[auto_renewal]
    text = (
        f"{document_type}: {company}は{counterparty}向けに{issue_date}付で"
        f"{amount:,}円の{PRODUCTS[(family_index + case_index) % len(PRODUCTS)]}を登録した。"
        f"{due_sentence}担当者は{owner}。{auto_sentence}{priority_sentence}"
    )
    if difficulty == "medium":
        text += f"参考: {COUNTERPARTIES[(case_index + 3) % len(COUNTERPARTIES)]}の旧案件は対象外。"
    if difficulty == "hard":
        text = (
            f"旧メモでは{company}の金額を{amount - 50000:,}円としていたが、"
            f"最新版の{document_type}では{amount:,}円に修正された。"
            + text
        )
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
    return BenchmarkExample(
        id=base_id,
        task_type=TaskType.STRUCTURED_EXTRACTION,
        input={"text": text},
        gold={
            "expected": expected,
            "json_schema": extraction_schema(),
            "schema_required": ["document_type", "company", "amount_jpy", "issue_date"],
        },
        metadata=metadata(
            source="synthetic_structured_extraction_v2",
            template_family=family,
            scenario_family=f"{family}_scenario_{family_index}",
            split=split,
            difficulty=difficulty,
            difficulty_factors=factors,
            tags=["structured_extraction", document_type, "date", "currency", "boolean"],
            challenge=bool(difficulty == "hard" or not has_due),
        ),
    )


def summarization_examples() -> list[BenchmarkExample]:
    examples = []
    families = [
        "sum_cloud_migration",
        "sum_security_review",
        "sum_vendor_negotiation",
        "sum_incident_response",
        "sum_launch_planning",
        "sum_audit_preparation",
        "sum_training_program",
        "sum_procurement",
        "sum_data_retention",
        "sum_office_move",
    ]
    for family_index, family in enumerate(families):
        split = "dev" if family_index < 6 else "test"
        for case_index in range(12):
            examples.append(summarization_example(family, family_index, case_index, split))
    return examples


def summarization_example(
    family: str,
    family_index: int,
    case_index: int,
    split: str,
) -> BenchmarkExample:
    base_id = f"v2-sum-{family_index:02d}-{case_index:02d}"
    company = COMPANIES[(family_index + case_index) % len(COMPANIES)]
    owner = PEOPLE[(family_index + case_index) % len(PEOPLE)]
    reviewer = PEOPLE[(family_index + case_index + 3) % len(PEOPLE)]
    deadline = date_for(family_index + 2, case_index + 4)
    secondary_deadline = date_for(family_index + 3, case_index + 5)
    budget = (family_index + case_index + 5) * 500000
    topic = {
        "sum_cloud_migration": "クラウド移行",
        "sum_security_review": "セキュリティレビュー",
        "sum_vendor_negotiation": "委託先交渉",
        "sum_incident_response": "障害対応",
        "sum_launch_planning": "新サービス公開",
        "sum_audit_preparation": "監査準備",
        "sum_training_program": "研修計画",
        "sum_procurement": "購買計画",
        "sum_data_retention": "データ保持方針",
        "sum_office_move": "拠点移転",
    }[family]
    difficulty, factors = difficulty_for(case_index)
    secondary_has_deadline = case_index % 4 == 0
    decision = f"{company}は{topic}を{budget:,}円以内で進める"
    risk = f"{topic}の関係者確認が遅れる可能性"
    secondary_sentence = (
        f"{reviewer}は{secondary_deadline}までに監査観点を確認する。"
        if secondary_has_deadline
        else f"{reviewer}は監査観点の確認を担当する。"
    )
    text = (
        f"会議メモ: {company}の{topic}について議論した。"
        f"決定事項は「{decision}」。"
        f"{owner}は{deadline}までに実行計画を更新する。"
        f"{secondary_sentence}"
        f"リスクは{risk}。"
    )
    if difficulty == "medium":
        text += "営業部からの別件要望は今回の判断対象外とした。"
    if difficulty == "hard":
        text = (
            f"前回案では上限を{budget + 300000:,}円としていたが、今回の決定で修正した。"
            + text
            + "旧日程の記載は参考情報であり採用しない。"
        )
    action_items = [
        {"owner": owner, "action": "実行計画を更新する", "deadline": deadline},
        {
            "owner": reviewer,
            "action": "監査観点を確認する",
            "deadline": secondary_deadline if secondary_has_deadline else None,
        },
    ]
    gold = {
        "summary": f"{company}は{topic}を予算内で進める。",
        "decisions": [decision],
        "action_items": action_items,
        "risks": [risk],
        "supported_facts": [company, topic, str(budget), owner, deadline, risk],
        "json_schema": summary_schema(),
    }
    return BenchmarkExample(
        id=base_id,
        task_type=TaskType.BUSINESS_SUMMARIZATION,
        input={"text": text},
        gold=gold,
        metadata=metadata(
            source="synthetic_business_summarization_v2",
            template_family=family,
            scenario_family=f"{family}_scenario_{family_index}",
            split=split,
            difficulty=difficulty,
            difficulty_factors=factors,
            tags=["business_summarization", family, "action_items", "risks"],
            challenge=bool(difficulty == "hard" or not secondary_has_deadline),
        ),
    )


def robustness_examples() -> list[BenchmarkExample]:
    examples = []
    phenomena = [
        "rob_negation",
        "rob_distractor",
        "rob_mixed_language",
        "rob_japanese_era",
        "rob_gregorian_date",
        "rob_full_width_numeral",
        "rob_conflicting_statements",
        "rob_insufficient_evidence",
        "rob_long_context",
        "rob_ambiguous_reference",
        "rob_technical_term",
        "rob_keigo_variation",
    ]
    for family_index, family in enumerate(phenomena):
        split = "dev" if family_index < 7 else "test"
        for case_index in range(10):
            examples.append(robustness_example(family, family_index, case_index, split))
    return examples


def robustness_example(
    family: str,
    family_index: int,
    case_index: int,
    split: str,
) -> BenchmarkExample:
    base_id = f"v2-rob-{family_index:02d}-{case_index:02d}"
    company = COMPANIES[(family_index + case_index) % len(COMPANIES)]
    days = [5, 7, 14, 30, 45][(family_index + case_index) % 5]
    difficulty, factors = difficulty_for(case_index)
    answerable = family != "rob_insufficient_evidence"
    question = f"{company}の承認依頼は何日前までに提出しますか。"
    answer = f"{days}日前" if answerable else ""
    evidence = [
        {"id": f"{base_id}-main", "text": f"{company}の承認依頼は{days}日前までに提出する。"},
        {"id": f"{base_id}-noise", "text": "この段落は別会社の旧手順であり、対象外です。"},
    ]
    tags = ["robustness", family.removeprefix("rob_")]
    if family == "rob_negation":
        evidence[0]["text"] = f"{company}では即日申請は認めず、{days}日前までの申請のみ受理する。"
        tags.append("negation")
    elif family == "rob_distractor":
        evidence[1]["text"] = f"参考: 別会社の申請期限は{max(1, days - 2)}日前である。"
    elif family == "rob_mixed_language":
        question = f"{company}のSLA reviewはいつまでに開始しますか。"
        answer = f"{days} business days前"
        evidence[0]["text"] = f"{company}のSLA reviewは{days} business days前までに開始する。"
    elif family == "rob_japanese_era":
        question = f"{company}の新料金の適用日はいつですか。"
        answer = "2026-04-01"
        evidence[0]["text"] = f"{company}では令和8年4月1日に新料金を適用する。"
    elif family == "rob_gregorian_date":
        question = f"{company}の棚卸し締切日はいつですか。"
        answer = "2026-09-30"
        evidence[0]["text"] = f"{company}の棚卸し締切は2026-09-30である。"
    elif family == "rob_full_width_numeral":
        full_width = str(days).translate(str.maketrans("0123456789", "０１２３４５６７８９"))
        question = f"{company}の支払猶予は何日ですか。"
        answer = f"{days}日"
        evidence[0]["text"] = f"{company}の支払猶予は全角表記で{full_width}日と記載する。"
    elif family == "rob_conflicting_statements":
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
        tags.append("conflicting_evidence")
    elif family == "rob_insufficient_evidence":
        evidence = [{"id": f"{base_id}-main", "text": f"{company}の申請窓口は総務部である。"}]
        answer = ""
        tags.extend(["insufficient_evidence", "unanswerable"])
    elif family == "rob_long_context":
        filler = " ".join(f"参考情報{i}は対象外です。" for i in range(24))
        evidence[0]["text"] = (
            f"{filler} 対象規程では{company}の承認依頼は{days}日前までに提出する。"
        )
        tags.append("long_context")
    elif family == "rob_ambiguous_reference":
        evidence[0]["text"] = (
            f"A案件は{days}日前、B案件は10日前に申請する。{company}の対象はA案件である。"
        )
        tags.append("ambiguous_reference")
    elif family == "rob_technical_term":
        question = f"{company}のDPAレビューは何日前までに起票しますか。"
        evidence[0]["text"] = f"{company}のDPAレビューは{days}日前までに法務へ起票する。"
        tags.append("technical_business_term")
    elif family == "rob_keigo_variation":
        evidence[0]["text"] = (
            f"恐れ入りますが、{company}の承認依頼は{days}日前までにご提出ください。"
        )
        tags.append("keigo_variation")
    if difficulty == "hard" and answerable:
        tags.append("multi_constraint")
    gold: dict[str, Any] = {
        "answerable": answerable,
        "answer": answer,
        "evidence": [f"{base_id}-main"] if answerable else [],
    }
    if answer:
        gold["acceptable_answers"] = acceptable_variants(answer)
    return BenchmarkExample(
        id=base_id,
        task_type=TaskType.ROBUSTNESS,
        input={
            "source_text": "\n".join(unit["text"] for unit in evidence),
            "question": question,
            "evidence_units": evidence,
        },
        gold=gold,
        metadata=metadata(
            source="synthetic_robustness_v2",
            template_family=family,
            scenario_family=f"{family}_scenario_{family_index}",
            split=split,
            difficulty=difficulty,
            difficulty_factors=factors + [family.removeprefix("rob_")],
            tags=tags,
            challenge=True,
        ),
    )


def metadata(
    *,
    source: str,
    template_family: str,
    scenario_family: str,
    split: str,
    difficulty: str,
    difficulty_factors: list[str],
    tags: list[str],
    challenge: bool,
) -> dict[str, Any]:
    return {
        "source": source,
        "difficulty": difficulty,
        "difficulty_factors": difficulty_factors,
        "tags": tags,
        "split": split,
        "template_family": template_family,
        "scenario_family": scenario_family,
        "generator_version": GENERATOR_VERSION,
        "split_policy": "heldout_template_family",
        "challenge": challenge,
        "provenance": "synthetic_business_style_v2",
    }


def difficulty_for(case_index: int) -> tuple[str, list[str]]:
    mode = case_index % 3
    if mode == 0:
        return "easy", ["one_relevant_fact", "minimal_distractors", "direct_wording"]
    if mode == 1:
        return (
            "medium",
            ["two_or_three_evidence_units", "distractor", "normalization_or_paraphrase"],
        )
    return "hard", [
        "multiple_evidence_units",
        "conflict_or_negation",
        "cross_sentence_or_temporal_reasoning",
    ]


def date_for(seed_a: int, seed_b: int) -> str:
    month = (seed_a + seed_b) % 12 + 1
    day = (seed_a * 3 + seed_b * 5) % 28 + 1
    return f"2026-{month:02d}-{day:02d}"


def acceptable_variants(answer: str) -> list[str]:
    variants = [answer]
    if "日前" in answer:
        variants.append(answer.replace("日前", "日"))
    if "営業日" in answer:
        variants.append(answer.replace("営業日", "営業日前"))
    if answer == "2026-04-01":
        variants.append("令和8年4月1日")
    return sorted(set(variants))


def auto_value(index: int) -> bool | None:
    mode = index % 3
    if mode == 0:
        return True
    if mode == 1:
        return False
    return None


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
    text = f"""# CollectiveEval Benchmark v2 Card

## Purpose

Benchmark v2 replaces archived Benchmark v1 after manual review found validity
defects in template support, deadlines, and grouped split design.

## Construction Methodology

Examples are synthetic Japanese business tasks generated from multiple scenario
families per task. Gold labels are checked against source text, difficulty is
assigned from persisted composition factors, and dev/test is grouped by held-out
template family.

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

## Held-Out Policy

The test split is blind for development: do not use it for prompt iteration,
router training, ablation design, or manual review samples. Template families
marked `heldout_template_family` must not appear in both dev and test.

## Known v1 Repairs

- Unsupported secondary action-item deadlines removed.
- Summarization schemas permit null deadlines.
- Extraction null dates now require source absence.
- Robustness questions and evidence refer to the same event/action/entity.
- Difficulty labels are factor-based rather than index-only.
"""
    (REPORT_DIR / "benchmark_v2_card.md").write_text(text, encoding="utf-8")


def write_review_sample(dev_examples: list[BenchmarkExample]) -> None:
    selected: dict[str, BenchmarkExample] = {}
    buckets: dict[tuple[str, str], list[BenchmarkExample]] = defaultdict(list)
    for example in dev_examples:
        buckets[(str(example.task_type), str(example.metadata["difficulty"]))].append(example)
    for _, examples in sorted(buckets.items()):
        for example in sorted(examples, key=lambda item: item.id)[:2]:
            selected[example.id] = example
    for example in sorted(dev_examples, key=lambda item: item.id):
        if example.metadata.get("challenge") or "unanswerable" in example.metadata.get("tags", []):
            selected[example.id] = example
        if len(selected) >= 48:
            break
    lines = [
        "# Benchmark v2 Manual Review Sample",
        "",
        "DEV ONLY. This deterministic sample is for human inspection only and is not "
        "an approval record.",
        "",
    ]
    for example in selected.values():
        lines.extend(
            [
                f"## {example.id}",
                "",
                f"- Task: `{example.task_type}`",
                f"- Split: `{example.metadata['split']}`",
                f"- Difficulty: `{example.metadata['difficulty']}`",
                f"- Difficulty factors: {', '.join(example.metadata['difficulty_factors'])}",
                f"- Template family: `{example.metadata['template_family']}`",
                f"- Scenario family: `{example.metadata['scenario_family']}`",
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
    (REPORT_DIR / "benchmark_v2_review_sample.md").write_text("\n".join(lines), encoding="utf-8")


def write_leakage_report(
    dev: list[BenchmarkExample],
    test: list[BenchmarkExample],
) -> None:
    report = near_duplicate_leakage_report(dev, test, threshold=0.97)
    lines = [
        "# Benchmark v2 Leakage Report",
        "",
        "```json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
    ]
    (REPORT_DIR / "benchmark_v2_leakage_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_defect_repair_log(examples: list[BenchmarkExample]) -> None:
    counts = count_examples(examples)
    lines = [
        "# Benchmark v2 Defects Fixed",
        "",
        "- Archived v1 after human review found unsupported summarization deadlines.",
        "- Removed inferred action-item deadlines unless the owner sentence explicitly "
        "contains the deadline.",
        "- Added source/gold checks for dates, amounts, booleans, companies, "
        "counterparties, owners, and priorities.",
        "- Added factor-based difficulty metadata.",
        "- Added held-out template-family dev/test split metadata and audits.",
        "- Generated a DEV-only manual review sample; no test gold is shown in review tooling.",
        "",
        "## V2 Counts",
        "",
        "```json",
        json.dumps(counts, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
    ]
    (REPORT_DIR / "benchmark_v2_defects_fixed.md").write_text("\n".join(lines), encoding="utf-8")


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
