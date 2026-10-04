"""Task prompt contracts and versioned provider-facing instructions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from collectiveeval.core import BenchmarkExample, TaskType

CONTRACT_VERSION = "task-contracts.v1"

CRITIC_CONTRACT_VERSION = "critic-contract.v1"

CRITIC_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["needs_revision", "issues"],
    "properties": {
        "needs_revision": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {"type": "string"},
        },
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
        },
        "valid": {"type": "boolean"},
    },
    "additionalProperties": True,
}


@dataclass(frozen=True)
class TaskContract:
    """Versioned task instructions and output expectations."""

    task_type: TaskType
    prompt_version: str
    task_instructions: str
    output_schema: dict[str, Any]
    abstention_behavior: str
    citation_rules: str
    critic_instructions: str
    revision_instructions: str
    peer_summary_instructions: str
    validation_expectations: str

    def build_prompt(
        self,
        example: BenchmarkExample,
        *,
        role: str,
        candidate: dict[str, Any] | None = None,
        peer_summaries: list[dict[str, Any]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Build a provider prompt without hidden chain-of-thought requests."""

        is_critic = role == "critic"
        extra = extra or {}

        parts = [
            f"Prompt version: {self.prompt_version}",
            f"Role: {role}",
            self.task_instructions,
            self.abstention_behavior,
            self.citation_rules,
            f"Input: {example.input}",
        ]

        if candidate is not None:
            parts.append(f"Candidate answer for critique/revision: {candidate}")

        if is_critic:
            parts.extend(
                [
                    f"Critic contract version: {CRITIC_CONTRACT_VERSION}",
                    (
                        "Evaluate the candidate using only the task input and candidate. "
                        "Do not use, infer, or request any gold/reference answer."
                    ),
                    self.critic_instructions,
                    f"Candidate task schema: {self.output_schema}",
                    (
                        "Return ONLY a JSON object for the critic result. "
                        "It MUST contain needs_revision as a JSON boolean and "
                        "issues as an array of short strings. "
                        "Evidence identifiers may be included in evidence. "
                        "Do not return the task answer schema as the critic result. "
                        "Do not include hidden reasoning."
                    ),
                    f"Expected critic response schema: {CRITIC_OUTPUT_SCHEMA}",
                ]
            )
        else:
            if candidate is not None:
                parts.append(self.revision_instructions)

                critic_feedback = extra.get("critic")
                if isinstance(critic_feedback, dict):
                    parts.append(f"Structured critic feedback to apply: {critic_feedback}")

            if peer_summaries:
                parts.append(self.peer_summary_instructions)
                parts.append(f"Peer summaries: {peer_summaries}")

            parts.extend(
                [
                    (
                        "Return only a JSON object matching the expected schema. "
                        "Do not include hidden reasoning."
                    ),
                    f"Expected schema: {self.output_schema}",
                ]
            )

        parts.append(self.validation_expectations)
        return "\n\n".join(parts)


def contract_for_example(example: BenchmarkExample) -> TaskContract:
    return contract_for_task(example.task_type, schema=example.gold.get("json_schema"))


def contract_for_task(task_type: TaskType, schema: dict[str, Any] | None = None) -> TaskContract:
    if task_type in {TaskType.GROUNDED_QA, TaskType.ROBUSTNESS}:
        return TaskContract(
            task_type=task_type,
            prompt_version=f"{CONTRACT_VERSION}.grounded_qa",
            task_instructions=(
                "Answer the Japanese enterprise question using only the provided context/evidence."
            ),
            output_schema={
                "type": "object",
                "required": ["answer", "citations", "abstain", "confidence"],
                "properties": {
                    "answer": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                    "abstain": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
            abstention_behavior="If evidence is insufficient, set abstain=true and answer=''.",
            citation_rules="Citations must be evidence IDs from the provided context only.",
            critic_instructions=(
                "Critique only schema validity, citation support, abstention, and "
                "evidence consistency."
            ),
            revision_instructions="Revise only if the critique identifies a concrete issue.",
            peer_summary_instructions=(
                "Use peer answers, citations, confidence, and disagreements only; "
                "no hidden reasoning."
            ),
            validation_expectations="The evaluator will score answer, citations, and abstention.",
        )
    if task_type == TaskType.STRUCTURED_EXTRACTION:
        output_schema = schema or {"type": "object"}
        return TaskContract(
            task_type=task_type,
            prompt_version=f"{CONTRACT_VERSION}.structured_extraction",
            task_instructions="Extract the requested fields from the input text.",
            output_schema=output_schema,
            abstention_behavior=(
                "Use null only when the schema permits null and the field is absent."
            ),
            citation_rules="No citations are required unless the task schema requests them.",
            critic_instructions=(
                "Critique only JSON validity, schema compliance, and extraction support."
            ),
            revision_instructions="Revise into schema-valid JSON using only the task input.",
            peer_summary_instructions="Compare extracted field values and disagreements only.",
            validation_expectations="The evaluator will validate strict JSON Schema compliance.",
        )
    return TaskContract(
        task_type=task_type,
        prompt_version=f"{CONTRACT_VERSION}.business_summarization",
        task_instructions="Summarize the Japanese business text into the requested sections.",
        output_schema={
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
        },
        abstention_behavior=(
            "If the input is insufficient, say so in summary and leave lists empty."
        ),
        citation_rules="No citations are required unless source snippets are explicitly provided.",
        critic_instructions="Critique section completeness and support from the input only.",
        revision_instructions="Revise only unsupported or malformed sections.",
        peer_summary_instructions="Compare section content and disagreements only.",
        validation_expectations="The evaluator will validate section shape and reference overlap.",
    )
