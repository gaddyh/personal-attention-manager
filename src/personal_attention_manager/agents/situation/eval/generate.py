# generate_situation_examples.py

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from personal_attention_manager.agents.shared.generate import (
    GeneratedMessage,
    dedupe_and_validate,
)
from personal_attention_manager.agents.shared.io import (
    append_jsonl,
    load_existing_ids,
    read_jsonl,
    read_text_file,
)


SituationTypeValue = Literal[
    "question_needs_reply",
    "user_action_requested",
    "user_decision_needed",
    "user_info_or_artifact_needed",
    "scheduling_coordination",
    "waiting_for_other_person",
    "fyi_only",
    "resolved",
    "social_or_emotional",
    "unclear",
]


# ----------------------------
# Generated dataset schema
# ----------------------------

class GeneratedExample(BaseModel):
    id: str
    recent_messages: list[GeneratedMessage]
    expected_situation_type: SituationTypeValue
    difficulty: Literal["easy", "medium", "hard"]
    scenario_family: str
    boundary_case: str
    competing_label: SituationTypeValue | None
    notes: str


class GeneratedDataset(BaseModel):
    examples: list[GeneratedExample]


# ----------------------------
# Generator agent
# ----------------------------

class SituationDatasetGenerator:
    def __init__(self, model: str = "gpt-5.4"):
        load_dotenv()

        llm = ChatOpenAI(
            model=model,
            temperature=0.7,
        )

        self.structured_llm = llm.with_structured_output(
            GeneratedDataset,
            method="json_schema",
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You generate evaluation examples for a chat situation classification agent.

The classifier schema is:

SituationType:
- question_needs_reply
- user_action_requested
- user_decision_needed
- user_info_or_artifact_needed
- scheduling_coordination
- waiting_for_other_person
- fyi_only
- resolved
- social_or_emotional
- unclear

Input:
- recent_messages: list of messages
  - sender
  - text
  - sent_time

Output gold label:
- expected_situation_type

Your job:
Generate high-quality labeled examples for evaluating situation classification accuracy.

Important rules:
1. Each example must be realistic WhatsApp-style conversation data.
2. Each example must have 1 to 5 messages.
3. Each example must have exactly one best expected_situation_type.
4. Include easy, medium, and ambiguous examples.
5. Do not make all examples keyword-obvious.
6. Do not leak the label inside the message text unnaturally.
7. Keep messages short and natural.
8. Include some Hebrew examples if the seed examples or instructions imply Hebrew.
9. Prefer diversity over repetition.
10. The notes field should explain why the gold label is correct.
""".strip(),
                ),
                (
                    "human",
                    """
Generate {n} new examples.

User instructions:
{instructions}

Seed examples to imitate and extend:
{seed_examples}

Existing IDs already used:
{existing_ids}

Required distribution:
{distribution}

Return only examples that match the schema.
""".strip(),
                ),
            ]
        )

        self.chain = self.prompt | self.structured_llm

    def generate_batch(
        self,
        *,
        n: int,
        instructions: str,
        seed_examples: str,
        existing_ids: list[str],
        distribution: dict[str, int],
    ) -> list[GeneratedExample]:
        result: GeneratedDataset = self.chain.invoke(
            {
                "n": n,
                "instructions": instructions,
                "seed_examples": seed_examples,
                "existing_ids": ", ".join(existing_ids) if existing_ids else "None",
                "distribution": json.dumps(distribution, ensure_ascii=False),
            }
        )

        return result.examples


def count_labels(examples: list[GeneratedExample]) -> dict[str, int]:
    counts = {
        "question_needs_reply": 0,
        "user_action_requested": 0,
        "user_decision_needed": 0,
        "user_info_or_artifact_needed": 0,
        "scheduling_coordination": 0,
        "waiting_for_other_person": 0,
        "fyi_only": 0,
        "resolved": 0,
        "social_or_emotional": 0,
        "unclear": 0,
    }
    for ex in examples:
        counts[ex.expected_situation_type] += 1
    return counts


def print_coverage_report(examples: list[GeneratedExample]) -> None:
    n = len(examples)
    if not n:
        return

    def _table(title: str, counter: Counter) -> None:
        print(f"\n{title}")
        print("-" * 48)
        for key, count in sorted(counter.items(), key=lambda x: -x[1]):
            bar = "#" * count
            print(f"  {str(key):<30} {count:>4}  {count / n * 100:>5.1f}%  {bar}")

    _table(
        "Label distribution",
        Counter(ex.expected_situation_type for ex in examples),
    )
    _table(
        "Message-count distribution",
        Counter(len(ex.recent_messages) for ex in examples),
    )
    _table(
        "Difficulty distribution",
        Counter(ex.difficulty for ex in examples),
    )
    _table(
        "Scenario-family distribution",
        Counter(ex.scenario_family for ex in examples),
    )
    _table(
        "Boundary-case distribution",
        Counter(ex.boundary_case for ex in examples),
    )
    _table(
        "Competing-label distribution",
        Counter(ex.competing_label if ex.competing_label else "none" for ex in examples),
    )


# ----------------------------
# Defaults
# ----------------------------

DEFAULT_INSTRUCTIONS = """
TODO: fill in default generation instructions for the situation agent.
"""

DEFAULT_SEED_EXAMPLES = """
{"id":"waiting_seed_001","recent_messages":[{"sender":"Me","text":"Can you review the doc?","sent_time":"2026-06-24T09:00:00"},{"sender":"Me","text":"Let me know when you get a chance","sent_time":"2026-06-24T09:01:00"}],"expected_situation_type":"waiting_for_other_person","notes":"User sent two messages with no reply. Waiting on the other person."}
{"id":"action_seed_001","recent_messages":[{"sender":"Dana","text":"Can you review this before I send?","sent_time":"2026-06-24T09:00:00"}],"expected_situation_type":"user_action_requested","notes":"Incoming request that requires a concrete action from the user."}
"""


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    _script_dir = Path(__file__).parent
    load_dotenv()
    parser = argparse.ArgumentParser()

    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--out", type=str, default=str(_script_dir / "generated_situation_examples.jsonl"))
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL_GENERATOR", "gpt-5.4-mini"))

    parser.add_argument(
        "--instructions-file",
        type=str,
        default=str(_script_dir / "dataset_instructions_v1.txt"),
        help="Optional text file with generation instructions.",
    )

    parser.add_argument(
        "--seed-file",
        type=str,
        default=str(_script_dir / "approved_seed_examples.jsonl"),
        help="JSONL file with seed examples.",
    )

    args = parser.parse_args()

    instructions = read_text_file(args.instructions_file, DEFAULT_INSTRUCTIONS)

    seed_rows = read_jsonl(args.seed_file)
    if seed_rows:
        seed_examples = "\n".join(
            json.dumps(row, ensure_ascii=False) for row in seed_rows
        )
    else:
        seed_examples = DEFAULT_SEED_EXAMPLES

    print(f"Using model: {args.model}")
    generator = SituationDatasetGenerator(model=args.model)

    total_accepted: list[GeneratedExample] = []
    existing_ids = load_existing_ids(args.out)

    while len(total_accepted) < args.n:
        remaining = args.n - len(total_accepted)
        current_batch_size = min(args.batch_size, remaining)

        distribution = {
            "question_needs_reply": max(1, round(current_batch_size * 0.20)),
            "user_action_requested": max(1, round(current_batch_size * 0.20)),
            "user_info_or_artifact_needed": max(1, round(current_batch_size * 0.15)),
            "user_decision_needed": max(1, round(current_batch_size * 0.10)),
            "scheduling_coordination": max(1, round(current_batch_size * 0.10)),
            "waiting_for_other_person": max(1, round(current_batch_size * 0.10)),
            "resolved": max(1, round(current_batch_size * 0.10)),
            "fyi_only": 1,
            "social_or_emotional": 1,
            "unclear": 1,
        }

        batch = generator.generate_batch(
            n=current_batch_size,
            instructions=instructions,
            seed_examples=seed_examples,
            existing_ids=sorted(existing_ids),
            distribution=distribution,
        )

        accepted, rejected = dedupe_and_validate(batch, existing_ids)

        if rejected:
            print("\nRejected:")
            for reason in rejected:
                print(f"  - {reason}")

        if not accepted:
            raise RuntimeError("No valid examples generated in this batch.")

        append_jsonl(args.out, accepted)

        for ex in accepted:
            existing_ids.add(ex.id)

        total_accepted.extend(accepted)

        counts = count_labels(total_accepted)
        print(
            f"Accepted {len(total_accepted)}/{args.n} | "
            + "  ".join(f"{k}={v}" for k, v in counts.items() if v > 0)
        )

    print(f"\nSaved {len(total_accepted)} examples to {args.out}")
    print_coverage_report(total_accepted)


if __name__ == "__main__":
    main()
