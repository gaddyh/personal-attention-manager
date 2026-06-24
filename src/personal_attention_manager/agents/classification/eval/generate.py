# generate_chat_type_examples.py

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
import enum

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


ChatTypeValue = Literal["family", "work", "home", "other"]


# ----------------------------
# Generated dataset schema
# ----------------------------

class GeneratedMessage(BaseModel):
    sender: str = Field(description="Name or role of the message sender.")
    text: str = Field(description="The message text.")
    sent_time: str = Field(
        description="ISO datetime string, for example 2026-06-24T09:30:00."
    )


class Difficulty(enum.Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class GeneratedExample(BaseModel):
    id: str
    recent_messages: list[GeneratedMessage]
    expected_chat_type: ChatTypeValue
    difficulty: Difficulty
    scenario_family: str
    boundary_case: str
    competing_label: ChatTypeValue | None
    notes: str


class GeneratedDataset(BaseModel):
    examples: list[GeneratedExample]


# ----------------------------
# Generator agent
# ----------------------------

class ChatTypeDatasetGenerator:
    def __init__(self, model: str = "gpt-5.4-mini"):
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
You generate evaluation examples for a chat classification agent.

The classifier schema is:

ChatType:
- family
- work
- home
- other

Input:
- recent_messages: list of messages
  - sender
  - text
  - sent_time

Output gold label:
- expected_chat_type

Your job:
Generate high-quality labeled examples for evaluating classification accuracy.

Important rules:
1. Each example must be realistic WhatsApp-style conversation data.
2. Each example must have 1 to 5 messages.
3. Each example must have exactly one best expected_chat_type.
4. Include easy, medium, and ambiguous examples.
5. Do not make all examples keyword-obvious.
6. Do not leak the label inside the message text unnaturally.
7. Keep messages short and natural.
8. Include some Hebrew examples if the seed examples or instructions imply Hebrew.
9. Prefer diversity over repetition.
10. The notes field should explain why the gold label is correct.

Classification policy:
- family: relatives, children, parents, spouse, family logistics, emotional family messages.
- work: job, clients, meetings, projects, tasks, interviews, professional communication.
- home: house, apartment, bills, rent, mortgage, repairs, maintenance, household logistics.
- other: anything that does not clearly fit the above.

Tie-breaking:
- If family and home both appear, choose family when the main issue is people, children, parents, or relationships.
- If family and home both appear, choose home when the main issue is property, bills, repairs, rent, mortgage, or maintenance.
- If work and casual/social both appear, choose work only when the main purpose is professional.
- If uncertain, choose the best label but make the example note clear.
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


# ----------------------------
# IO helpers
# ----------------------------

def read_text_file(path: str | None, default: str) -> str:
    if not path:
        return default

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return p.read_text(encoding="utf-8")


def read_jsonl(path: str | None) -> list[dict]:
    if not path:
        return []

    p = Path(path)
    if not p.exists():
        return []

    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, examples: list[GeneratedExample]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for ex in examples:
            row = ex.model_dump()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_ids(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()

    ids = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids.add(row["id"])
    return ids


# ----------------------------
# Validation / filtering
# ----------------------------

def validate_example(ex: GeneratedExample) -> tuple[bool, str]:
    if not ex.id.strip():
        return False, "missing id"

    if not (1 <= len(ex.recent_messages) <= 5):
        return False, "recent_messages must contain 1 to 5 messages"

    for msg in ex.recent_messages:
        if not msg.sender.strip():
            return False, "empty sender"

        if not msg.text.strip():
            return False, "empty text"

        try:
            datetime.fromisoformat(msg.sent_time)
        except ValueError:
            return False, f"invalid sent_time: {msg.sent_time}"

    if not ex.notes.strip():
        return False, "missing notes"

    return True, "ok"


def dedupe_and_validate(
    examples: list[GeneratedExample],
    existing_ids: set[str],
) -> tuple[list[GeneratedExample], list[str]]:
    accepted: list[GeneratedExample] = []
    rejected: list[str] = []

    seen_ids = set(existing_ids)

    for ex in examples:
        if ex.id in seen_ids:
            rejected.append(f"{ex.id}: duplicate id")
            continue

        valid, reason = validate_example(ex)
        if not valid:
            rejected.append(f"{ex.id}: {reason}")
            continue

        accepted.append(ex)
        seen_ids.add(ex.id)

    return accepted, rejected


def count_labels(examples: list[GeneratedExample]) -> dict[str, int]:
    counts = {"family": 0, "work": 0, "home": 0, "other": 0}
    for ex in examples:
        counts[ex.expected_chat_type] += 1
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
        Counter(ex.expected_chat_type for ex in examples),
    )
    _table(
        "Message-count distribution",
        Counter(len(ex.recent_messages) for ex in examples),
    )
    _table(
        "Difficulty distribution",
        Counter(ex.difficulty.value for ex in examples),
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
Generate examples with this distribution:
- 25% one-message chats
- 35% two-message chats
- 25% three-message chats
- 15% four-to-five-message chats

At least 40% should be ambiguous or boundary cases.

Do not always include "You" as a participant.

Include:
- group chat style messages
- Hebrew slang
- mixed Hebrew/English
- vague sender names
- typos
- short replies like "ok", "done", "sure"
- cases where the first message suggests one label but the later message changes the correct label
- family vs home conflicts
- work vs casual conflicts
- home logistics discussed by family members
"""

DEFAULT_SEED_EXAMPLES = """
{"id":"family_seed_001","recent_messages":[{"sender":"Mom","text":"Are the kids coming for dinner tonight?","sent_time":"2026-06-24T17:00:00"}],"expected_chat_type":"family","notes":"A mother is asking about kids and dinner."}
{"id":"work_seed_001","recent_messages":[{"sender":"Noa","text":"Can we move the client meeting to tomorrow?","sent_time":"2026-06-24T09:00:00"}],"expected_chat_type":"work","notes":"The chat is about a client meeting."}
{"id":"home_seed_001","recent_messages":[{"sender":"Landlord","text":"The plumber can come tomorrow to fix the bathroom leak.","sent_time":"2026-06-24T12:30:00"}],"expected_chat_type":"home","notes":"The chat is about a home repair."}
{"id":"other_seed_001","recent_messages":[{"sender":"Amit","text":"Want to grab coffee later?","sent_time":"2026-06-24T15:00:00"}],"expected_chat_type":"other","notes":"A casual social message that is not clearly family, work, or home."}
"""


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    _script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser()

    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--out", type=str, default=str(_script_dir / "generated_chat_type_examples.jsonl"))
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))

    parser.add_argument(
        "--instructions-file",
        type=str,
        default=str(_script_dir / "dataset_instructions_v3.txt"),
        help="Optional text file with generation instructions.",
    )

    parser.add_argument(
        "--seed-file",
        type=str,
        default=str(_script_dir / "approved_seed_examples_v2.jsonl"),
        help="JSONL file with seed examples (default: approved_seed_examples_v2.jsonl next to this script).",
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

    generator = ChatTypeDatasetGenerator(model=args.model)

    total_accepted: list[GeneratedExample] = []
    existing_ids = load_existing_ids(args.out)

    while len(total_accepted) < args.n:
        remaining = args.n - len(total_accepted)
        current_batch_size = min(args.batch_size, remaining)

        # Simple target distribution for this batch.
        # You can later make this smarter based on label counts.
        distribution = {
            "family": max(1, current_batch_size // 4),
            "work": max(1, current_batch_size // 4),
            "home": max(1, current_batch_size // 4),
            "other": max(1, current_batch_size // 4),
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
            f"family={counts['family']} work={counts['work']} "
            f"home={counts['home']} other={counts['other']}"
        )

    print(f"\nSaved {len(total_accepted)} examples to {args.out}")
    print_coverage_report(total_accepted)


if __name__ == "__main__":
    main()