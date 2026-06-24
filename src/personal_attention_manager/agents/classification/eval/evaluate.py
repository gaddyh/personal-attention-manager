# eval_chat_type_agent.py

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import statistics
from personal_attention_manager.agents.classification.core.agent import ChatTypeAgent
from personal_attention_manager.agents.classification.core.schemas import (
    ChatType,
    ClassificationInput,
    Message,
)
from personal_attention_manager.agents.shared.eval import (
    build_report_md,
    print_confusion_matrix,
    save_run,
)


@dataclass(frozen=True)
class EvalRow:
    id: str
    input: ClassificationInput
    expected_chat_type: ChatType
    notes: str


def load_jsonl(path: str) -> list[EvalRow]:
    rows: list[EvalRow] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            raw = json.loads(line)

            messages = [
                Message(
                    sender=msg["sender"],
                    text=msg["text"],
                    sent_time=datetime.fromisoformat(msg["sent_time"]),
                )
                for msg in raw["recent_messages"]
            ]

            rows.append(
                EvalRow(
                    id=raw["id"],
                    input=ClassificationInput(recent_messages=messages),
                    expected_chat_type=ChatType(raw["expected_chat_type"]),
                    notes=raw.get("notes", ""),
                )
            )

    return rows


_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def main() -> None:
    _script_dir = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(_script_dir / "generated_chat_type_examples.jsonl"),
        help="Path to JSONL eval dataset.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override. If omitted, chat_type_agent.py default is used.",
    )
    parser.add_argument(
        "--runs-dir",
        default=str(_RUNS_DIR),
        help="Directory where run reports are saved.",
    )
    args = parser.parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = load_jsonl(args.data)
    agent = ChatTypeAgent(model=args.model)

    total = 0
    passed = 0
    failures = []

    per_label_total = defaultdict(int)
    per_label_passed = defaultdict(int)
    confusion = defaultdict(int)

    latencies_ms: list[float] = []

    for row in rows:
        total += 1
        per_label_total[row.expected_chat_type] += 1

        t0 = time.perf_counter()
        prediction = agent.classify(row.input)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        actual = prediction.chat_type
        expected = row.expected_chat_type

        confusion[(expected, actual)] += 1

        is_correct = actual == expected
        if is_correct:
            passed += 1
            per_label_passed[expected] += 1
        else:
            failures.append(
                {
                    "id": row.id,
                    "expected": expected.value,
                    "actual": actual.value,
                    "confidence": prediction.chat_type_confidence,
                    "reasoning": prediction.reasoning,
                    "notes": row.notes,
                    "messages": row.input.recent_messages,
                }
            )

    accuracy = passed / total if total else 0.0

    labels = [ChatType.FAMILY, ChatType.WORK, ChatType.HOME, ChatType.OTHER]

    report = build_report_md(
        title="Chat Type Eval",
        run_id=run_id,
        model=args.model,
        data_path=args.data,
        total=total,
        passed=passed,
        per_label_total=per_label_total,
        per_label_passed=per_label_passed,
        confusion=confusion,
        failures=failures,
        labels=labels,
        latencies_ms=latencies_ms,
    )
    saved_path = save_run(run_id, report, Path(args.runs_dir))
    print(f"\nRun report saved → {saved_path}")

    print("\nCHAT TYPE EVAL SUMMARY")
    print("-" * 80)
    print(f"Total:    {total}")
    print(f"Passed:   {passed}")
    print(f"Failed:   {total - passed}")
    avg_ms = statistics.mean(latencies_ms) if latencies_ms else 0.0
    median_ms = statistics.median(latencies_ms) if latencies_ms else 0.0
    p95_ms = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)] if latencies_ms else 0.0

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Latency:  avg={avg_ms:.0f}ms  median={median_ms:.0f}ms  p95={p95_ms:.0f}ms")

    print("\nPER-LABEL ACCURACY")
    print("-" * 80)

    for label in labels:
        label_total = per_label_total[label]
        label_passed = per_label_passed[label]
        label_acc = label_passed / label_total if label_total else 0.0

        print(
            f"{label.value.ljust(8)} "
            f"{label_passed}/{label_total} = {label_acc:.3f}"
        )

    print_confusion_matrix(labels, confusion)

    print("\nFAILURES")
    print("-" * 80)

    if not failures:
        print("None")
        return

    for failure in failures:
        print(f"\nID:         {failure['id']}")
        print(f"Expected:   {failure['expected']}")
        print(f"Actual:     {failure['actual']}")
        print(f"Confidence: {failure['confidence']}")
        print(f"Reasoning:  {failure['reasoning']}")
        print(f"Gold notes: {failure['notes']}")
        print("Messages:")

        for msg in failure["messages"]:
            print(f"  [{msg.sent_time.isoformat()}] {msg.sender}: {msg.text}")


if __name__ == "__main__":
    main()