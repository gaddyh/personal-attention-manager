# eval_chat_type_agent.py

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from personal_attention_manager.agents.classification.core.agent import ChatTypeAgent
from personal_attention_manager.agents.classification.core.schemas import (
    ChatType,
    ClassificationInput,
    Message,
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


def print_confusion_matrix(
    labels: list[ChatType],
    confusion: dict[tuple[ChatType, ChatType], int],
) -> None:
    label_names = [label.value for label in labels]

    print("\nCONFUSION MATRIX")
    print("-" * 80)

    header = "expected \\ actual".ljust(20)
    for name in label_names:
        header += name.ljust(12)
    print(header)

    for expected in labels:
        row = expected.value.ljust(20)
        for actual in labels:
            row += str(confusion[(expected, actual)]).ljust(12)
        print(row)


_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def build_report_md(
    *,
    run_id: str,
    model: str | None,
    data_path: str,
    total: int,
    passed: int,
    per_label_total: dict,
    per_label_passed: dict,
    confusion: dict,
    failures: list,
    labels: list[ChatType],
) -> str:
    lines: list[str] = []
    accuracy = passed / total if total else 0.0

    lines.append(f"# Chat Type Eval — `{run_id}`")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Model:** {model or 'env default'}  ")
    lines.append(f"**Data:** {data_path}  ")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total | {total} |")
    lines.append(f"| Passed | {passed} |")
    lines.append(f"| Failed | {total - passed} |")
    lines.append(f"| Accuracy | {accuracy:.3f} |")
    lines.append("")

    lines.append("## Per-Label Accuracy")
    lines.append("")
    lines.append("| Label | Passed | Total | Accuracy |")
    lines.append("|-------|--------|-------|----------|")
    for label in labels:
        lt = per_label_total[label]
        lp = per_label_passed[label]
        la = lp / lt if lt else 0.0
        lines.append(f"| {label.value} | {lp} | {lt} | {la:.3f} |")
    lines.append("")

    lines.append("## Confusion Matrix")
    lines.append("")
    header_cells = ["expected \\ actual"] + [l.value for l in labels]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")
    for expected in labels:
        row_cells = [expected.value] + [str(confusion[(expected, actual)]) for actual in labels]
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    lines.append("## Failures")
    lines.append("")
    if not failures:
        lines.append("_None_")
    else:
        for f in failures:
            lines.append(f"### {f['id']}")
            lines.append("")
            lines.append(f"- **Expected:** {f['expected']}")
            lines.append(f"- **Actual:** {f['actual']}")
            lines.append(f"- **Confidence:** {f['confidence']}")
            lines.append(f"- **Reasoning:** {f['reasoning']}")
            lines.append(f"- **Notes:** {f['notes']}")
            lines.append("- **Messages:**")
            for msg in f["messages"]:
                lines.append(f"  - `[{msg.sent_time.isoformat()}]` {msg.sender}: {msg.text}")
            lines.append("")

    return "\n".join(lines) + "\n"


def save_run(run_id: str, content: str, runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / f"{run_id}.md"
    out_path.write_text(content, encoding="utf-8")
    return out_path


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

    for row in rows:
        total += 1
        per_label_total[row.expected_chat_type] += 1

        prediction = agent.classify(row.input)

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
    )
    saved_path = save_run(run_id, report, Path(args.runs_dir))
    print(f"\nRun report saved → {saved_path}")

    print("\nCHAT TYPE EVAL SUMMARY")
    print("-" * 80)
    print(f"Total:    {total}")
    print(f"Passed:   {passed}")
    print(f"Failed:   {total - passed}")
    print(f"Accuracy: {accuracy:.3f}")

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