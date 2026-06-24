from __future__ import annotations

import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


def print_confusion_matrix(
    labels: list[Any],
    confusion: dict[tuple[Any, Any], int],
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


def build_report_md(
    *,
    title: str,
    run_id: str,
    model: str | None,
    data_path: str,
    total: int,
    passed: int,
    per_label_total: dict,
    per_label_passed: dict,
    confusion: dict,
    failures: list,
    labels: list[Any],
    latencies_ms: list[float],
) -> str:
    lines: list[str] = []
    accuracy = passed / total if total else 0.0

    lines.append(f"# {title} — `{run_id}`")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Model:** {model or 'env default'}  ")
    lines.append(f"**Data:** {data_path}  ")
    lines.append("")

    avg_ms = statistics.mean(latencies_ms) if latencies_ms else 0.0
    median_ms = statistics.median(latencies_ms) if latencies_ms else 0.0
    p95_ms = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)] if latencies_ms else 0.0

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total | {total} |")
    lines.append(f"| Passed | {passed} |")
    lines.append(f"| Failed | {total - passed} |")
    lines.append(f"| Accuracy | {accuracy:.3f} |")
    lines.append(f"| Latency avg | {avg_ms:.0f} ms |")
    lines.append(f"| Latency median | {median_ms:.0f} ms |")
    lines.append(f"| Latency p95 | {p95_ms:.0f} ms |")
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
