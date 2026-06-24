# scripts/print_waiting_failures.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich import box


FAIL_THRESHOLD = 1.0


def metric_mark(value: float) -> str:
    if value == 1.0:
        return "[green]✓[/green]"
    if value == 0.0:
        return "[red]✗[/red]"
    return f"[yellow]{value:.2f}[/yellow]"


def short(value: Any, max_len: int = 42) -> str:
    if value is None:
        return "-"
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def load_data(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data["predictions"], data.get("summary")
    return data, None


def metric_md(value: float) -> str:
    if value == 1.0:
        return "✓"
    if value == 0.0:
        return "✗"
    return f"{value:.2f}"


def build_md_report(rows: list[dict[str, Any]], summary: dict[str, Any] | None) -> str:
    lines: list[str] = []

    lines.append("# Waiting Obligation Agent — Failure Report")
    lines.append("")

    if summary:
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **run_id**: `{summary.get('run_id', '-')}`")
        lines.append(f"- **dataset**: `{summary.get('dataset', '-')}`")
        lines.append(f"- **model**: `{summary.get('model', '-')}`")
        lines.append(f"- **count**: {summary.get('count', '-')}")
        lines.append(f"- **avg_score**: {summary.get('avg_score', '-')}")
        lines.append("")
        ma = summary.get("metric_averages", {})
        if ma:
            lines.append("| metric | avg |")
            lines.append("|---|---|")
            for k, v in ma.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")
        fc = summary.get("failure_counts", {})
        if fc:
            lines.append("| failure | count |")
            lines.append("|---|---|")
            for k, v in fc.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")

    lines.append("## Failing Rows")
    lines.append("")
    lines.append("| ID | score | failure | exp waiting_on | pred waiting_on | exp obligation | pred obligation | waiting_on | state | obligation | exp_action | no_fake_action | evidence |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    for row in rows:
        score = float(row["score"])
        if score >= FAIL_THRESHOLD and row.get("failure_type") is None:
            continue
        expected = row["expected"]
        prediction = row["prediction"]
        metrics = row["metrics"]
        lines.append(
            "| {} | {:.2f} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["id"],
                score,
                row.get("failure_type") or "-",
                expected.get("waiting_on", "-"),
                prediction.get("waiting_on", "-"),
                expected.get("obligation_type", "-"),
                prediction.get("obligation_type", "-"),
                metric_md(metrics["waiting_on_exact"]),
                metric_md(metrics["conversation_state_exact"]),
                metric_md(metrics["obligation_type_exact"]),
                metric_md(metrics["has_expected_action_when_me"]),
                metric_md(metrics["no_expected_action_when_not_me"]),
                metric_md(metrics["evidence_present"]),
            )
        )

    return "\n".join(lines) + "\n"


def build_failures_table(rows: list[dict[str, Any]]) -> Table:
    table = Table(
        title="Waiting Obligation Agent — Failing Rows",
        box=box.SIMPLE_HEAVY,
        show_lines=True,
    )

    table.add_column("ID", style="bold", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Failure", style="red", no_wrap=True)

    table.add_column("exp waiting_on", no_wrap=True)
    table.add_column("pred waiting_on", no_wrap=True)

    table.add_column("exp obligation", no_wrap=True)
    table.add_column("pred obligation", no_wrap=True)

    table.add_column("waiting_on", no_wrap=True)
    table.add_column("state", no_wrap=True)
    table.add_column("obligation", no_wrap=True)
    table.add_column("exp_action", no_wrap=True)
    table.add_column("no_fake_action", no_wrap=True)
    table.add_column("evidence", no_wrap=True)

    for row in rows:
        score = float(row["score"])

        if score >= FAIL_THRESHOLD and row.get("failure_type") is None:
            continue

        expected = row["expected"]
        prediction = row["prediction"]
        metrics = row["metrics"]

        table.add_row(
            row["id"],
            f"{score:.2f}",
            row.get("failure_type") or "-",
            short(expected.get("waiting_on")),
            short(prediction.get("waiting_on")),
            short(expected.get("obligation_type")),
            short(prediction.get("obligation_type")),
            metric_mark(metrics["waiting_on_exact"]),
            metric_mark(metrics["conversation_state_exact"]),
            metric_mark(metrics["obligation_type_exact"]),
            metric_mark(metrics["has_expected_action_when_me"]),
            metric_mark(metrics["no_expected_action_when_not_me"]),
            metric_mark(metrics["evidence_present"]),
        )

    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()

    rows, summary = load_data(args.predictions)

    console = Console(width=220)
    table = build_failures_table(rows)
    console.print(table)

    md_path = args.predictions.with_suffix(".md")
    md_path.write_text(build_md_report(rows, summary), encoding="utf-8")
    print(f"Saved report to {md_path}")


if __name__ == "__main__":
    main()