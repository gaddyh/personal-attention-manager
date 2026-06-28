"""
Rich + scikit-learn reporting for the booking-agent eval.

Renders two confusion matrices with per-class precision / recall / f1:
  1. BEHAVIOR  -> act / clarify / respond   (the Act-vs-Clarify metric)
  2. TOOL      -> check_availability / book_appointment / cancel_appointment
                  (catches wrong-tool errors the behavior matrix can't see,
                   because a book-instead-of-cancel still reads as 'act')

Plus per-layer clean-rates and failure-bucket frequencies.

Run:  python report.py
"""

from __future__ import annotations

from sklearn.metrics import classification_report, confusion_matrix
from rich import box
from rich.console import Console
from rich.table import Table

from grading import (
    aggregate,
    behavior_pairs,
    grade,
    per_layer_stats,
    tool_choice_pairs,
)
from examples_dataset import CASES


def _ordered_labels(y_true, y_pred, tail=("(none)", "(no_call)")):
    """Stable label order: real classes alphabetical, sentinel labels last."""
    seen = set(y_true) | set(y_pred)
    real = sorted(l for l in seen if l not in tail)
    return real + [l for l in tail if l in seen]


def render_confusion(console: Console, y_true, y_pred, title: str) -> None:
    labels = _ordered_labels(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    table = Table(title=title, box=box.SIMPLE_HEAVY, title_style="bold")
    table.add_column("gold \\ pred", style="bold", justify="right")
    for l in labels:
        table.add_column(l, justify="right")
    table.add_column("support", justify="right", style="dim")

    for i, gl in enumerate(labels):
        row = [gl]
        support = int(cm[i].sum())
        for j in range(len(labels)):
            v = int(cm[i][j])
            cell = str(v)
            if v == 0:
                cell = "[dim]·[/dim]"
            elif i == j:
                cell = f"[green]{v}[/green]"      # correct
            else:
                cell = f"[red]{v}[/red]"          # confusion
            row.append(cell)
        row.append(str(support))
        table.add_row(*row)
    console.print(table)


def render_metrics(console: Console, y_true, y_pred, title: str) -> None:
    labels = [l for l in _ordered_labels(y_true, y_pred)
              if l not in ("(none)", "(no_call)")]  # don't score sentinels as classes
    rep = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0,
    )

    table = Table(title=title, box=box.SIMPLE_HEAVY, title_style="bold")
    for col in ("class", "precision", "recall", "f1", "support"):
        table.add_column(col, justify="right")

    def _row(name, d, style=""):
        table.add_row(
            f"[{style}]{name}[/{style}]" if style else name,
            f"{d['precision']:.3f}", f"{d['recall']:.3f}",
            f"{d['f1-score']:.3f}", f"{int(d['support'])}",
        )

    for c in labels:
        _row(c, rep[c])
    table.add_section()
    acc = rep.get("accuracy")
    if acc is not None:
        table.add_row("[bold]accuracy[/bold]", "", "", f"[bold]{acc:.3f}[/bold]",
                      f"{int(rep['macro avg']['support'])}")
    _row("macro avg", rep["macro avg"], style="cyan")
    _row("weighted avg", rep["weighted avg"], style="cyan")
    console.print(table)


def render_layers(console: Console, reports) -> None:
    stats = per_layer_stats(reports)
    table = Table(title="Clean-rate by localization layer", box=box.SIMPLE_HEAVY,
                  title_style="bold")
    for col in ("layer", "applicable", "clean", "clean_rate"):
        table.add_column(col, justify="right")
    for layer, s in stats.items():
        rate = s["clean_rate"]
        colored = (f"[green]{rate:.3f}[/green]" if rate and rate >= 0.8
                   else f"[red]{rate:.3f}[/red]" if rate is not None else "-")
        table.add_row(layer, str(s["applicable"]), str(s["clean"]), colored)
    console.print(table)


def render_buckets(console: Console, reports) -> None:
    agg = aggregate(reports)
    table = Table(title="Failure buckets", box=box.SIMPLE_HEAVY, title_style="bold")
    table.add_column("layer", style="bold")
    table.add_column("bucket")
    table.add_column("count", justify="right")
    for layer_name, key in (("behavior", "behavior_buckets"),
                            ("arg", "arg_fail_buckets"),
                            ("response", "response_buckets")):
        d = agg[key]
        if not d:
            table.add_row(layer_name, "[dim](clean)[/dim]", "0")
        for i, (bucket, cnt) in enumerate(d.items()):
            table.add_row(layer_name if i == 0 else "", bucket, str(cnt))
        table.add_section()
    console.print(table)


def main() -> None:
    console = Console()
    reports = [grade(gold, observed) for _, gold, observed in CASES]

    # headline pass rates
    agg = aggregate(reports)
    console.rule("[bold]Booking-agent eval[/bold]")
    console.print(
        f"rows: [bold]{agg['rows']}[/bold]   "
        f"strict_pass_rate: [bold]{agg['strict_pass_rate']}[/bold]   "
        f"outcome_pass_rate: [bold]{agg['outcome_pass_rate']}[/bold]\n"
    )

    bt, bp = behavior_pairs(reports)
    render_confusion(console, bt, bp, "Behavior confusion (Act / Clarify / Respond)")
    render_metrics(console, bt, bp, "Behavior metrics")

    tt, tp = tool_choice_pairs(reports)
    console.print()
    render_confusion(console, tt, tp, "Tool-choice confusion")
    render_metrics(console, tt, tp, "Tool-choice metrics")

    console.print()
    render_layers(console, reports)
    console.print()
    render_buckets(console, reports)


if __name__ == "__main__":
    main()
