from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


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


def append_jsonl(path: str, examples: list[BaseModel]) -> None:
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
