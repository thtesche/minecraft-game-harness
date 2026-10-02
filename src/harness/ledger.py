"""Append-only decision ledger.

One row per decision. Two consumers, both of which are unobtainable after the
fact if the fields are not written at decision time:

* the Phase 2 baseline (LLM calls per objective, success rate)
* the Phase 3 eval set (state, question, answer, full distribution)

The full probability distribution is recorded, not just the argmax.
Calibration work needs the alternatives.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import LedgerConfig


@dataclass
class DecisionRow:
    """One decision and what came of it."""

    #: What the model saw.
    state_vector: dict[str, Any] = field(default_factory=dict)
    state_hash: str = ""

    #: The question as asked.
    question: dict[str, Any] = field(default_factory=dict)
    options: list[str] = field(default_factory=list)
    raw_scores: dict[str, Any] = field(default_factory=dict)

    #: The decision, and whether it was trusted.
    answer: str | None = None
    answer_confidence: float | None = None
    escalated: bool = False
    escalation_reason: str | None = None

    #: What was actually submitted.
    objective_tool: str | None = None
    objective_args: dict[str, Any] = field(default_factory=dict)

    #: What the server settled with.
    outcome: str | None = None
    evidence_ok: bool | None = None
    duration_ms: int | None = None
    error: str | None = None

    #: Bookkeeping.
    run_id: str = ""
    at: float = field(default_factory=time.time)
    row_id: str = field(default_factory=lambda: uuid.uuid4().hex)


class Ledger:
    """JSONL append plus a SQLite index over the same rows."""

    def __init__(self, config: LedgerConfig, run_id: str) -> None:
        self.config = config
        self.run_id = run_id
        self.path = Path(config.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sqlite: sqlite3.Connection | None = None
        if config.sqlite_path is not None:
            sqlite_path = Path(config.sqlite_path)
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._sqlite = sqlite3.connect(sqlite_path)
            self._sqlite.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    row_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    at REAL NOT NULL,
                    state_hash TEXT,
                    answer TEXT,
                    answer_confidence REAL,
                    escalated INTEGER,
                    objective_tool TEXT,
                    outcome TEXT,
                    evidence_ok INTEGER,
                    duration_ms INTEGER,
                    payload TEXT NOT NULL
                )
                """
            )
            self._sqlite.commit()

    def record(self, row: DecisionRow) -> DecisionRow:
        """Append one row durably.

        Flushes per row. A ledger that loses its tail when the process dies is
        useless for measuring, and the tail is exactly where the interesting
        failures are.
        """
        row.run_id = row.run_id or self.run_id
        line = json.dumps(asdict(row), default=str)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

        if self._sqlite is not None:
            self._sqlite.execute(
                """
                INSERT OR REPLACE INTO decisions
                    (row_id, run_id, at, state_hash, answer, answer_confidence,
                     escalated, objective_tool, outcome, evidence_ok, duration_ms, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.row_id,
                    row.run_id,
                    row.at,
                    row.state_hash,
                    row.answer,
                    row.answer_confidence,
                    int(row.escalated),
                    row.objective_tool,
                    row.outcome,
                    None if row.evidence_ok is None else int(row.evidence_ok),
                    row.duration_ms,
                    line,
                ),
            )
            self._sqlite.commit()
        return row

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def counts(self) -> dict[str, int]:
        """Row counts by outcome, for a quick sanity read."""
        totals: dict[str, int] = {}
        for row in self.read_all():
            key = row.get("outcome") or "no_outcome"
            totals[key] = totals.get(key, 0) + 1
        return totals

    def close(self) -> None:
        if self._sqlite is not None:
            self._sqlite.close()
            self._sqlite = None

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()