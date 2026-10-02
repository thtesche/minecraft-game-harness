"""Configuration.

Every bound the objective runner enforces lives here, because each one exists
because something was measured to run away without it.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("harness.config.json")


@dataclass(frozen=True)
class McpConfig:
    """Connection to the mine-ai-mcp host."""

    url: str = "http://localhost:25575/mcp"
    health_url: str = "http://localhost:25575/health"

    #: Not optional. Smelting one stack in an ordinary furnace takes over ten
    #: minutes; the upstream docs record a 5-minute idle timeout being hit while
    #: the server kept working. A timeout is not cancellation.
    tool_timeout_ms: int = 3_600_000

    #: Bounded first wait on submission. Completion returns the full settled
    #: result here; otherwise we get pending progress and an action id.
    initial_wait_ms: int = 5_000

    #: Follow-up waits. The SDK's live playtest qualified 100 ms - 30 s.
    poll_ms: int = 2_000

    #: Ceiling on follow-up waits per objective. No unbounded polling.
    max_polls: int = 900

    #: Ceiling on a single server-sent event.
    #:
    #: httpx2 caps an SSE event at 1 MiB and the SDK offers no way to raise it,
    #: so a reply above that limit is dropped and surfaces as a lost stream. The
    #: live host advertises 37 tools in one event of 2.91 MiB, because
    #: mine-ai-mcp inlines every ``$ref`` into ``properties`` - ``wait_for_action``
    #: alone is 913 KB. Measured 2026-10-02; this is 2.7x that, bounded because
    #: an unbounded buffer is how a malformed server becomes an out-of-memory
    #: kill instead of a diagnosable error.
    max_sse_event_bytes: int = 8 * 1024 * 1024

    def health(self) -> str:
        return self.health_url


@dataclass(frozen=True)
class LedgerConfig:
    """Where decisions are recorded.

    Every field written here is unobtainable after the fact, which is why the
    runner writes them at decision time rather than at exit time.
    """

    path: Path = Path("runs/ledger.jsonl")
    sqlite_path: Path | None = Path("runs/ledger.sqlite")


@dataclass(frozen=True)
class BudgetConfig:
    """Bounds on a single objective.

    Defaults are deliberately tight. They are placeholders to be replaced with
    per-objective values once a scenario set exists, not tuned constants.
    """

    #: Wall-clock ceiling for one objective.
    objective_ms: int = 300_000

    #: Attempts at one goal. "Tried to find a log thirty times" is a fact about
    #: the goal, and a recipe may legitimately take many steps without any of
    #: them being a search.
    max_attempts_per_goal: int = 12

    #: Consecutive failures tolerated before the objective aborts.
    max_consecutive_failures: int = 3

    #: How long to wait for a survival reflex to return the body before giving up
    #: on the gate. An ``ACTION_BUSY`` refusal carrying no action id means a
    #: reflex owns the body, and the reflex is finite - but a fight can run for
    #: minutes, and the server's own combat budget allows 90 s of recovery. Bounded
    #: because an ownership that never clears would otherwise hold every
    #: objective for the whole objective budget and report a stall as progress.
    gate_wait_ms: int = 180_000


@dataclass(frozen=True)
class LayaConfig:
    """Decision-model settings.

    ``min_confidence`` has no default on purpose. A threshold is a policy chosen
    from measured accuracy at a chosen coverage on your own data; both shipped
    checkpoints are over-confident and ``laya-multilingual`` ships no fitted
    temperatures at all. Until the eval exists, an unset threshold means the
    node escalates rather than guessing.
    """

    model: str = "laya"
    device: str = "mps"
    preload: bool = True
    lang: str = "en"

    #: English checkpoint window. Raise only against a measured need: a
    #: 4,000-token state costs ~1.7 s on an Apple GPU and accuracy falls off.
    max_len: int = 512

    #: Option-prompt budget. Past roughly 20 options labels get trimmed until
    #: similar ones reach the model as identical text - a wrong answer, not an
    #: error. Measured: 48 options score 1/48 at the default.
    head_max_len: int = 192

    #: Concurrent MPS forwards abort the process. One unless batching.
    max_concurrency: int = 1

    #: Fitted in Phase 3. None means "escalate rather than assume".
    min_confidence: float | None = None


@dataclass(frozen=True)
class LlmConfig:
    """Frontier model, reached only on escalation."""

    model: str = ""
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str | None = None
    temperature: float = 0.0


@dataclass(frozen=True)
class Config:
    mcp: McpConfig = field(default_factory=McpConfig)
    ledger: LedgerConfig = field(default_factory=LedgerConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    laya: LayaConfig = field(default_factory=LayaConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)

    #: One prefix per process run, so a restarted harness never reuses a
    #: submission_id and trips SUBMISSION_CONFLICT against its own history.
    run_id: str = ""

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        """Read a JSON config file over the defaults.

        Unknown keys raise. A silently ignored key in a config file is a
        configuration that does not do what its author believes.
        """
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not config_path.exists():
            return cls()

        raw = json.loads(config_path.read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        # Section classes are resolved from the field types rather than looked up
        # by name on cls, so a dataclass field with no corresponding class is a
        # loud error instead of a silently ignored key.
        sections: dict[str, Any] = {}
        by_name = {section.name: section for section in fields(cls) if section.name != "run_id"}

        unknown_top = set(raw) - set(by_name) - {"run_id"}
        if unknown_top:
            raise ValueError(
                f"unknown config sections {sorted(unknown_top)}; expected {sorted(by_name)}"
            )

        for name, section in by_name.items():
            value = raw.get(name)
            if value is None:
                continue
            target = _resolve(section.type, name)
            sections[name] = _section_from_dict(target, name, value)

        run_id = raw.get("run_id") or os.environ.get("HARNESS_RUN_ID") or ""
        return cls(run_id=run_id, **sections)

    def resolved_run_id(self) -> str:
        return self.run_id or "harness"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ledger"]["path"] = str(self.ledger.path)
        if self.ledger.sqlite_path is not None:
            data["ledger"]["sqlite_path"] = str(self.ledger.sqlite_path)
        return data


def _resolve(annotation: Any, name: str) -> type:
    """The class a config section is built from.

    Annotations are strings under ``from __future__ import annotations``, so they
    are resolved against this module's namespace rather than evaluated.
    """
    resolved = globals().get(annotation if isinstance(annotation, str) else getattr(annotation, "__name__", ""))
    if not isinstance(resolved, type):
        raise ValueError(f"config section {name!r} has no resolvable type: {annotation!r}")
    return resolved


def _section_from_dict(target: type, name: str, values: dict[str, Any]) -> Any:
    """Build one config section, rejecting keys it does not have.

    ``path`` fields arrive as strings and stay strings as long as they are in
    the file, so they are converted here rather than at every call site.
    """
    known = {field.name for field in fields(target)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"{name}: unknown keys {sorted(unknown)}; expected {sorted(known)}")

    kwargs: dict[str, Any] = {}
    for key, value in values.items():
        if key in ("path", "sqlite_path") and value is not None:
            value = Path(value)
        kwargs[key] = value
    return target(**kwargs)