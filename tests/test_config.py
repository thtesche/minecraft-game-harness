"""Config tests.

The rule under test: a key the config does not have is an error, not a silent
default. A typo that is quietly ignored produces a harness that does not do what
its author believes, and the only symptom is a surprising run hours later.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.config import BudgetConfig, Config, LayaConfig, McpConfig


def test_defaults_survive_with_no_config_file(tmp_path: Path):
    config = Config.load(tmp_path / "absent.json")

    assert config.mcp.tool_timeout_ms == 3_600_000, "a smelt exceeds a 5-minute read timeout"
    assert config.laya.min_confidence is None, "no fitted threshold means escalate, not assume"


def test_example_config_is_loadable():
    """The shipped example is the documentation, so it has to parse."""
    example = Path(__file__).resolve().parents[1] / "config.example.json"

    config = Config.load(example)

    assert config.mcp.tool_timeout_ms == 3_600_000
    assert config.laya.head_max_len == 192


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        Config.from_dict({"mcp": {"urlz": "http://localhost:25575/mcp"}})


def test_unknown_section_is_rejected():
    with pytest.raises(ValueError, match="unknown config sections"):
        Config.from_dict({"nonsense": {}})


def test_paths_are_converted(tmp_path: Path):
    config = Config.from_dict({"ledger": {"path": str(tmp_path / "a.jsonl")}})

    assert isinstance(config.ledger.path, Path)
    assert config.ledger.path == tmp_path / "a.jsonl"


def test_run_id_comes_from_the_environment(monkeypatch):
    """One prefix per process run.

    A restarted harness that reuses a submission_id from its own history trips
    SUBMISSION_CONFLICT against itself.
    """
    monkeypatch.setenv("HARNESS_RUN_ID", "from-env")

    assert Config.from_dict({}).run_id == "from-env"
    assert Config.from_dict({"run_id": "explicit"}).run_id == "explicit"


def test_sections_are_resolved_not_guessed():
    """A field with no class behind it is a loud error, not a dropped key."""
    from harness.config import _resolve

    assert _resolve("McpConfig", "mcp") is McpConfig
    assert _resolve("BudgetConfig", "budget") is BudgetConfig

    with pytest.raises(ValueError, match="no resolvable type"):
        _resolve("NotARealSection", "budget")


def test_budget_fields_are_explicit():
    """Every bound exists because something was measured to run away without it."""
    budget = BudgetConfig()

    assert budget.objective_ms > 0
    assert budget.max_attempts_per_goal > 0
    assert budget.max_consecutive_failures > 0


def test_polling_is_bounded_on_both_axes():
    config = McpConfig()

    assert config.max_polls > 0, "an unbounded poll loop terminates only by luck"
    assert config.poll_ms <= 120_000, "the server rejects timeout_ms above 120000"
    assert config.initial_wait_ms <= 120_000, "the same bound applies to the first wait"


def test_laya_option_budget_is_not_raised_by_default():
    """Past roughly 20 options labels get trimmed until they collide.

    Measured: 48 options score 1/48 at the shipped default. Raising this to
    make a bigger option list fit would trade a loud failure for a wrong answer.
    """
    laya = LayaConfig()

    assert laya.head_max_len <= 192
    assert laya.max_concurrency == 1, "concurrent MPS forwards abort the process"