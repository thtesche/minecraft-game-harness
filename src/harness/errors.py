"""Errors raised by the harness.

The distinction that matters throughout: an *unverified* read is not a failed
read. It is a read whose answer the harness refuses to guess at, and it escalates
rather than substituting a plausible default.
"""


class HarnessError(Exception):
    """Base class for every error the harness raises deliberately."""


class UnverifiedRead(HarnessError):
    """A read could not be parsed or validated.

    Never converted into a default value. A plausible-looking number that was
    invented is worse than an escalation, because it is invisible.
    """

    def __init__(self, source: str, detail: str) -> None:
        super().__init__(f"{source}: {detail}")
        self.source = source
        self.detail = detail


class ProtocolError(HarnessError):
    """The server answered in a shape the harness does not model."""


class ObjectiveFailed(HarnessError):
    """An objective ran to a terminal non-success state.

    Carries the server's own evidence rather than a summary of it.
    """

    def __init__(self, objective: str, state: str, evidence: object = None) -> None:
        super().__init__(f"{objective} ended in state {state!r}")
        self.objective = objective
        self.state = state
        self.evidence = evidence


class BudgetExceeded(HarnessError):
    """A bound fired: time, attempts per goal, or consecutive failures."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"budget {kind}: {detail}")
        self.kind = kind
        self.detail = detail