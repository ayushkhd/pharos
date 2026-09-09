"""The provider protocol.

A provider turns a prompt plus a JSON schema into a ``RawResponse`` and
knows nothing about trajectories. That is the whole boundary: the
infrastructure on one side never trusts what comes back, and providers
on the other never see a ``Trajectory``.
"""

import dataclasses
from typing import Any, Literal, Protocol

# Why an attempt failed. The runner maps these onto execution statuses
# and uses them to decide whether and how long to back off.
ResponseKind = Literal["timeout", "provider", "parse"]
KIND_TIMEOUT: ResponseKind = "timeout"
KIND_PROVIDER: ResponseKind = "provider"
KIND_PARSE: ResponseKind = "parse"

# Only the tail of subprocess output is kept: enough to diagnose a
# failure, not enough to bloat every raw file with a model's stdout.
STDOUT_TAIL_CHARS = 6000
TIMEOUT_TAIL_CHARS = 4000


class ProviderError(Exception):
  """Base class for provider failures raised at construction time."""


class ProviderUnavailableError(ProviderError):
  """The provider cannot run here: a missing binary or credential."""


def tail(text: str | bytes | None, limit: int) -> str:
  """Returns the last ``limit`` characters of ``text``, or ``""``."""
  if text is None:
    return ""
  if isinstance(text, bytes):
    text = text.decode("utf-8", errors="replace")
  return text[-limit:]


@dataclasses.dataclass
class RawResponse:
  """What a provider returned for one prompt.

  Attributes:
    text: Final message text, expected to be JSON.
    parsed: ``text`` decoded as JSON, when it decoded.
    stdout: Tail of subprocess stdout, for CLI providers.
    stderr: Tail of subprocess stderr, for CLI providers.
    exit_code: Subprocess exit code, or ``None`` for API providers.
    duration_s: Wall-clock seconds for this attempt.
    tokens_used: Total tokens the provider reported, if any.
    error: Description of the failure, when the attempt failed.
    kind: Failure class; ``None`` on success.
    meta: Provider-specific extras kept verbatim for audit.
  """

  text: str = ""
  parsed: dict[str, Any] | None = None
  stdout: str = ""
  stderr: str = ""
  exit_code: int | None = None
  duration_s: float = 0.0
  tokens_used: int | None = None
  error: str | None = None
  kind: ResponseKind | None = None
  meta: dict[str, Any] = dataclasses.field(default_factory=dict)

  @property
  def ok(self) -> bool:
    """Returns whether this attempt produced usable JSON."""
    return self.parsed is not None and self.error is None

  def fail(self, kind: ResponseKind, error: str) -> "RawResponse":
    """Marks this response as failed and returns it, for one-line returns."""
    self.kind = kind
    self.error = error
    return self

  def to_json(self) -> dict[str, Any]:
    """Returns a JSON-serialisable form for the raw file."""
    return dataclasses.asdict(self)


class Provider(Protocol):
  """Anything that can answer a prompt with schema-constrained JSON.

  Attributes:
    name: Short provider name recorded in manifests, e.g. ``codex-cli``.
    model: Model identifier recorded in manifests.
    effort: Reasoning effort recorded in manifests, or a marker such as
      ``model-embedded`` when the provider has no such knob.
  """

  name: str
  model: str
  effort: str

  def complete(self, prompt: str, schema: dict[str, Any]) -> RawResponse:
    """Runs one attempt and returns whatever came back, never raising.

    Args:
      prompt: The full rendered prompt.
      schema: JSON Schema the reply must satisfy.
    """
