"""Headless Claude Code CLI provider.

Uses the ``claude`` binary and the user's Claude subscription login. The
prompt is fed on stdin and ``--json-schema`` makes the CLI enforce the
output contract; ``--output-format json`` wraps the reply in a result
envelope this provider unpacks.

Quirks observed on claude 2.1.15, which the code below accommodates:

* The ``opus`` alias resolves to an older Opus, so pass a full model id.
* Every structured-output call reports ``is_error: true`` from a failed
  second turn *after* ``structured_output`` has been captured. A present
  ``structured_output`` is therefore treated as success, and the CLI's
  own error text is kept in ``meta`` for audit.
* The CLI exposes no effort flag; ``effort`` is recorded as
  ``cli-default``.
"""

import json
import os
import shutil
from typing import Any

from pharos.providers import base
from pharos.providers import shell

PROVIDER_NAME = "claude-cli"
DEFAULT_MODEL = "claude-opus-5"
ENV_BINARY = "CLAUDE_BIN"
EFFORT_MARKER = "cli-default"

# Variables the CLI sets for its own child processes. Inheriting them
# makes the CLI believe it is nested inside a session and change its
# behaviour, so they are stripped from the environment we pass.
STRIP_ENV: tuple[str, ...] = (
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_CHILD_SESSION",
)
_USAGE_KEYS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)


def find_binary(explicit: str | None = None) -> str:
  """Resolves the Claude Code binary.

  Args:
    explicit: A path supplied by the caller, if any.

  Returns:
    The binary path.

  Raises:
    base.ProviderUnavailableError: If none is found.
  """
  binary = explicit or os.environ.get(ENV_BINARY) or shutil.which("claude")
  if not binary:
    raise base.ProviderUnavailableError(
        f"claude binary not found; install Claude Code or set ${ENV_BINARY}"
    )
  return binary


def _total_tokens(usage: dict[str, Any]) -> int | None:
  """Sums the usage counters the CLI reports, or ``None`` if absent."""
  total = sum(int(usage.get(key) or 0) for key in _USAGE_KEYS)
  return total or None


class ClaudeCliProvider:
  """Runs prompts through ``claude -p`` with an enforced output schema.

  Attributes:
    name: Provider name recorded in manifests.
    model: Model identifier passed to the CLI.
    effort: Always ``cli-default``; the CLI has no effort knob.
    timeout: Seconds allowed per call.
    binary: The resolved binary.
    max_budget_usd: Spend cap passed to the CLI per call.
    work: Empty scratch directory used as the working directory.
  """

  name = PROVIDER_NAME

  def __init__(
      self,
      model: str = DEFAULT_MODEL,
      effort: str = EFFORT_MARKER,
      timeout: float = 900,
      binary: str | None = None,
      max_budget_usd: float = 5.0,
  ):
    """Resolves the binary and prepares the environment.

    Args:
      model: Model identifier. Use a full id, not an alias.
      effort: Ignored; recorded as ``cli-default``.
      timeout: Seconds allowed per call.
      binary: Explicit path to the ``claude`` binary.
      max_budget_usd: Spend cap passed to the CLI per call.

    Raises:
      base.ProviderUnavailableError: If no binary can be found.
    """
    del effort  # The CLI exposes no effort flag.
    self.model = model
    self.effort = EFFORT_MARKER
    self.timeout = timeout
    self.binary = find_binary(binary)
    self.max_budget_usd = max_budget_usd
    self.env = {k: v for k, v in os.environ.items() if k not in STRIP_ENV}
    self.work = shell.make_scratch_dir("pharos_claude_")

  def _argv(self, schema: dict[str, Any]) -> list[str]:
    """Builds the ``claude -p`` command line."""
    return [
        self.binary,
        "-p",
        "--model",
        self.model,
        "--tools",
        "",
        "--no-session-persistence",
        "--max-budget-usd",
        str(self.max_budget_usd),
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(schema),
    ]

  def complete(self, prompt: str, schema: dict[str, Any]) -> base.RawResponse:
    """Runs one prompt.

    Args:
      prompt: The rendered prompt, fed on stdin.
      schema: JSON Schema the reply must satisfy.

    Returns:
      The response; failures are recorded on it, never raised.
    """
    result = shell.run_command(
        self._argv(schema),
        stdin=prompt,
        timeout_s=self.timeout,
        cwd=self.work,
        env=self.env,
    )
    response = shell.response_from(result, self.timeout)
    if response.kind is not None:
      return response

    try:
      envelope = json.loads(result.stdout.strip())
    except json.JSONDecodeError as e:
      kind = base.KIND_PROVIDER if result.exit_code != 0 else base.KIND_PARSE
      return response.fail(
          kind,
          f"cli output not JSON (exit {result.exit_code}): {e};"
          f" stderr={base.tail(result.stderr, 400)}",
      )
    if not isinstance(envelope, dict):
      return response.fail(base.KIND_PARSE, "cli output is not a JSON object")

    usage = envelope.get("usage") or {}
    response.tokens_used = _total_tokens(usage)
    response.meta = {
        "total_cost_usd": envelope.get("total_cost_usd"),
        "num_turns": envelope.get("num_turns"),
        "is_error": envelope.get("is_error"),
        "subtype": envelope.get("subtype"),
        "models": list((envelope.get("modelUsage") or {}).keys()),
        "cli_result": str(envelope.get("result", ""))[:500],
        "session_id": envelope.get("session_id"),
        "usage": usage,
    }

    structured = envelope.get("structured_output")
    if isinstance(structured, dict):
      response.parsed = structured
      response.text = json.dumps(structured)
      return response

    # No structured output: the plain result may still be JSON.
    text = str(envelope.get("result") or "")
    response.text = text
    try:
      decoded = json.loads(text)
    except json.JSONDecodeError:
      decoded = None
    if isinstance(decoded, dict):
      response.parsed = decoded
      return response
    kind = base.KIND_PROVIDER if envelope.get("is_error") else base.KIND_PARSE
    return response.fail(
        kind,
        f"no structured_output; is_error={envelope.get('is_error')}:"
        f" {text[:600]}",
    )
