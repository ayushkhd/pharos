"""Antigravity CLI (``agy``) provider: Gemini and other models.

Uses the user's Antigravity login. Observed on agy 1.1.10, and encoded
below:

* Print mode takes the prompt only from argv (``-p=<prompt>``); stdin is
  ignored. macOS caps a process's argument block at 1 MiB, so prompts
  are size-checked before spawning and oversized ones fail fast with a
  hint to truncate.
* ``--json-schema`` takes a file path, and ``--output-format json``
  prints one JSON object per run, possibly after unrelated log lines.
* Effort is embedded in model names (``gemini-3.1-pro-high``), so no
  effort flag is passed and ``effort`` is recorded as ``model-embedded``.
"""

import json
import os
import pathlib
import shutil
from typing import Any

from pharos.providers import base
from pharos.providers import shell

PROVIDER_NAME = "agy-cli"
DEFAULT_MODEL = "gemini-3.1-pro-high"
ENV_BINARY = "AGY_BIN"
EFFORT_MARKER = "model-embedded"
STATUS_SUCCESS = "SUCCESS"

# Headroom under macOS ARG_MAX (1,048,576 bytes) for the environment
# block and the other arguments.
ARGV_LIMIT_BYTES = 900_000
# Grace period beyond the CLI's own --print-timeout before we give up.
_KILL_GRACE_S = 30


def find_binary(explicit: str | None = None) -> str:
  """Resolves the ``agy`` binary.

  Args:
    explicit: A path supplied by the caller, if any.

  Returns:
    The binary path.

  Raises:
    base.ProviderUnavailableError: If none is found.
  """
  binary = explicit or os.environ.get(ENV_BINARY) or shutil.which("agy")
  if not binary:
    raise base.ProviderUnavailableError(
        f"agy binary not found; install Antigravity or set ${ENV_BINARY}"
    )
  return binary


def last_json_object(stdout: str) -> dict[str, Any] | None:
  """Finds the CLI's result envelope among unrelated log output.

  Args:
    stdout: Everything the CLI printed.

  Returns:
    The last line that decodes to a JSON object, or ``None``.
  """
  for line in reversed(stdout.strip().splitlines()):
    line = line.strip()
    if not line.startswith("{"):
      continue
    try:
      decoded = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(decoded, dict):
      return decoded
  return None


class AntigravityCliProvider:
  """Runs prompts through ``agy -p`` with an enforced output schema.

  Attributes:
    name: Provider name recorded in manifests.
    model: Model identifier passed to the CLI.
    effort: Always ``model-embedded``.
    timeout: Seconds allowed per call.
    binary: The resolved binary.
    work: Empty scratch directory used as the sandbox root.
  """

  name = PROVIDER_NAME

  def __init__(
      self,
      model: str = DEFAULT_MODEL,
      effort: str = EFFORT_MARKER,
      timeout: float = 900,
      binary: str | None = None,
  ):
    """Resolves the binary and prepares a scratch directory.

    Args:
      model: Model identifier, with effort embedded in the name.
      effort: Ignored; recorded as ``model-embedded``.
      timeout: Seconds allowed per call.
      binary: Explicit path to the ``agy`` binary.

    Raises:
      base.ProviderUnavailableError: If no binary can be found.
    """
    del effort  # Effort is part of the model name.
    self.model = model
    self.effort = EFFORT_MARKER
    self.timeout = timeout
    self.binary = find_binary(binary)
    self.work = shell.make_scratch_dir("pharos_agy_")
    self._schema_path: pathlib.Path | None = None

  def _schema_file(self, schema: dict[str, Any]) -> pathlib.Path:
    """Returns the schema file, writing it on first use."""
    if self._schema_path is None:
      self._schema_path = shell.write_schema_file(self.work, schema)
    return self._schema_path

  def _argv(self, prompt: str, schema_path: pathlib.Path) -> list[str]:
    """Builds the ``agy`` command line."""
    return [
        self.binary,
        f"-p={prompt}",
        "--model",
        self.model,
        "--output-format",
        "json",
        "--json-schema",
        str(schema_path),
        "--print-timeout",
        f"{int(self.timeout)}s",
        "--sandbox",
        "--disable-slash-commands",
    ]

  def complete(self, prompt: str, schema: dict[str, Any]) -> base.RawResponse:
    """Runs one prompt.

    Args:
      prompt: The rendered prompt, passed as an argument.
      schema: JSON Schema the reply must satisfy.

    Returns:
      The response; failures are recorded on it, never raised.
    """
    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_bytes > ARGV_LIMIT_BYTES:
      return base.RawResponse().fail(
          base.KIND_PROVIDER,
          f"prompt is {prompt_bytes} bytes; agy only accepts the prompt via"
          f" argv (limit ~{ARGV_LIMIT_BYTES}). Re-run with"
          " --max-total-chars to truncate long tool messages.",
      )
    result = shell.run_command(
        self._argv(prompt, self._schema_file(schema)),
        timeout_s=self.timeout + _KILL_GRACE_S,
        cwd=self.work,
    )
    response = shell.response_from(result, self.timeout)
    if response.kind is not None:
      return response

    envelope = last_json_object(result.stdout)
    if envelope is None:
      kind = base.KIND_PROVIDER if result.exit_code != 0 else base.KIND_PARSE
      return response.fail(
          kind,
          f"agy output not JSON (exit {result.exit_code}):"
          f" {base.tail(result.stderr, 400) or base.tail(result.stdout, 400)}",
      )

    usage = envelope.get("usage") or {}
    status = envelope.get("status")
    response.tokens_used = int(usage.get("total_tokens") or 0) or None
    response.meta = {
        "status": status,
        "conversation_id": envelope.get("conversation_id"),
        "num_turns": envelope.get("num_turns"),
        "duration_seconds": envelope.get("duration_seconds"),
        "usage": usage,
        "cli_error": envelope.get("error"),
        "response_head": str(envelope.get("response", ""))[:300],
    }

    structured = envelope.get("structured_output")
    if isinstance(structured, dict):
      response.parsed = structured
      response.text = json.dumps(structured)
      if status != STATUS_SUCCESS:
        response.meta["note"] = f"structured_output present but status={status}"
      return response

    text = str(envelope.get("response") or "")
    response.text = text
    try:
      decoded = json.loads(text)
    except json.JSONDecodeError:
      decoded = None
    if isinstance(decoded, dict):
      response.parsed = decoded
      return response
    kind = base.KIND_PROVIDER if status != STATUS_SUCCESS else base.KIND_PARSE
    return response.fail(
        kind,
        f"status={status} no structured_output:"
        f" {envelope.get('error') or text[:500]}",
    )
