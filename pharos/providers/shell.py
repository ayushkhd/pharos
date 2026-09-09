"""Shared subprocess plumbing for CLI-backed providers.

Every CLI provider runs one process per prompt inside an empty scratch
directory (so a sandboxed agent never sees this repository), captures
output, and turns a timeout into a failed response instead of an
exception. That logic lives here once.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import json
import pathlib
import subprocess
import tempfile
import time
from typing import Any

from pharos.providers import base

SCHEMA_FILE = "schema.json"


@dataclasses.dataclass(frozen=True)
class CommandResult:
  """Outcome of one subprocess run.

  Attributes:
    stdout: Everything the process wrote to stdout (partial on timeout).
    stderr: Everything the process wrote to stderr (partial on timeout).
    exit_code: The exit status, or ``None`` if the process timed out.
    duration_s: Wall-clock seconds.
    timed_out: Whether the process was killed for exceeding its timeout.
  """

  stdout: str
  stderr: str
  exit_code: int | None
  duration_s: float
  timed_out: bool


def make_scratch_dir(prefix: str) -> pathlib.Path:
  """Creates an empty working directory for a provider's subprocesses."""
  return pathlib.Path(tempfile.mkdtemp(prefix=prefix))


def write_schema_file(
    directory: pathlib.Path, schema: Mapping[str, Any]
) -> pathlib.Path:
  """Writes ``schema`` as JSON for CLIs that take a schema path.

  Args:
    directory: Where to write; the file name is fixed.
    schema: The JSON Schema.

  Returns:
    The file written.
  """
  path = directory / SCHEMA_FILE
  path.write_text(json.dumps(schema), encoding="utf-8")
  return path


def run_command(
    argv: Sequence[str],
    *,
    timeout_s: float,
    cwd: pathlib.Path,
    stdin: str | None = None,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
  """Runs a command to completion or timeout.

  Args:
    argv: The command and its arguments.
    timeout_s: Seconds after which the process is killed.
    cwd: Working directory for the process.
    stdin: Text to feed on standard input, if any.
    env: Environment for the process; defaults to the current one.

  Returns:
    The captured outcome. Never raises for a non-zero exit or timeout.
  """
  started = time.time()
  try:
    proc = subprocess.run(
        list(argv),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=str(cwd),
        env=dict(env) if env is not None else None,
        check=False,
    )
  except subprocess.TimeoutExpired as e:
    return CommandResult(
        stdout=base.tail(e.stdout, base.TIMEOUT_TAIL_CHARS),
        stderr=base.tail(e.stderr, base.TIMEOUT_TAIL_CHARS),
        exit_code=None,
        duration_s=time.time() - started,
        timed_out=True,
    )
  return CommandResult(
      stdout=proc.stdout,
      stderr=proc.stderr,
      exit_code=proc.returncode,
      duration_s=time.time() - started,
      timed_out=False,
  )


def response_from(result: CommandResult, timeout_s: float) -> base.RawResponse:
  """Seeds a RawResponse from a command outcome.

  Args:
    result: The command outcome.
    timeout_s: The timeout that applied, for the error message.

  Returns:
    A response carrying output tails and timing. On timeout it is already
    marked failed; otherwise the caller parses stdout and decides.
  """
  response = base.RawResponse(
      stdout=base.tail(result.stdout, base.STDOUT_TAIL_CHARS),
      stderr=base.tail(result.stderr, base.STDOUT_TAIL_CHARS),
      exit_code=result.exit_code,
      duration_s=result.duration_s,
  )
  if result.timed_out:
    return response.fail(base.KIND_TIMEOUT, f"timeout after {timeout_s}s")
  return response


def attach_json(response: base.RawResponse, text: str) -> base.RawResponse:
  """Records the final message text and decodes it as JSON.

  Args:
    response: The response to fill in.
    text: The model's final message.

  Returns:
    ``response``, with ``parsed`` set or a parse failure recorded.
  """
  response.text = text
  if not text.strip():
    return response.fail(base.KIND_PARSE, "empty final message")
  try:
    decoded = json.loads(text)
  except json.JSONDecodeError as e:
    return response.fail(base.KIND_PARSE, f"json parse failed: {e}")
  if not isinstance(decoded, dict):
    return response.fail(base.KIND_PARSE, "final message is not a JSON object")
  response.parsed = decoded
  return response
