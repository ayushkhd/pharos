"""Headless Codex CLI provider.

Uses the Codex binary bundled with the ChatGPT desktop app (or one on
``PATH``), authenticated through the user's ChatGPT login. The prompt is
passed on stdin, ``--output-schema`` makes the CLI enforce the JSON
contract, and ``-o`` writes the final message to a file this provider
reads back. Each call runs in a read-only sandbox rooted at an empty
scratch directory, so the model can never see this repository.
"""

import os
import pathlib
import re
import shutil
import tempfile
from typing import Any

from pharos.providers import base
from pharos.providers import shell

PROVIDER_NAME = "codex-cli"
DEFAULT_MODEL = "gpt-5.6-luna"
ENV_BINARY = "CODEX_BIN"
# Where the ChatGPT desktop app on macOS keeps its Codex binary.
APP_BUNDLE_BINARY = "/Applications/ChatGPT.app/Contents/Resources/codex"

_TOKENS_USED = re.compile(r"tokens used\s*\n\s*([\d,]+)")


def find_binary(explicit: str | None = None) -> pathlib.Path:
  """Resolves the Codex binary.

  Order: the explicit argument, ``$CODEX_BIN``, ``codex`` on ``PATH``,
  then the macOS app bundle.

  Args:
    explicit: A path supplied by the caller, if any.

  Returns:
    The binary path.

  Raises:
    base.ProviderUnavailableError: If no candidate exists.
  """
  candidates = [
      explicit,
      os.environ.get(ENV_BINARY),
      shutil.which("codex"),
      APP_BUNDLE_BINARY,
  ]
  for candidate in candidates:
    if candidate and pathlib.Path(candidate).is_file():
      return pathlib.Path(candidate)
  raise base.ProviderUnavailableError(
      f"codex binary not found; install Codex or set ${ENV_BINARY}"
  )


def parse_tokens_used(output: str) -> int | None:
  """Returns the token count Codex prints, if present in ``output``."""
  match = _TOKENS_USED.search(output)
  return int(match.group(1).replace(",", "")) if match else None


class CodexCliProvider:
  """Runs prompts through ``codex exec`` with an enforced output schema.

  Attributes:
    name: Provider name recorded in manifests.
    model: Model identifier passed to the CLI.
    effort: Reasoning effort passed to the CLI.
    timeout: Seconds allowed per call.
    binary: The resolved Codex binary.
    work: Empty scratch directory used as the sandbox root.
  """

  name = PROVIDER_NAME

  def __init__(
      self,
      model: str = DEFAULT_MODEL,
      effort: str = "low",
      timeout: float = 900,
      binary: str | None = None,
  ):
    """Resolves the binary and prepares a scratch directory.

    Args:
      model: Model identifier.
      effort: Reasoning effort (``minimal`` .. ``xhigh``).
      timeout: Seconds allowed per call.
      binary: Explicit path to the Codex binary.

    Raises:
      base.ProviderUnavailableError: If no Codex binary can be found.
    """
    self.model = model
    self.effort = effort
    self.timeout = timeout
    self.binary = find_binary(binary)
    self.work = shell.make_scratch_dir("pharos_codex_")
    self._schema_path: pathlib.Path | None = None

  def _schema_file(self, schema: dict[str, Any]) -> pathlib.Path:
    """Returns the schema file, writing it on first use."""
    if self._schema_path is None:
      self._schema_path = shell.write_schema_file(self.work, schema)
    return self._schema_path

  def _argv(self, schema_path: pathlib.Path, out_path: str) -> list[str]:
    """Builds the ``codex exec`` command line."""
    return [
        str(self.binary),
        "exec",
        "--model",
        self.model,
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--ephemeral",
        "-C",
        str(self.work),
        "-c",
        f'model_reasoning_effort="{self.effort}"',
        "--output-schema",
        str(schema_path),
        "-o",
        out_path,
        "-",
    ]

  def complete(self, prompt: str, schema: dict[str, Any]) -> base.RawResponse:
    """Runs one prompt.

    Args:
      prompt: The rendered prompt, fed on stdin.
      schema: JSON Schema the reply must satisfy.

    Returns:
      The response; failures are recorded on it, never raised.
    """
    fd, out_path = tempfile.mkstemp(
        prefix="out_", suffix=".json", dir=str(self.work)
    )
    os.close(fd)
    try:
      result = shell.run_command(
          self._argv(self._schema_file(schema), out_path),
          stdin=prompt,
          timeout_s=self.timeout,
          cwd=self.work,
      )
      response = shell.response_from(result, self.timeout)
      if response.kind is not None:
        return response
      response.tokens_used = parse_tokens_used(
          f"{result.stdout}\n{result.stderr}"
      )
      if result.exit_code != 0:
        return response.fail(
            base.KIND_PROVIDER,
            f"codex exit {result.exit_code}: {base.tail(result.stderr, 800)}",
        )
      try:
        text = pathlib.Path(out_path).read_text(encoding="utf-8")
      except OSError:
        text = ""
      return shell.attach_json(response, text)
    finally:
      pathlib.Path(out_path).unlink(missing_ok=True)
