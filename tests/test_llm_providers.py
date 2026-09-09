"""Tests for the CLI- and API-backed providers, using fake binaries."""

import json
import os
import pathlib
import stat
from typing import Any

import pytest

from pharos import schema
from pharos.providers import agy_cli
from pharos.providers import base
from pharos.providers import claude_cli
from pharos.providers import codex_cli
from pharos.providers import factory
from pharos.providers import openai_api
from pharos.providers import shell

OUTPUT: dict[str, Any] = {"verdict": "clean", "summary": "s", "findings": []}


def _fake_binary(tmp_path: pathlib.Path, name: str, script: str) -> str:
  path = tmp_path / name
  path.write_text("#!/bin/sh\n" + script, encoding="utf-8")
  path.chmod(path.stat().st_mode | stat.S_IXUSR)
  return str(path)


# A stand-in for `codex exec`: saves stdin, writes $FAKE_OUTPUT to the -o
# file, prints a token line, exits with $FAKE_EXIT.
CODEX_SCRIPT = """
cat > "$PWD/prompt.txt"
out=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-o" ]; then out="$2"; shift; fi
  shift
done
printf '%s' "$FAKE_OUTPUT" > "$out"
printf 'thinking...\\ntokens used\\n1,234\\n'
exit "${FAKE_EXIT:-0}"
"""

# Stand-ins for `claude -p` and `agy -p`: echo $FAKE_STDOUT.
ECHO_SCRIPT = """
printf '%s' "$FAKE_STDOUT"
exit "${FAKE_EXIT:-0}"
"""


# --- shell -----------------------------------------------------------------


def test_run_command_captures_output_and_exit_code(tmp_path: pathlib.Path):
  binary = _fake_binary(tmp_path, "b", "cat; echo err >&2; exit 3\n")

  result = shell.run_command([binary], stdin="in", timeout_s=5, cwd=tmp_path)

  assert result == shell.CommandResult(
      stdout="in",
      stderr="err\n",
      exit_code=3,
      duration_s=result.duration_s,
      timed_out=False,
  )
  response = shell.response_from(result, 5)
  assert response.kind is None and response.exit_code == 3


def test_run_command_turns_timeout_into_failed_response(
    tmp_path: pathlib.Path,
):
  binary = _fake_binary(tmp_path, "slow", "echo partial; sleep 5\n")

  result = shell.run_command([binary], timeout_s=0.3, cwd=tmp_path)
  response = shell.response_from(result, 0.3)

  assert result.timed_out and result.exit_code is None
  assert response.kind == "timeout"
  assert response.error == "timeout after 0.3s"
  # Partial output on timeout is captured only on Windows; elsewhere
  # CPython discards it, so nothing is asserted about stdout here.


@pytest.mark.parametrize(
    "text, kind, error_prefix",
    [
        (json.dumps(OUTPUT), None, None),
        ("", "parse", "empty final message"),
        ("{not json", "parse", "json parse failed"),
        ("[1]", "parse", "final message is not a JSON object"),
    ],
)
def test_attach_json(text: str, kind: str | None, error_prefix: str | None):
  response = shell.attach_json(base.RawResponse(), text)

  assert response.kind == kind
  assert response.ok == (kind is None)
  if error_prefix:
    assert response.error is not None
    assert response.error.startswith(error_prefix)


# --- codex ---------------------------------------------------------------


def _codex(tmp_path: pathlib.Path) -> codex_cli.CodexCliProvider:
  binary = _fake_binary(tmp_path, "codex", CODEX_SCRIPT)
  return codex_cli.CodexCliProvider(
      model="m", effort="low", timeout=5, binary=binary
  )


def test_codex_success_parses_output_and_tokens(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  monkeypatch.setenv("FAKE_OUTPUT", json.dumps(OUTPUT))
  provider = _codex(tmp_path)

  response = provider.complete("PROMPT", schema.MODEL_OUTPUT_SCHEMA)

  assert response.ok
  assert response.parsed == OUTPUT
  assert response.tokens_used == 1234
  assert response.exit_code == 0
  assert (provider.work / "prompt.txt").read_text() == "PROMPT"
  assert json.loads((provider.work / shell.SCHEMA_FILE).read_text()) == (
      schema.MODEL_OUTPUT_SCHEMA
  )
  assert not list(provider.work.glob("out_*"))  # temp output cleaned up


def test_codex_nonzero_exit_is_a_provider_failure(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  monkeypatch.setenv("FAKE_OUTPUT", json.dumps(OUTPUT))
  monkeypatch.setenv("FAKE_EXIT", "2")

  response = _codex(tmp_path).complete("p", schema.MODEL_OUTPUT_SCHEMA)

  assert response.kind == "provider"
  assert response.error is not None and response.error.startswith(
      "codex exit 2"
  )
  assert response.parsed is None


def test_codex_bad_output_is_a_parse_failure(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  monkeypatch.setenv("FAKE_OUTPUT", "garbage")

  response = _codex(tmp_path).complete("p", schema.MODEL_OUTPUT_SCHEMA)

  assert response.kind == "parse"


def test_codex_binary_resolution(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  monkeypatch.delenv(codex_cli.ENV_BINARY, raising=False)
  monkeypatch.setattr(codex_cli.shutil, "which", lambda _: None)
  monkeypatch.setattr(codex_cli, "APP_BUNDLE_BINARY", str(tmp_path / "nope"))
  with pytest.raises(base.ProviderUnavailableError, match="codex binary"):
    codex_cli.CodexCliProvider(binary=None)

  binary = _fake_binary(tmp_path, "codex", CODEX_SCRIPT)
  monkeypatch.setenv(codex_cli.ENV_BINARY, binary)
  assert codex_cli.find_binary() == pathlib.Path(binary)
  assert codex_cli.parse_tokens_used("x\ntokens used\n  12,345\n") == 12345
  assert codex_cli.parse_tokens_used("nothing") is None


# --- claude --------------------------------------------------------------


def _claude(tmp_path: pathlib.Path) -> claude_cli.ClaudeCliProvider:
  binary = _fake_binary(tmp_path, "claude", ECHO_SCRIPT)
  return claude_cli.ClaudeCliProvider(
      model="claude-x", timeout=5, binary=binary
  )


def test_claude_structured_output_wins_even_when_cli_reports_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  envelope = {
      "structured_output": OUTPUT,
      "is_error": True,
      "result": "second turn failed",
      "usage": {"input_tokens": 10, "output_tokens": 5},
      "total_cost_usd": 0.01,
      "modelUsage": {"claude-x": {}},
  }
  monkeypatch.setenv("FAKE_STDOUT", json.dumps(envelope))
  provider = _claude(tmp_path)

  response = provider.complete("p", schema.MODEL_OUTPUT_SCHEMA)

  assert response.ok and response.parsed == OUTPUT
  assert response.tokens_used == 15
  assert response.meta["is_error"] is True
  assert response.meta["models"] == ["claude-x"]
  assert provider.effort == claude_cli.EFFORT_MARKER
  assert all(k not in provider.env for k in claude_cli.STRIP_ENV)


def test_claude_error_without_structured_output(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  monkeypatch.setenv(
      "FAKE_STDOUT", json.dumps({"is_error": True, "result": "budget exceeded"})
  )

  response = _claude(tmp_path).complete("p", schema.MODEL_OUTPUT_SCHEMA)

  assert response.kind == "provider"
  assert "budget exceeded" in (response.error or "")


def test_claude_non_json_stdout(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  monkeypatch.setenv("FAKE_STDOUT", "Error: not logged in")
  monkeypatch.setenv("FAKE_EXIT", "1")

  response = _claude(tmp_path).complete("p", schema.MODEL_OUTPUT_SCHEMA)

  assert response.kind == "provider"
  assert (response.error or "").startswith("cli output not JSON (exit 1)")


def test_claude_missing_binary(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.delenv(claude_cli.ENV_BINARY, raising=False)
  monkeypatch.setattr(claude_cli.shutil, "which", lambda _: None)
  with pytest.raises(base.ProviderUnavailableError, match="claude binary"):
    claude_cli.ClaudeCliProvider()


# --- agy -----------------------------------------------------------------


def _agy(tmp_path: pathlib.Path) -> agy_cli.AntigravityCliProvider:
  binary = _fake_binary(tmp_path, "agy", ECHO_SCRIPT)
  return agy_cli.AntigravityCliProvider(
      model="gemini-x", timeout=5, binary=binary
  )


def test_agy_takes_the_last_json_line(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  envelope = {
      "status": "SUCCESS",
      "structured_output": OUTPUT,
      "usage": {"total_tokens": 77},
  }
  monkeypatch.setenv(
      "FAKE_STDOUT", "loading extensions...\n{bad json\n" + json.dumps(envelope)
  )

  response = _agy(tmp_path).complete("p", schema.MODEL_OUTPUT_SCHEMA)

  assert response.ok and response.parsed == OUTPUT
  assert response.tokens_used == 77
  assert response.meta["status"] == "SUCCESS"


def test_agy_error_status_without_structured_output(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  monkeypatch.setenv(
      "FAKE_STDOUT", json.dumps({"status": "ERROR", "error": "quota exceeded"})
  )

  response = _agy(tmp_path).complete("p", schema.MODEL_OUTPUT_SCHEMA)

  assert response.kind == "provider"
  assert "quota exceeded" in (response.error or "")


def test_agy_refuses_prompts_that_exceed_argv_limit(tmp_path: pathlib.Path):
  provider = _agy(tmp_path)
  huge = "x" * (agy_cli.ARGV_LIMIT_BYTES + 1)

  response = provider.complete(huge, schema.MODEL_OUTPUT_SCHEMA)

  assert response.kind == "provider"
  assert "--max-total-chars" in (response.error or "")
  assert response.duration_s == 0.0  # nothing was spawned


# --- openai --------------------------------------------------------------


class _FakeResponses:

  def __init__(self, reply: Any = None, error: Exception | None = None):
    self.reply = reply
    self.error = error
    self.calls: list[dict[str, Any]] = []

  def create(self, **kwargs: Any) -> Any:
    self.calls.append(kwargs)
    if self.error:
      raise self.error
    return self.reply


class _FakeClient:

  def __init__(self, responses: _FakeResponses):
    self.responses = responses


class _Reply:

  def __init__(self, text: str):
    self.output_text = text
    self.id = "resp_1"
    self.usage = type("Usage", (), {"total_tokens": 42})()


def test_openai_success_sends_strict_schema():
  responses = _FakeResponses(reply=_Reply(json.dumps(OUTPUT)))
  provider = openai_api.OpenAIProvider(
      model="m", effort="medium", client=_FakeClient(responses)
  )

  response = provider.complete("p", schema.MODEL_OUTPUT_SCHEMA)

  assert response.ok and response.parsed == OUTPUT
  assert response.tokens_used == 42
  assert response.meta == {"response_id": "resp_1"}
  assert len(responses.calls) == 1
  call = responses.calls[0]
  assert call["model"] == "m"
  assert call["reasoning"] == {"effort": "medium"}
  assert call["text"]["format"]["strict"] is True
  assert call["text"]["format"]["schema"] == schema.MODEL_OUTPUT_SCHEMA


@pytest.mark.parametrize(
    "error, kind",
    [
        (RuntimeError("Request timeout after 900s"), "timeout"),
        (RuntimeError("HTTP 429"), "provider"),
    ],
)
def test_openai_sdk_errors_become_failed_responses(error: Exception, kind: str):
  provider = openai_api.OpenAIProvider(
      client=_FakeClient(_FakeResponses(error=error))
  )

  response = provider.complete("p", schema.MODEL_OUTPUT_SCHEMA)

  assert response.kind == kind
  assert (response.error or "").startswith("RuntimeError")


def test_openai_bad_reply_is_a_parse_failure():
  provider = openai_api.OpenAIProvider(
      client=_FakeClient(_FakeResponses(reply=_Reply("nope")))
  )

  assert provider.complete("p", {}).kind == "parse"


def test_load_dotenv_parses_quotes_and_respects_existing_env(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  env_file = tmp_path / ".env"
  env_file.write_text(
      "# comment\n\nA_KEY=\"quoted\"\nB_KEY='single'\nC_KEY = spaced \nNOEQ\n"
  )
  monkeypatch.delenv("A_KEY", raising=False)
  monkeypatch.delenv("B_KEY", raising=False)
  monkeypatch.setenv("C_KEY", "already")

  openai_api.load_dotenv(env_file)

  assert os.environ["A_KEY"] == "quoted"
  assert os.environ["B_KEY"] == "single"
  assert os.environ["C_KEY"] == "already"
  for key in ("A_KEY", "B_KEY"):
    monkeypatch.delenv(key)


def test_openai_requires_api_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  monkeypatch.chdir(tmp_path)
  monkeypatch.delenv(openai_api.ENV_API_KEY, raising=False)

  with pytest.raises(base.ProviderUnavailableError, match="OPENAI_API_KEY"):
    openai_api.OpenAIProvider()


# --- factory -------------------------------------------------------------


def test_factory_defaults_and_unknown_names(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  assert factory.default_model("claude-cli") == "claude-opus-5"
  with pytest.raises(ValueError, match="unknown provider 'nope'"):
    factory.default_model("nope")
  with pytest.raises(ValueError, match="unknown provider 'nope'"):
    factory.make_provider("nope", None, "low", 1)

  binary = _fake_binary(tmp_path, "codex", CODEX_SCRIPT)
  monkeypatch.setenv(codex_cli.ENV_BINARY, binary)
  provider = factory.make_provider("codex-cli", None, "low", 1)
  assert isinstance(provider, codex_cli.CodexCliProvider)
  assert provider.model == factory.DEFAULT_MODELS["codex-cli"]
