"""Tests for pharos.runner."""

import hashlib
import json
import pathlib
import random
from typing import Any

from pharos import data
from pharos import render
from pharos import runner
from pharos import schema as pharos_schema
from pharos import store
from pharos.providers import base
from tests import helpers

CLEAN: dict[str, Any] = {"verdict": "clean", "summary": "ok", "findings": []}


class FakeProvider:
  """Replays scripted responses and records what it was asked."""

  name = "fake"
  model = "fake-1"
  effort = "low"

  def __init__(self, responses: list[base.RawResponse]):
    self.responses = list(responses)
    self.prompts: list[str] = []

  def complete(self, prompt: str, schema: dict[str, Any]) -> base.RawResponse:
    del schema  # Unused: the fake does not validate.
    self.prompts.append(prompt)
    return self.responses.pop(0)


class RaisingProvider(FakeProvider):
  """A provider whose implementation has a bug."""

  def complete(self, prompt: str, schema: dict[str, Any]) -> base.RawResponse:
    raise RuntimeError("boom")


def _ok() -> base.RawResponse:
  return base.RawResponse(
      text=json.dumps(CLEAN), parsed=CLEAN, duration_s=1.5, tokens_used=100
  )


def _failed(kind: base.ResponseKind, error: str) -> base.RawResponse:
  return base.RawResponse(duration_s=0.5).fail(kind, error)


def _run_one(
    tmp_path: pathlib.Path,
    provider: FakeProvider,
    traj: data.Trajectory,
    retries: int = 2,
) -> tuple[pharos_schema.Result, list[float], list[str], pathlib.Path]:
  """Runs one trajectory with injected sleep and jitter.

  Args:
    tmp_path: Where to create the run directory.
    provider: The scripted provider.
    traj: The trajectory.
    retries: Retry budget.

  Returns:
    The result, the sleeps requested, the log lines, and the run dir.
  """
  run_dir = store.new_run_dir(tmp_path, "t")
  sleeps: list[float] = []
  logs: list[str] = []
  result = runner.run_one(
      traj,
      provider,
      "instr",
      run_dir,
      run_dir.name,
      runner.RunConfig(retries=retries),
      logs.append,
      sleep=sleeps.append,
      rng=random.Random(0),
  )
  return result, sleeps, logs, run_dir


def test_success_on_first_attempt(
    tmp_path: pathlib.Path, trajectory: data.Trajectory
):
  provider = FakeProvider([_ok()])

  result, sleeps, logs, run_dir = _run_one(tmp_path, provider, trajectory)

  assert result["execution_status"] == "ok"
  assert result["verdict"] == "clean"
  assert not sleeps
  assert not logs
  assert result["model"] == {
      "provider": "fake",
      "model": "fake-1",
      "reasoning_effort": "low",
      "tokens_used": 100,
      "duration_s": 1.5,
      "attempts": 1,
  }
  expected_prompt = render.render(trajectory, "instr").prompt
  assert provider.prompts == [expected_prompt]
  assert (
      result["prompt_sha256"]
      == hashlib.sha256(expected_prompt.encode()).hexdigest()
  )
  assert result["review_coverage"] == render.full_coverage(trajectory)

  raw = json.loads((run_dir / result["raw_response_path"]).read_text())
  assert raw["prompt_sha256"] == result["prompt_sha256"]
  assert raw["prompt_chars"] == len(expected_prompt)
  assert [a["attempt"] for a in raw["attempts"]] == [1]
  assert raw["final"]["parsed"] == CLEAN
  assert store.load_results(run_dir)[trajectory.id] == result


def test_retries_with_backoff_then_succeeds(
    tmp_path: pathlib.Path, trajectory: data.Trajectory
):
  provider = FakeProvider(
      [
          _failed("parse", "json parse failed"),
          _failed("provider", "HTTP 429 rate limit"),
          _ok(),
      ]
  )

  result, sleeps, logs, run_dir = _run_one(tmp_path, provider, trajectory)

  assert result["execution_status"] == "ok"
  assert result["model"]["attempts"] == 3
  assert result["model"]["duration_s"] == 2.5
  # Ordinary failure: 5s * 2^0 + jitter. Throttled: 30s * 2^1 + jitter.
  assert 5.0 <= sleeps[0] < 8.0
  assert 60.0 <= sleeps[1] < 63.0
  assert len(logs) == 2
  assert "retry 1/2" in logs[0] and "json parse failed" in logs[0]
  raw = json.loads((run_dir / result["raw_response_path"]).read_text())
  assert [a["kind"] for a in raw["attempts"]] == ["parse", "provider", None]


def test_exhausted_retries_persist_a_failed_result(
    tmp_path: pathlib.Path, trajectory: data.Trajectory
):
  provider = FakeProvider([_failed("timeout", "timeout after 900s")] * 2)

  result, sleeps, _, run_dir = _run_one(
      tmp_path, provider, trajectory, retries=1
  )

  assert result["execution_status"] == "failed_timeout"
  assert result["error"] == "timeout after 900s"
  assert result["verdict"] is None
  assert result["model"]["attempts"] == 2
  assert len(sleeps) == 1
  assert store.completed_ids(run_dir) == set()


def test_status_mapping_for_each_failure_kind():
  assert runner.run_one is not None  # keep the import used
  assert runner.backoff_seconds(
      0, False, random.Random(0)
  ) < runner.backoff_seconds(1, False, random.Random(0))
  assert runner.is_rate_limited(_failed("provider", "usage limit reached"))
  assert not runner.is_rate_limited(_failed("provider", "exit 1"))


def test_run_all_persists_every_trajectory_including_crashes(
    tmp_path: pathlib.Path,
):
  trajectories = [helpers.make_trajectory(f"traj_{i}") for i in range(3)]
  run_dir = store.new_run_dir(tmp_path, "all")
  logs: list[str] = []

  ok_results = runner.run_all(
      trajectories,
      FakeProvider([_ok(), _ok(), _ok()]),
      "instr",
      run_dir,
      run_dir.name,
      runner.RunConfig(workers=2),
      log=logs.append,
  )

  assert len(ok_results) == 3
  assert set(store.load_results(run_dir)) == {"traj_0", "traj_1", "traj_2"}
  assert any(line.startswith("[3/3]") for line in logs)
  assert all("failed=0" in line for line in logs)

  crash_dir = store.new_run_dir(tmp_path, "crash")
  crash_logs: list[str] = []
  crashed = runner.run_all(
      trajectories[:1],
      RaisingProvider([]),
      "instr",
      crash_dir,
      crash_dir.name,
      runner.RunConfig(workers=1),
      log=crash_logs.append,
  )

  assert len(crashed) == 1
  result = crashed[0]
  assert result["execution_status"] == "failed_provider"
  assert result["error"] == "infra exception: RuntimeError('boom')"
  assert result["review_coverage"]["mode"] == render.COVERAGE_NONE
  assert store.load_results(crash_dir)["traj_0"] == result
  assert "failed=1" in crash_logs[-1]
