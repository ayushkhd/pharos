"""Tests for pharos.store."""

import json
import pathlib

import pytest

from pharos import data
from pharos import render
from pharos import schema
from pharos import store
from pharos import validate
from tests import helpers


def _result(
    traj_id: str, status: str = "ok", verdict: str | None = "clean"
) -> schema.Result:
  traj = helpers.make_trajectory(traj_id)
  parsed = None
  if verdict is not None:
    parsed = {"verdict": verdict, "summary": "", "findings": []}
  return validate.build_result(
      traj,
      run_id="run",
      coverage=render.full_coverage(traj),
      model_info={"provider": "fake", "model": "m", "reasoning_effort": "low"},
      raw_response_path=None,
      parsed=parsed,
      status=status,
  )


def test_new_run_dir_creates_layout(tmp_path: pathlib.Path):
  run_dir = store.new_run_dir(tmp_path / "runs", "smoke")

  assert run_dir.parent == tmp_path / "runs"
  assert run_dir.name.endswith("_smoke")
  assert (run_dir / store.RESULTS_DIR).is_dir()
  assert (run_dir / store.RAW_DIR).is_dir()


def test_atomic_write_leaves_no_temp_files(tmp_path: pathlib.Path):
  target = tmp_path / "nested" / "x.json"

  store.atomic_write_json(target, {"a": [1, 2], "b": "é"})

  assert json.loads(target.read_text(encoding="utf-8")) == {
      "a": [1, 2],
      "b": "é",
  }
  assert [p.name for p in target.parent.iterdir()] == ["x.json"]


def test_atomic_write_cleans_up_when_serialisation_fails(
    tmp_path: pathlib.Path,
):
  target = tmp_path / "x.json"

  with pytest.raises(TypeError):
    store.atomic_write_json(target, {"bad": object()})

  assert not target.exists()
  assert not list(tmp_path.iterdir())


def test_manifest_and_prompt_round_trip(tmp_path: pathlib.Path):
  run_dir = store.new_run_dir(tmp_path, "r")

  assert store.read_manifest(run_dir) == {}
  assert store.read_prompt(run_dir) == ""

  store.write_manifest(run_dir, {"run_id": run_dir.name, "model": "m"})
  store.write_prompt(run_dir, "# Monitor\n")

  assert store.read_manifest(run_dir)["model"] == "m"
  assert store.read_prompt(run_dir) == "# Monitor\n"


def test_results_and_raw_round_trip(tmp_path: pathlib.Path):
  run_dir = store.new_run_dir(tmp_path, "r")

  written = store.write_result(run_dir, _result("traj_b"))
  store.write_result(run_dir, _result("traj_a"))
  raw_rel = store.write_raw(run_dir, "traj_a", {"attempts": []})

  assert written == run_dir / "results" / "traj_b.json"
  assert raw_rel == "raw/traj_a.json"
  assert (run_dir / raw_rel).is_file()
  assert list(store.load_results(run_dir)) == ["traj_a", "traj_b"]


def test_completed_ids_skips_failures(tmp_path: pathlib.Path):
  run_dir = store.new_run_dir(tmp_path, "r")
  store.write_result(run_dir, _result("traj_ok"))
  store.write_result(
      run_dir, _result("traj_warn", status="ok", verdict="catastrophic")
  )
  store.write_result(
      run_dir, _result("traj_fail", status="failed_timeout", verdict=None)
  )

  assert store.completed_ids(run_dir) == {"traj_ok", "traj_warn"}


def test_finalize_writes_jsonl_and_counts(tmp_path: pathlib.Path):
  run_dir = store.new_run_dir(tmp_path, "r")
  store.write_manifest(run_dir, {"run_id": run_dir.name})
  store.write_result(run_dir, _result("traj_2", verdict="bad"))
  store.write_result(run_dir, _result("traj_1"))
  store.write_result(
      run_dir, _result("traj_3", status="failed_parse", verdict=None)
  )

  manifest = store.finalize(run_dir)

  lines = (run_dir / store.RESULTS_JSONL).read_text().splitlines()
  assert [json.loads(l)["traj_id"] for l in lines] == [
      "traj_1",
      "traj_2",
      "traj_3",
  ]
  assert manifest["run_id"] == run_dir.name
  assert manifest["n_results"] == 3
  # A "bad" verdict with no findings is inconsistent, so validation
  # demotes it to ok_with_warnings rather than counting it as clean ok.
  assert manifest["status_counts"] == {
      "ok": 1,
      "ok_with_warnings": 1,
      "failed_parse": 1,
  }
  assert manifest["verdict_counts"] == {"clean": 1, "bad": 1, "none": 1}
  assert manifest["finished_at"]
  assert store.read_manifest(run_dir) == manifest


def test_trajectory_fixture_is_reusable(trajectory: data.Trajectory):
  # Guard against helpers drifting: the store tests rely on this shape.
  assert trajectory.n_messages == 6
