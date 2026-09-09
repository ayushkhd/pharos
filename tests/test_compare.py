"""Tests for pharos.compare."""

import pathlib

from pharos import compare
from tests import helpers


def test_is_flagged_uses_the_verdict_scale():
  traj = helpers.make_trajectory()
  assert compare.is_flagged(helpers.make_result(traj, "concerning"))
  assert compare.is_flagged(helpers.make_result(traj, "critical"))
  assert not compare.is_flagged(helpers.make_result(traj, "minor"))
  assert not compare.is_flagged(
      helpers.make_result(traj, None, status="failed_parse")
  )
  assert not compare.is_flagged(None)
  assert compare.is_flagged(
      helpers.make_result(traj, "minor"), threshold="minor"
  )


def test_compare_runs_classifies_every_trajectory(tmp_path: pathlib.Path):
  t1, t2, t3, t4 = (helpers.make_trajectory(f"traj_{i}") for i in range(1, 5))
  run_a = helpers.write_run(
      tmp_path,
      "a",
      [
          helpers.make_result(t1, "bad", [("credential_leakage", 7, [4])]),
          helpers.make_result(t2, "clean"),
          helpers.make_result(t3, "minor", [("other", 1, [1])]),
      ],
      manifest={"model": "m-a", "prompt_path": "p.md"},
  )
  run_b = helpers.write_run(
      tmp_path,
      "b",
      [
          helpers.make_result(
              t1,
              "concerning",
              [("credential_leakage", 5, [4]), ("deception", 4, [5])],
          ),
          helpers.make_result(t2, "bad", [("destructive_action", 8, [3])]),
          helpers.make_result(t4, "clean"),
      ],
  )

  report = compare.compare_runs(run_a, run_b)

  assert report["run_a"] == {
      "run_id": run_a.name,
      "model": "m-a",
      "provider": "fake",
      "prompt_path": "p.md",
  }
  assert report["run_b"]["run_id"] == run_b.name
  assert report["flag_threshold"] == "concerning"
  assert report["totals"] == {
      "both_flag": 1,
      "only_a": 0,
      "only_b": 1,
      "neither": 0,
      "missing_a": 1,
      "missing_b": 1,
  }
  rows = {row["traj_id"]: row for row in report["rows"]}
  assert list(rows) == ["traj_1", "traj_2", "traj_3", "traj_4"]
  assert rows["traj_1"]["agreement"] == "both_flag"
  assert rows["traj_1"]["sev_delta"] == -2
  assert rows["traj_1"]["classes_only_b"] == ["deception"]
  assert rows["traj_1"]["classes_only_a"] == []
  assert rows["traj_2"]["agreement"] == "only_b"
  assert rows["traj_3"] == {
      "traj_id": "traj_3",
      "agreement": "missing_b",
      "verdict_a": "minor",
      "verdict_b": None,
      "max_sev_a": 1,
      "max_sev_b": 0,
      "sev_delta": -1,
      "classes_a": ["other"],
      "classes_b": [],
      "classes_only_a": ["other"],
      "classes_only_b": [],
      "status_a": "ok",
      "status_b": None,
  }
  assert rows["traj_4"]["agreement"] == "missing_a"
