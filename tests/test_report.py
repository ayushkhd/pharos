"""Tests for pharos.report and the viewer's security posture."""

import json
import pathlib

import pytest

from pharos import report
from tests import helpers


def _runs_payload(index: pathlib.Path) -> dict:
  text = (index.parent / "data" / "runs.js").read_text(encoding="utf-8")
  assert text.startswith("window.__RUNS__ = ") and text.endswith(";\n")
  return json.loads(text[len("window.__RUNS__ = ") : -2])


def test_build_report_for_one_run(tmp_path: pathlib.Path):
  trajs = [helpers.make_trajectory("t1"), helpers.make_trajectory("t2")]
  run = helpers.write_run(
      tmp_path,
      "a",
      [helpers.make_result(trajs[0], "bad", [("deception", 6, [5])])],
      manifest={"model": "m"},
  )
  reports_root = tmp_path / "reports"

  index = report.build_report([run], trajs, reports_root)

  assert index.name == "index.html"
  assert index.parent.parent == reports_root
  assert index.parent.name.endswith(f"_{run.name}")
  assert index.read_text(encoding="utf-8") == report.VIEWER_TEMPLATE.read_text(
      encoding="utf-8"
  )
  payload = _runs_payload(index)
  assert payload["compare"] is None
  assert payload["traj_base"] == "../_traj/"
  assert [r["run_id"] for r in payload["runs"]] == [run.name]
  assert payload["runs"][0]["manifest"]["model"] == "m"
  assert payload["runs"][0]["results"]["t1"]["verdict"] == "bad"
  assert payload["traj_index"] == [
      {
          "id": "t1",
          "repo": "example-org/example-repo",
          "instance_id": "t1-instance",
          "n": 6,
          "n_user_mid": 1,
      },
      {
          "id": "t2",
          "repo": "example-org/example-repo",
          "instance_id": "t2-instance",
          "n": 6,
          "n_user_mid": 1,
      },
  ]
  traj_js = (reports_root / "_traj" / "t1.js").read_text(encoding="utf-8")
  assert traj_js.startswith(
      'window.__TRAJ__ = window.__TRAJ__ || {}; window.__TRAJ__["t1"] = '
  )
  assert '"index": 4, "role": "user"' in traj_js


def test_build_report_for_two_runs_embeds_comparison(tmp_path: pathlib.Path):
  traj = helpers.make_trajectory("t1")
  run_a = helpers.write_run(tmp_path, "a", [helpers.make_result(traj, "bad")])
  run_b = helpers.write_run(tmp_path, "b", [helpers.make_result(traj, "clean")])
  out = tmp_path / "custom_out"

  index = report.build_report(
      [run_a, run_b], [traj], tmp_path / "reports", out_dir=out
  )

  assert index == out / "index.html"
  payload = _runs_payload(index)
  assert payload["compare"]["totals"]["only_a"] == 1
  assert payload["compare"]["run_b"]["run_id"] == run_b.name


def test_build_report_rejects_bad_run_counts(tmp_path: pathlib.Path):
  with pytest.raises(ValueError):
    report.build_report([], [], tmp_path)
  with pytest.raises(ValueError):
    report.build_report([tmp_path] * 3, [], tmp_path)


def test_trajectory_files_are_written_once(tmp_path: pathlib.Path):
  trajs = [helpers.make_trajectory("t1")]

  assert report.write_trajectory_files(tmp_path, trajs) == 1
  (tmp_path / "t1.js").write_text("sentinel")
  assert report.write_trajectory_files(tmp_path, trajs) == 0
  assert (tmp_path / "t1.js").read_text(encoding="utf-8") == "sentinel"


def test_viewer_never_uses_inner_html_and_ships_a_csp():
  html = report.VIEWER_TEMPLATE.read_text(encoding="utf-8")

  assert "innerHTML" not in html
  assert "outerHTML" not in html
  assert 'http-equiv="Content-Security-Policy"' in html
  assert "default-src 'none'" in html
  assert "https://" not in html.split("<body")[1]  # no external assets
