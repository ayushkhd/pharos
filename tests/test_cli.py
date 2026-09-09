"""End-to-end tests for the command-line interface on a tiny dataset."""

import json
import pathlib
from typing import Any

import pytest

from pharos import cli
from pharos import store
from pharos.providers import base
from pharos.providers import factory
from tests import helpers

BAD: dict[str, Any] = {
    "verdict": "bad",
    "summary": "leaked",
    "findings": [
        {
            "category": "credential_leakage",
            "evidence_indices": [4],
            "quotes": [{"index": 4, "quote": "AKIAIOSFODNN7EXAMPLE"}],
            "observed_behavior": "wrote the key",
            "severity": 7,
            "limitations": [],
        }
    ],
}


class StubProvider:
  """Answers every prompt with the same verdict and counts calls."""

  name = "stub"
  model = "stub-1"
  effort = "low"
  calls = 0

  def complete(self, prompt: str, schema: dict[str, Any]) -> base.RawResponse:
    del prompt, schema
    StubProvider.calls += 1
    return base.RawResponse(text=json.dumps(BAD), parsed=BAD, duration_s=0.1)


def _install_stub_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> type[StubProvider]:
  """Routes factory.make_provider to StubProvider for one test."""
  StubProvider.calls = 0
  monkeypatch.setattr(factory, "make_provider", lambda *a, **k: StubProvider())
  return StubProvider


def _only_run(runs_dir: pathlib.Path) -> pathlib.Path:
  (run_dir,) = [p for p in runs_dir.iterdir() if p.is_dir()]
  return run_dir


def test_run_persists_manifest_and_results_and_resumes(
    tmp_path: pathlib.Path,
    dataset_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
  stub_provider = _install_stub_provider(monkeypatch)
  prompt = tmp_path / "p.md"
  prompt.write_text("# Monitor\n", encoding="utf-8")
  runs_dir = tmp_path / "runs"
  argv = [
      "run",
      "--dataset",
      str(dataset_path),
      "--prompt",
      str(prompt),
      "--name",
      "smoke",
      "--runs-dir",
      str(runs_dir),
      "--workers",
      "2",
      "--limit",
      "2",
  ]

  assert cli.main(argv) == 0

  run_dir = _only_run(runs_dir)
  manifest = store.read_manifest(run_dir)
  assert manifest["provider"] == "stub" and manifest["model"] == "stub-1"
  assert manifest["n_selected"] == 2
  assert manifest["ids"] == ["traj_0001", "traj_0002"]
  assert manifest["n_results"] == 2
  assert manifest["verdict_counts"] == {"bad": 2}
  assert manifest["prompt_sha256"] and manifest["dataset_sha256"]
  assert store.read_prompt(run_dir) == "# Monitor\n"
  results = store.load_results(run_dir)
  assert results["traj_0001"]["findings"][0]["evidence"][0]["role"] == "user"
  summary = json.loads(capsys.readouterr().out)
  assert summary["run_id"] == run_dir.name
  assert stub_provider.calls == 2

  # Resuming a complete run makes no provider calls and keeps the manifest.
  assert cli.main(argv + ["--resume", str(run_dir)]) == 0
  assert stub_provider.calls == 2
  assert store.read_manifest(run_dir)["n_results"] == 2


def test_heuristics_compare_report_validate_round_trip(
    tmp_path: pathlib.Path,
    dataset_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
):
  runs_dir = tmp_path / "runs"
  common = ["--dataset", str(dataset_path), "--runs-dir", str(runs_dir)]

  assert cli.main(["heuristics", *common, "--name", "rx_all"]) == 0
  assert (
      cli.main(
          [
              "heuristics",
              *common,
              "--name",
              "rx_destr",
              "--rules",
              "destructive",
          ]
      )
      == 0
  )
  run_all, run_destr = sorted(p for p in runs_dir.iterdir() if p.is_dir())
  capsys.readouterr()  # discard the two heuristics summaries
  assert store.read_manifest(run_all)["verdict_counts"] == {"concerning": 3}
  assert store.read_manifest(run_destr)["verdict_counts"] == {"clean": 3}

  assert (
      cli.main(["compare", "--run", str(run_all), "--run", str(run_destr)]) == 0
  )
  totals = json.loads(capsys.readouterr().out)
  assert totals["only_a"] == 3
  assert (run_destr / f"compare_vs_{run_all.name}.json").is_file()

  out = tmp_path / "report"
  assert (
      cli.main(
          [
              "report",
              "--dataset",
              str(dataset_path),
              "--run",
              str(run_all),
              "--run",
              str(run_destr),
              "--reports-dir",
              str(tmp_path / "reports"),
              "--out",
              str(out),
          ]
      )
      == 0
  )
  assert (out / "index.html").is_file()
  assert (tmp_path / "reports" / "_traj" / "traj_0002.js").is_file()
  assert capsys.readouterr().out.strip() == str(out / "index.html")

  before = store.load_results(run_all)
  assert (
      cli.main(
          ["validate", "--dataset", str(dataset_path), "--run", str(run_all)]
      )
      == 0
  )
  after = store.load_results(run_all)
  for traj_id, result in after.items():
    assert result["findings"] == before[traj_id]["findings"]
    assert result["verdict"] == before[traj_id]["verdict"]


def test_labels_dossier_and_metrics(
    tmp_path: pathlib.Path,
    dataset_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
):
  runs_dir = tmp_path / "runs"
  common = ["--dataset", str(dataset_path)]
  assert (
      cli.main(
          [
              "heuristics",
              *common,
              "--runs-dir",
              str(runs_dir),
              "--name",
              "rx",
          ]
      )
      == 0
  )
  run_dir = _only_run(runs_dir)
  capsys.readouterr()  # discard the heuristics summary
  ids_file = tmp_path / "ids.txt"
  ids_file.write_text("# pool\ntraj_0001\ntraj_0003\n")

  labels = tmp_path / "labels.json"
  assert (
      cli.main(
          [
              "labels",
              *common,
              "--ids-file",
              str(ids_file),
              "--run",
              str(run_dir),
              "--out",
              str(labels),
          ]
      )
      == 0
  )
  document = json.loads(labels.read_text())
  assert [e["traj_id"] for e in document["entries"]] == [
      "traj_0001",
      "traj_0003",
  ]
  assert document["entries"][0]["label"] is None

  derived = tmp_path / "derived.json"
  assert (
      cli.main(
          [
              "labels",
              *common,
              "--out",
              str(derived),
              "--from-run",
              str(run_dir),
              "--leak-min-sev",
              "6",
          ]
      )
      == 0
  )
  by_id = {e["traj_id"]: e for e in json.loads(derived.read_text())["entries"]}
  assert by_id["traj_0002"]["label"] == "leak"  # regex sev 6 >= 6

  dossiers = tmp_path / "dossiers"
  assert (
      cli.main(
          [
              "dossier",
              *common,
              "--ids",
              "traj_0002",
              "--run",
              str(run_dir),
              "--out",
              str(dossiers),
          ]
      )
      == 0
  )
  assert (dossiers / "traj_0002.md").read_text().startswith("# traj_0002")

  rows_out = tmp_path / "rows.json"
  assert (
      cli.main(
          [
              "metrics",
              "--labels",
              str(derived),
              "--run",
              str(run_dir),
              "--thresholds",
              "1,6",
              "--out",
              str(rows_out),
          ]
      )
      == 0
  )
  table = capsys.readouterr().out
  assert table.startswith("labeled: 3/3")
  rows = json.loads(rows_out.read_text())
  assert {r["min_sev"] for r in rows} == {1, 6}


def test_user_errors_exit_2_with_a_message(
    tmp_path: pathlib.Path,
    dataset_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
):
  status = cli.main(
      [
          "heuristics",
          "--dataset",
          str(dataset_path),
          "--name",
          "x",
          "--runs-dir",
          str(tmp_path),
          "--ids",
          "traj_0001,traj_nope",
      ]
  )
  assert status == 2
  assert "unknown trajectory ids: ['traj_nope']" in capsys.readouterr().err

  status = cli.main(["compare", "--run", "a"])
  assert status == 2
  assert "exactly two --run" in capsys.readouterr().err


def test_run_reports_unavailable_provider(
    tmp_path: pathlib.Path,
    dataset_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
  def unavailable(*_: Any, **__: Any) -> base.Provider:
    raise base.ProviderUnavailableError("codex binary not found")

  monkeypatch.setattr(factory, "make_provider", unavailable)
  prompt = tmp_path / "p.md"
  prompt.write_text("x")

  status = cli.main(
      [
          "run",
          "--dataset",
          str(dataset_path),
          "--prompt",
          str(prompt),
          "--name",
          "n",
          "--runs-dir",
          str(tmp_path),
      ]
  )

  assert status == 2
  assert "codex binary not found" in capsys.readouterr().err
  assert not any(p.is_dir() for p in tmp_path.iterdir() if p.name != "p.md")


def test_module_entry_point_parses_help(capsys: pytest.CaptureFixture[str]):
  with pytest.raises(SystemExit) as exc_info:
    cli.main(["--help"])
  assert exc_info.value.code == 0
  assert "heuristics" in capsys.readouterr().out
  assert helpers.make_trajectory().id == "traj_0001"
