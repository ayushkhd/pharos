"""Comparison of two runs by trajectory id.

Because every provider persists the same ``Result`` shape, LLM versus
regex, low effort versus medium, sample one versus sample two, and
provider X versus provider Y are all the same comparison. The useful
output is the *disagreement set*: trajectories flagged by one run and
not the other are where a human should look first.
"""

import pathlib
from typing import Any, Literal

from pharos import schema
from pharos import store

# A verdict at or above this counts as "flagged".
FLAG_THRESHOLD = "concerning"

Agreement = Literal[
    "both_flag", "only_a", "only_b", "neither", "missing_a", "missing_b"
]
AGREEMENTS: tuple[Agreement, ...] = (
    "both_flag",
    "only_a",
    "only_b",
    "neither",
    "missing_a",
    "missing_b",
)


def is_flagged(
    result: schema.Result | None, threshold: str = FLAG_THRESHOLD
) -> bool:
  """Returns whether a result's verdict reaches ``threshold``."""
  if not result:
    return False
  verdict = result.get("verdict")
  if verdict not in schema.VERDICT_RANK:
    return False
  return schema.VERDICT_RANK[verdict] >= schema.VERDICT_RANK[threshold]


def _agreement(
    a: schema.Result | None, b: schema.Result | None, threshold: str
) -> Agreement:
  """Classifies one trajectory's pair of results.

  Args:
    a: Result from the first run, if any.
    b: Result from the second run, if any.
    threshold: Verdict at or above which a result counts as flagged.

  Returns:
    The agreement class.
  """
  if a is None:
    return "missing_a"
  if b is None:
    return "missing_b"
  flagged_a, flagged_b = is_flagged(a, threshold), is_flagged(b, threshold)
  if flagged_a and flagged_b:
    return "both_flag"
  if flagged_a:
    return "only_a"
  if flagged_b:
    return "only_b"
  return "neither"


def _run_summary(run_dir: pathlib.Path) -> dict[str, Any]:
  """Returns the manifest fields worth carrying into a comparison."""
  manifest = store.read_manifest(run_dir)
  return {
      "run_id": run_dir.name,
      "model": manifest.get("model"),
      "provider": manifest.get("provider"),
      "prompt_path": manifest.get("prompt_path"),
  }


def _row(
    traj_id: str,
    a: schema.Result | None,
    b: schema.Result | None,
    threshold: str,
) -> dict[str, Any]:
  """Builds the per-trajectory comparison row."""
  classes_a = set(a["threat_classes"]) if a else set()
  classes_b = set(b["threat_classes"]) if b else set()
  sev_a = a["max_severity"] if a else 0
  sev_b = b["max_severity"] if b else 0
  return {
      "traj_id": traj_id,
      "agreement": _agreement(a, b, threshold),
      "verdict_a": a["verdict"] if a else None,
      "verdict_b": b["verdict"] if b else None,
      "max_sev_a": sev_a,
      "max_sev_b": sev_b,
      "sev_delta": sev_b - sev_a,
      "classes_a": sorted(classes_a),
      "classes_b": sorted(classes_b),
      "classes_only_a": sorted(classes_a - classes_b),
      "classes_only_b": sorted(classes_b - classes_a),
      "status_a": a["execution_status"] if a else None,
      "status_b": b["execution_status"] if b else None,
  }


def compare_runs(
    a_dir: pathlib.Path,
    b_dir: pathlib.Path,
    threshold: str = FLAG_THRESHOLD,
) -> dict[str, Any]:
  """Joins two runs by trajectory id.

  Args:
    a_dir: First run directory.
    b_dir: Second run directory.
    threshold: Verdict at or above which a result counts as flagged.

  Returns:
    A JSON-serialisable report with ``run_a``, ``run_b``,
    ``flag_threshold``, ``totals`` (a count per agreement value) and
    ``rows`` (one per trajectory present in either run, sorted by id).
  """
  results_a = store.load_results(a_dir)
  results_b = store.load_results(b_dir)
  totals: dict[str, int] = {agreement: 0 for agreement in AGREEMENTS}
  rows = []
  for traj_id in sorted(set(results_a) | set(results_b)):
    row = _row(
        traj_id, results_a.get(traj_id), results_b.get(traj_id), threshold
    )
    totals[row["agreement"]] += 1
    rows.append(row)
  return {
      "run_a": _run_summary(pathlib.Path(a_dir)),
      "run_b": _run_summary(pathlib.Path(b_dir)),
      "flag_threshold": threshold,
      "totals": totals,
      "rows": rows,
  }
