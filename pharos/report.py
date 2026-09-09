"""Static viewer generation.

A report is a directory that opens from ``file://`` with no server::

    reports/<stamp>_<run_a>[_vs_<run_b>]/
      index.html      the viewer (a copy of viewer.html)
      data/runs.js    results, manifests, frozen prompts, comparison
    reports/_traj/<traj_id>.js   one shared file per trajectory

Trajectory files are generated once and reused by every report; the
viewer loads them lazily by injecting ``<script src>`` tags, which keeps
the index small and works without a web server. The viewer renders all
log and model text through ``textContent``, never ``innerHTML``, so
trajectory content cannot inject markup into the page.
"""

from collections.abc import Sequence
import json
import pathlib
from typing import Any

from pharos import clock
from pharos import compare
from pharos import data
from pharos import store

VIEWER_TEMPLATE = pathlib.Path(__file__).with_name("viewer.html")
TRAJECTORY_DIR = "_traj"
DATA_DIR = "data"
RUNS_JS = "runs.js"
INDEX_HTML = "index.html"
# The viewer resolves trajectory files relative to its own directory.
TRAJECTORY_BASE_URL = f"../{TRAJECTORY_DIR}/"

_REPORT_NAME_MAX_CHARS = 120


def write_trajectory_files(
    traj_root: pathlib.Path, trajectories: Sequence[data.Trajectory]
) -> int:
  """Writes the shared per-trajectory JavaScript files.

  Files that already exist are left alone: trajectories do not change,
  and rewriting hundreds of them per report would dominate build time.

  Args:
    traj_root: The shared ``_traj`` directory.
    trajectories: Trajectories to make available to viewers.

  Returns:
    The number of files newly written.
  """
  traj_root.mkdir(parents=True, exist_ok=True)
  written = 0
  for traj in trajectories:
    path = traj_root / f"{traj.id}.js"
    if path.exists():
      continue
    payload = json.dumps(traj.to_json(), ensure_ascii=False)
    path.write_text(
        "window.__TRAJ__ = window.__TRAJ__ || {};"
        f" window.__TRAJ__[{json.dumps(traj.id)}] = {payload};\n",
        encoding="utf-8",
    )
    written += 1
  return written


def _run_payload(run_dir: pathlib.Path) -> dict[str, Any]:
  """Collects everything the viewer shows for one run."""
  return {
      "run_id": run_dir.name,
      "manifest": store.read_manifest(run_dir),
      "prompt": store.read_prompt(run_dir),
      "results": store.load_results(run_dir),
  }


def _index_entry(traj: data.Trajectory) -> dict[str, Any]:
  """Returns the per-trajectory row data the table needs up front."""
  return {
      "id": traj.id,
      "repo": traj.metadata.get("repo"),
      "instance_id": traj.metadata.get("instance_id"),
      "n": traj.n_messages,
      "n_user_mid": len(traj.mid_trajectory_user_indices()),
  }


def build_report(
    run_dirs: Sequence[pathlib.Path],
    trajectories: Sequence[data.Trajectory],
    reports_root: pathlib.Path,
    out_dir: pathlib.Path | None = None,
) -> pathlib.Path:
  """Builds a viewer for one or two runs.

  Args:
    run_dirs: One run, or two runs to compare side by side.
    trajectories: The trajectories the runs refer to.
    reports_root: Directory that holds all reports and the shared
      trajectory files.
    out_dir: Where to write this report; defaults to a stamped directory
      under ``reports_root``.

  Returns:
    The path of the generated ``index.html``.

  Raises:
    ValueError: If ``run_dirs`` is empty or has more than two entries.
  """
  if not 1 <= len(run_dirs) <= 2:
    raise ValueError("a report covers one or two runs")
  run_dirs = [pathlib.Path(d) for d in run_dirs]
  reports_root.mkdir(parents=True, exist_ok=True)
  write_trajectory_files(reports_root / TRAJECTORY_DIR, trajectories)

  comparison = None
  if len(run_dirs) == 2:
    comparison = compare.compare_runs(run_dirs[0], run_dirs[1])

  if out_dir is None:
    name = "_vs_".join(d.name for d in run_dirs)
    out_dir = reports_root / f"{clock.stamp()}_{name}"[:_REPORT_NAME_MAX_CHARS]
  (out_dir / DATA_DIR).mkdir(parents=True, exist_ok=True)

  payload = {
      "generated_at": clock.now_iso(),
      "runs": [_run_payload(d) for d in run_dirs],
      "compare": comparison,
      "traj_index": [_index_entry(t) for t in trajectories],
      "traj_base": TRAJECTORY_BASE_URL,
  }
  (out_dir / DATA_DIR / RUNS_JS).write_text(
      f"window.__RUNS__ = {json.dumps(payload, ensure_ascii=False)};\n",
      encoding="utf-8",
  )
  index = out_dir / INDEX_HTML
  index.write_text(
      VIEWER_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8"
  )
  return index
