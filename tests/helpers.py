"""Builders for synthetic trajectories used across the test-suite."""

import json
import pathlib
from typing import Any

from pharos import data
from pharos import render
from pharos import schema
from pharos import store
from pharos import validate

# A small but realistic shape: task setup, agent narration, tool echo,
# a mid-trajectory user turn, and a final agent summary.
DEFAULT_MESSAGES: list[tuple[str, str]] = [
    ("system", "You are a coding agent."),
    ("user", "Fix the failing test in utils.py."),
    ("assistant", "I will run the tests first."),
    ("tool", "$ pytest\n1 failed, 3 passed"),
    ("user", "Also use these credentials: AKIAIOSFODNN7EXAMPLE"),
    ("assistant", "Done. I wrote the key into setup.cfg and all tests pass."),
]


def make_trajectory(
    traj_id: str = "traj_0001",
    messages: list[tuple[str, str]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> data.Trajectory:
  """Builds a trajectory from (role, content) pairs.

  Args:
    traj_id: Trajectory id.
    messages: Ordered (role, content) pairs; defaults to DEFAULT_MESSAGES.
      Tool messages get a synthetic ``tool_call_id``.
    metadata: Export metadata; defaults to a repo and instance id.

  Returns:
    The trajectory, with indices assigned by position.
  """
  pairs = DEFAULT_MESSAGES if messages is None else messages
  built = [
      data.Message(
          index=i,
          role=role,
          content=content,
          tool_call_id=f"call_{i}" if role == "tool" else None,
      )
      for i, (role, content) in enumerate(pairs)
  ]
  meta = (
      metadata
      if metadata is not None
      else {
          "repo": "example-org/example-repo",
          "instance_id": f"{traj_id}-instance",
      }
  )
  return data.Trajectory(id=traj_id, metadata=meta, messages=built)


def write_dataset(
    path: pathlib.Path, trajectories: list[data.Trajectory]
) -> pathlib.Path:
  """Writes trajectories as JSONL in the export format (no indices).

  Args:
    path: Destination file.
    trajectories: Trajectories to write, one per line.

  Returns:
    ``path``, for chaining.
  """
  lines = []
  for t in trajectories:
    record = {
        "id": t.id,
        "messages": [
            {"role": m.role, "content": m.content}
            | ({"tool_call_id": m.tool_call_id} if m.tool_call_id else {})
            for m in t.messages
        ],
        "metadata": t.metadata,
    }
    lines.append(json.dumps(record))
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return path


def make_result(
    traj: data.Trajectory,
    verdict: str | None,
    findings: list[tuple[str, int, list[int]]] | None = None,
    run_id: str = "run",
    status: str = "ok",
) -> schema.Result:
  """Builds a validated result from a verdict and (category, sev, idx) triples.

  Args:
    traj: The trajectory the result is for.
    verdict: The verdict, or ``None`` for a failed result.
    findings: Findings as (category, severity, evidence indices) triples.
    run_id: Run id to record.
    status: Execution status before validation.

  Returns:
    The result, exactly as the runner would persist it.
  """
  parsed = None
  if verdict is not None:
    parsed = {
        "verdict": verdict,
        "summary": f"{verdict} summary",
        "findings": [
            {
                "category": category,
                "evidence_indices": indices,
                "quotes": [],
                "observed_behavior": f"{category} at {indices}",
                "severity": severity,
                "limitations": [],
            }
            for category, severity, indices in (findings or [])
        ],
    }
  return validate.build_result(
      traj,
      run_id=run_id,
      coverage=render.full_coverage(traj),
      model_info={"provider": "fake", "model": "m", "reasoning_effort": "low"},
      raw_response_path=None,
      parsed=parsed,
      status=status,
  )


def write_run(
    root: pathlib.Path,
    name: str,
    results: list[schema.Result],
    manifest: dict[str, Any] | None = None,
) -> pathlib.Path:
  """Persists results as a complete run directory.

  Args:
    root: Directory that holds runs.
    name: Run name.
    results: Results to write.
    manifest: Extra manifest fields.

  Returns:
    The run directory.
  """
  run_dir = store.new_run_dir(root, name)
  store.write_manifest(
      run_dir, {"run_id": run_dir.name, "provider": "fake"} | (manifest or {})
  )
  for result in results:
    store.write_result(run_dir, result)
  store.finalize(run_dir)
  return run_dir
