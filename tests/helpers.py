"""Builders for synthetic trajectories used across the test-suite."""

import json
import pathlib
from typing import Any

from pharos import data

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
