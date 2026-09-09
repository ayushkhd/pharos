"""Tests for pharos.data."""

import hashlib
import pathlib

import pytest

from pharos import data
from tests import helpers


def test_load_preserves_order_indices_roles_and_tool_call_ids(
    dataset_path: pathlib.Path,
):
  trajectories = data.load_trajectories(dataset_path)

  assert [t.id for t in trajectories] == ["traj_0001", "traj_0002", "traj_0003"]
  first = trajectories[0]
  assert [m.index for m in first.messages] == list(range(6))
  assert [m.role for m in first.messages] == [
      "system",
      "user",
      "assistant",
      "tool",
      "user",
      "assistant",
  ]
  assert first.messages[3].tool_call_id == "call_3"
  assert first.messages[2].tool_call_id is None
  assert first.metadata["repo"] == "example-org/example-repo"


def test_load_skips_blank_lines(tmp_path: pathlib.Path):
  path = helpers.write_dataset(
      tmp_path / "d.jsonl", [helpers.make_trajectory()]
  )
  path.write_text("\n" + path.read_text() + "\n\n", encoding="utf-8")

  assert len(data.load_trajectories(path)) == 1


@pytest.mark.parametrize(
    "line, message",
    [
        ("not json", "line 1: invalid JSON"),
        ('{"messages": []}', "missing or non-string 'id'"),
        ('{"id": "t", "messages": "nope"}', "'messages' must be a list"),
        ("[1, 2]", "expected a JSON object"),
    ],
)
def test_load_rejects_malformed_records(
    tmp_path: pathlib.Path, line: str, message: str
):
  path = tmp_path / "bad.jsonl"
  path.write_text(line + "\n", encoding="utf-8")

  with pytest.raises(data.DatasetError, match=message):
    data.load_trajectories(path)


def test_origin_labels():
  assert data.origin_of("assistant") == "agent-authored"
  assert data.origin_of("tool") == "environment-output"
  assert data.origin_of("user") == "task-or-user"
  assert data.origin_of("system") == "task-or-user"
  assert data.origin_of("weird") == data.UNKNOWN_ORIGIN
  assert data.Message(0, "tool", "x").origin == "environment-output"


def test_trajectory_accessors(trajectory: data.Trajectory):
  assert trajectory.n_messages == 6
  assert trajectory.total_chars == sum(
      len(m.content) for m in trajectory.messages
  )
  assert trajectory.message(3).role == "tool"
  # Message at index 1 is the task statement; index 4 is the injected turn.
  assert trajectory.mid_trajectory_user_indices() == [4]
  with pytest.raises(IndexError):
    trajectory.message(6)
  with pytest.raises(IndexError):
    trajectory.message(-1)


def test_to_json_round_trips_through_loader(
    tmp_path: pathlib.Path, trajectory: data.Trajectory
):
  path = helpers.write_dataset(tmp_path / "d.jsonl", [trajectory])
  reloaded = data.load_trajectories(path)[0]

  assert reloaded.to_json() == trajectory.to_json()


def test_select_by_ids_keeps_dataset_order(dataset_path: pathlib.Path):
  trajectories = data.load_trajectories(dataset_path)

  chosen = data.select(trajectories, ids=["traj_0003", "traj_0001"])

  assert [t.id for t in chosen] == ["traj_0001", "traj_0003"]


def test_select_limit_applies_after_id_filter(dataset_path: pathlib.Path):
  trajectories = data.load_trajectories(dataset_path)

  assert [t.id for t in data.select(trajectories, limit=2)] == [
      "traj_0001",
      "traj_0002",
  ]
  assert [
      t.id
      for t in data.select(
          trajectories, ids=["traj_0002", "traj_0003"], limit=1
      )
  ] == ["traj_0002"]


def test_select_unknown_id_raises(dataset_path: pathlib.Path):
  trajectories = data.load_trajectories(dataset_path)

  with pytest.raises(data.UnknownTrajectoryIdError) as exc_info:
    data.select(trajectories, ids=["traj_0001", "traj_zzz", "traj_yyy"])

  assert exc_info.value.missing == ["traj_yyy", "traj_zzz"]


def test_read_id_file_ignores_comments_and_blanks(tmp_path: pathlib.Path):
  path = tmp_path / "ids.txt"
  path.write_text("# pooled from run A\n\ntraj_a\n  traj_b  \n#traj_c\n")

  assert data.read_id_file(path) == ["traj_a", "traj_b"]


def test_dataset_sha256_matches_hashlib(dataset_path: pathlib.Path):
  expected = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

  assert data.dataset_sha256(dataset_path) == expected
