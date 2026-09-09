"""Shared pytest fixtures."""

import pathlib

import pytest

from pharos import data
from tests import helpers


@pytest.fixture
def trajectory() -> data.Trajectory:
  """A six-message trajectory with a mid-trajectory user turn."""
  return helpers.make_trajectory()


@pytest.fixture
def dataset_path(tmp_path: pathlib.Path) -> pathlib.Path:
  """A three-trajectory JSONL dataset on disk."""
  trajectories = [
      helpers.make_trajectory("traj_0001"),
      helpers.make_trajectory("traj_0002"),
      helpers.make_trajectory("traj_0003"),
  ]
  return helpers.write_dataset(tmp_path / "dataset.jsonl", trajectories)
