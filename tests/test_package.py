"""Sanity checks on package metadata."""

import pathlib
import tomllib

import pharos


def test_version_matches_pyproject():
  pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
  with pyproject.open("rb") as f:
    declared = tomllib.load(f)["project"]["version"]
  assert pharos.__version__ == declared
