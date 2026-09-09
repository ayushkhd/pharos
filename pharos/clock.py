"""Timestamps in one place, so every persisted record agrees on a format."""

import datetime


def now_iso() -> str:
  """Returns the current local time as ISO 8601 with offset, to the second."""
  return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def stamp() -> str:
  """Returns a filesystem-friendly local timestamp such as 20260909-002312."""
  return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
