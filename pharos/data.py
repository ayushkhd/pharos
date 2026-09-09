"""Trajectory loading.

Trajectories are read from JSONL exactly as exported: ids, message order,
roles and content are preserved verbatim. The only thing this module adds
is the message index (its position in the array), which is the citation
key used by every other module and by the viewer.
"""

from collections.abc import Iterable
import dataclasses
import hashlib
import json
import pathlib
from typing import Any

# Role -> origin label. Origin answers "who produced this text?", which
# matters because tool-call arguments are typically stripped from exported
# logs: what the agent *did* is evidenced by tool output, and what it
# *claimed* by assistant text.
ORIGIN_BY_ROLE: dict[str, str] = {
    "assistant": "agent-authored",
    "tool": "environment-output",
    "user": "task-or-user",
    "system": "task-or-user",
}
UNKNOWN_ORIGIN = "unknown"

# The first messages of a trajectory are the task setup (system prompt and
# the initial task statement). A user message after them is a
# mid-trajectory turn, which is where injected scenarios usually begin.
TASK_SETUP_MESSAGES = 2

_HASH_CHUNK_BYTES = 1 << 20


class DatasetError(ValueError):
  """A JSONL record could not be turned into a Trajectory."""


class UnknownTrajectoryIdError(ValueError):
  """A requested trajectory id is not present in the dataset.

  Attributes:
    missing: The ids that were requested but not found, sorted.
  """

  def __init__(self, missing: Iterable[str]):
    self.missing = sorted(missing)
    super().__init__(f"unknown trajectory ids: {self.missing}")


def origin_of(role: str) -> str:
  """Returns the provenance label for a message role."""
  return ORIGIN_BY_ROLE.get(role, UNKNOWN_ORIGIN)


@dataclasses.dataclass(frozen=True)
class Message:
  """One message of a trajectory.

  Attributes:
    index: Position in the trajectory's message array. This is the
      citation key used in prompts, results and the viewer.
    role: One of ``system``, ``user``, ``assistant`` or ``tool``.
    content: Message text, verbatim.
    tool_call_id: Identifier linking a tool message to the call that
      produced it, when the export carries one.
  """

  index: int
  role: str
  content: str
  tool_call_id: str | None = None

  @property
  def origin(self) -> str:
    """Returns the provenance label for this message's role."""
    return origin_of(self.role)


@dataclasses.dataclass
class Trajectory:
  """A complete coding-agent log.

  Attributes:
    id: Stable identifier, used as the join key across runs.
    metadata: Free-form export metadata such as ``repo`` and
      ``instance_id``.
    messages: Messages in original order; ``messages[i].index == i``.
  """

  id: str
  metadata: dict[str, Any]
  messages: list[Message] = dataclasses.field(default_factory=list)

  @property
  def n_messages(self) -> int:
    """Returns the number of messages."""
    return len(self.messages)

  @property
  def total_chars(self) -> int:
    """Returns the character count across all messages."""
    return sum(len(m.content) for m in self.messages)

  def message(self, index: int) -> Message:
    """Returns the message at a citation index.

    Args:
      index: Zero-based message index.

    Raises:
      IndexError: If ``index`` is outside ``[0, n_messages)``.
    """
    if not 0 <= index < len(self.messages):
      raise IndexError(
          f"message index {index} out of range for {self.id}"
          f" (n_messages={len(self.messages)})"
      )
    return self.messages[index]

  def mid_trajectory_user_indices(self) -> list[int]:
    """Returns indices of user messages that follow the task setup."""
    return [
        m.index for m in self.messages[TASK_SETUP_MESSAGES:] if m.role == "user"
    ]

  def to_json(self) -> dict[str, Any]:
    """Returns a JSON-serialisable form, including message indices."""
    return {
        "id": self.id,
        "metadata": self.metadata,
        "messages": [
            {
                "index": m.index,
                "role": m.role,
                "content": m.content,
                "tool_call_id": m.tool_call_id,
            }
            for m in self.messages
        ],
    }


def _parse_record(record: Any, line_no: int) -> Trajectory:
  """Builds a Trajectory from one decoded JSONL record.

  Args:
    record: The decoded JSON value for one line.
    line_no: One-based line number, for error messages.

  Returns:
    The parsed trajectory.

  Raises:
    DatasetError: If the record lacks an ``id`` or a ``messages`` list.
  """
  if not isinstance(record, dict):
    raise DatasetError(f"line {line_no}: expected a JSON object")
  traj_id = record.get("id")
  raw_messages = record.get("messages")
  if not isinstance(traj_id, str) or not traj_id:
    raise DatasetError(f"line {line_no}: missing or non-string 'id'")
  if not isinstance(raw_messages, list):
    raise DatasetError(f"line {line_no} ({traj_id}): 'messages' must be a list")
  messages = [
      Message(
          index=i,
          role=str(m.get("role")),
          content=str(m.get("content", "")),
          tool_call_id=m.get("tool_call_id"),
      )
      for i, m in enumerate(raw_messages)
  ]
  metadata = record.get("metadata") or {}
  return Trajectory(id=traj_id, metadata=metadata, messages=messages)


def load_trajectories(path: str | pathlib.Path) -> list[Trajectory]:
  """Loads every trajectory from a JSONL file.

  Blank lines are skipped. Order is preserved.

  Args:
    path: Path to a file with one JSON object per line.

  Returns:
    Trajectories in file order.

  Raises:
    DatasetError: If a line is not valid JSON or is not a trajectory.
  """
  out: list[Trajectory] = []
  with open(path, encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        record = json.loads(line)
      except json.JSONDecodeError as e:
        raise DatasetError(f"line {line_no}: invalid JSON: {e}") from e
      out.append(_parse_record(record, line_no))
  return out


def dataset_sha256(path: str | pathlib.Path) -> str:
  """Returns the hex SHA-256 of a file, read in chunks."""
  digest = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(_HASH_CHUNK_BYTES), b""):
      digest.update(chunk)
  return digest.hexdigest()


def select(
    trajectories: list[Trajectory],
    ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[Trajectory]:
  """Narrows a trajectory list by id and then by count.

  Args:
    trajectories: The full list, in dataset order.
    ids: If given, keep only these ids. Order of the result follows the
      dataset, not ``ids``.
    limit: If given and positive, keep at most this many after the id
      filter.

  Returns:
    The selected trajectories.

  Raises:
    UnknownTrajectoryIdError: If any requested id is absent.
  """
  if ids is not None:
    wanted = set(ids)
    trajectories = [t for t in trajectories if t.id in wanted]
    missing = wanted - {t.id for t in trajectories}
    if missing:
      raise UnknownTrajectoryIdError(missing)
  if limit:
    trajectories = trajectories[:limit]
  return trajectories


def read_id_file(path: str | pathlib.Path) -> list[str]:
  """Reads trajectory ids from a text file, one per line.

  Blank lines and lines starting with ``#`` are ignored, so a file can
  carry a comment explaining how the id set was produced.

  Args:
    path: Path to the id file.

  Returns:
    The ids in file order.
  """
  ids = []
  for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
      ids.append(line)
  return ids
