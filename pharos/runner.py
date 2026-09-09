"""Bounded concurrency and bounded retries around a provider.

Each trajectory is rendered, sent, validated and persisted independently,
and its result is written the moment it completes. Failures are persisted
too, so a run always ends with one result per trajectory and a resume can
retry exactly the ones that failed.

Retries are bounded and backed off. Rate-limit signatures in the
provider's error text earn a longer wait than an ordinary failure, and
jitter keeps a pool of workers from retrying in lock-step.
"""

from collections.abc import Callable, Sequence
import concurrent.futures
import dataclasses
import hashlib
import random
import sys
import threading
import time
from typing import Any

from pharos import data
from pharos import render
from pharos import schema
from pharos import store
from pharos import validate
from pharos.providers import base

# Substrings of an error or stderr that indicate throttling.
RATE_LIMIT_MARKERS: tuple[str, ...] = (
    "429",
    "rate limit",
    "rate_limit",
    "usage limit",
    "too many requests",
    "quota",
)

_BASE_BACKOFF_S = 5.0
_RATE_LIMIT_BACKOFF_S = 30.0
_JITTER_S = 3.0

_STATUS_BY_KIND: dict[str, str] = {
    base.KIND_TIMEOUT: schema.EXEC_STATUS_FAILED_TIMEOUT,
    base.KIND_PROVIDER: schema.EXEC_STATUS_FAILED_PROVIDER,
    base.KIND_PARSE: schema.EXEC_STATUS_FAILED_PARSE,
}

Logger = Callable[[str], None]
Sleeper = Callable[[float], None]


@dataclasses.dataclass(frozen=True)
class RunConfig:
  """Knobs for one run.

  Attributes:
    workers: Concurrent provider calls.
    retries: Further attempts after the first failure.
    max_msg_chars: Per-tool-message character cap; 0 never truncates.
    max_total_chars: Whole-prompt character budget; 0 means none.
  """

  workers: int = 6
  retries: int = 2
  max_msg_chars: int = 0
  max_total_chars: int = 0


def is_rate_limited(response: base.RawResponse) -> bool:
  """Returns whether a failed response looks like throttling."""
  blob = f"{response.error or ''} {response.stderr or ''}".lower()
  return any(marker in blob for marker in RATE_LIMIT_MARKERS)


def backoff_seconds(
    attempt: int, rate_limited: bool, rng: random.Random
) -> float:
  """Returns how long to wait before retrying after ``attempt``.

  Args:
    attempt: Zero-based index of the attempt that just failed.
    rate_limited: Whether the failure looked like throttling.
    rng: Source of jitter, injectable for tests.
  """
  base_wait = _RATE_LIMIT_BACKOFF_S if rate_limited else _BASE_BACKOFF_S
  return base_wait * (2.0**attempt) + rng.uniform(0, _JITTER_S)


def _status_for(response: base.RawResponse | None) -> str:
  """Maps the final attempt's outcome onto an execution status."""
  if response is None:
    return schema.EXEC_STATUS_FAILED_PROVIDER
  if response.ok:
    return schema.EXEC_STATUS_OK
  if response.kind is None:
    return schema.EXEC_STATUS_FAILED_PARSE
  return _STATUS_BY_KIND[response.kind]


def _attempt_record(attempt: int, response: base.RawResponse) -> dict[str, Any]:
  """Returns the per-attempt summary kept in the raw file."""
  return {
      "attempt": attempt + 1,
      "error": response.error,
      "kind": response.kind,
      "duration_s": round(response.duration_s, 1),
      "exit_code": response.exit_code,
      "tokens_used": response.tokens_used,
  }


def run_one(
    traj: data.Trajectory,
    provider: base.Provider,
    instructions: str,
    run_dir: store.pathlib.Path,
    run_id: str,
    config: RunConfig,
    log: Logger,
    sleep: Sleeper = time.sleep,
    rng: random.Random | None = None,
) -> schema.Result:
  """Evaluates one trajectory end to end and persists the result.

  Args:
    traj: The trajectory.
    provider: Where to send the prompt.
    instructions: The monitor prompt.
    run_dir: The run directory.
    run_id: The run id recorded on the result.
    config: Retry and truncation settings.
    log: Receives one line per retry.
    sleep: Backoff sleeper, injectable for tests.
    rng: Jitter source, injectable for tests.

  Returns:
    The persisted result, successful or not.
  """
  rng = rng or random.Random()
  rendered = render.render(
      traj,
      instructions,
      max_msg_chars=config.max_msg_chars,
      max_total_chars=config.max_total_chars,
  )
  prompt_sha256 = hashlib.sha256(rendered.prompt.encode("utf-8")).hexdigest()

  attempts: list[dict[str, Any]] = []
  response: base.RawResponse | None = None
  for attempt in range(config.retries + 1):
    response = provider.complete(rendered.prompt, schema.MODEL_OUTPUT_SCHEMA)
    attempts.append(_attempt_record(attempt, response))
    if response.ok:
      break
    if attempt < config.retries:
      wait = backoff_seconds(attempt, is_rate_limited(response), rng)
      log(
          f"  retry {attempt + 1}/{config.retries} for {traj.id} after"
          f" {response.kind}: {(response.error or '')[:120]}"
          f" (sleep {wait:.0f}s)"
      )
      sleep(wait)

  raw_path = store.write_raw(
      run_dir,
      traj.id,
      {
          "traj_id": traj.id,
          "prompt_sha256": prompt_sha256,
          "prompt_chars": len(rendered.prompt),
          "attempts": attempts,
          "final": response.to_json() if response else None,
      },
  )
  status = _status_for(response)
  model_info: schema.ModelInfo = {
      "provider": provider.name,
      "model": provider.model,
      "reasoning_effort": provider.effort,
      "tokens_used": response.tokens_used if response else None,
      "duration_s": round(sum(a["duration_s"] for a in attempts), 1),
      "attempts": len(attempts),
  }
  result = validate.build_result(
      traj,
      run_id=run_id,
      coverage=rendered.coverage,
      model_info=model_info,
      raw_response_path=raw_path,
      parsed=response.parsed if response and response.ok else None,
      status=status,
      error=response.error if response else "no response",
      prompt_sha256=prompt_sha256,
  )
  store.write_result(run_dir, result)
  return result


def _infra_failure_result(
    traj: data.Trajectory,
    provider: base.Provider,
    run_id: str,
    error: BaseException,
) -> schema.Result:
  """Builds the result persisted when run_one itself raised.

  A bug in rendering, validation or persistence must not lose the
  trajectory's slot in the run: it is recorded as a provider failure
  with the exception text so a resume retries it.

  Args:
    traj: The trajectory whose evaluation crashed.
    provider: The provider in use, for the model info.
    run_id: The run id.
    error: The exception raised.

  Returns:
    A failed result with empty coverage.
  """
  return validate.build_result(
      traj,
      run_id=run_id,
      coverage=render.empty_coverage(traj),
      model_info={
          "provider": provider.name,
          "model": provider.model,
          "reasoning_effort": provider.effort,
      },
      raw_response_path=None,
      parsed=None,
      status=schema.EXEC_STATUS_FAILED_PROVIDER,
      error=f"infra exception: {error!r}",
  )


def _stderr_logger() -> Logger:
  """Returns a logger that serialises lines from worker threads."""
  lock = threading.Lock()

  def log(message: str) -> None:
    with lock:
      print(message, file=sys.stderr, flush=True)

  return log


def run_all(
    trajectories: Sequence[data.Trajectory],
    provider: base.Provider,
    instructions: str,
    run_dir: store.pathlib.Path,
    run_id: str,
    config: RunConfig,
    log: Logger | None = None,
) -> list[schema.Result]:
  """Evaluates every trajectory with a bounded worker pool.

  Args:
    trajectories: Trajectories to evaluate.
    provider: Where to send prompts. Must be safe to call from several
      threads at once.
    instructions: The monitor prompt.
    run_dir: The run directory.
    run_id: The run id recorded on results.
    config: Concurrency, retry and truncation settings.
    log: Receives one progress line per completion; defaults to stderr.

  Returns:
    Results in completion order, one per trajectory.
  """
  log = log or _stderr_logger()
  results: list[schema.Result] = []
  total = len(trajectories)
  failed = 0
  started = time.time()
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=config.workers
  ) as pool:
    futures = {
        pool.submit(
            run_one, traj, provider, instructions, run_dir, run_id, config, log
        ): traj
        for traj in trajectories
    }
    for future in concurrent.futures.as_completed(futures):
      traj = futures[future]
      try:
        result = future.result()
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Any bug in the per-trajectory path is recorded, not propagated:
        # one bad trajectory must not abort the other 231.
        result = _infra_failure_result(traj, provider, run_id, e)
        store.write_result(run_dir, result)
      results.append(result)
      if not schema.is_ok_status(result["execution_status"]):
        failed += 1
      classes = ",".join(result["threat_classes"]) or "-"
      log(
          f"[{len(results)}/{total}] {traj.id} {result['execution_status']}"
          f" verdict={result['verdict']} sev={result['max_severity']}"
          f" classes={classes} {result['model'].get('duration_s', 0)}s"
          f" | failed={failed} elapsed={time.time() - started:.0f}s"
      )
  return results
