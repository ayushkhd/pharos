"""Validation of model output against the source trajectory.

Turns the raw JSON a provider returned into a persisted ``Result``. The
rules are deterministic and run before anything is stored:

1. Every evidence index must exist in the trajectory; others are dropped
   with a recorded warning. A finding left without evidence is kept but
   marked unverifiable rather than silently deleted.
2. Provenance (role, origin, tool_call_id) of every cited message is
   looked up from the trajectory, never taken from the model.
3. Every quote is searched verbatim, then whitespace-normalised, in its
   cited message; a miss is recorded on the quote and the message is
   still shown in full.
4. Severity is clamped to the 1-10 scale; unknown verdicts and
   categories fall back with a warning.
5. Inconsistent verdicts (``clean`` with findings, ``concerning`` or
   worse without) are surfaced, not corrected.

Any warning demotes ``ok`` to ``ok_with_warnings`` so the viewer shows
the result as annotated. The raw response is kept separately by the
runner, so every decision made here can be audited or replayed.
"""

import re
from typing import Any, TypeGuard

from pharos import clock
from pharos import data
from pharos import schema

_WHITESPACE = re.compile(r"\s+")
_FALLBACK_VERDICT = "concerning"


def _normalise_whitespace(text: str) -> str:
  """Returns ``text`` with runs of whitespace collapsed to one space."""
  return _WHITESPACE.sub(" ", text).strip()


def quote_found(quote: str, content: str) -> bool:
  """Returns whether ``quote`` occurs in ``content``.

  A verbatim match is tried first, then a match with whitespace
  normalised on both sides, which forgives line-wrapping differences
  without accepting paraphrase.

  Args:
    quote: The text the model claims to have copied.
    content: The cited message's content.
  """
  if not quote:
    return False
  if quote in content:
    return True
  return _normalise_whitespace(quote) in _normalise_whitespace(content)


def _is_valid_index(value: Any, n_messages: int) -> TypeGuard[int]:
  """Returns whether ``value`` is an int naming an existing message."""
  return (
      isinstance(value, int)
      and not isinstance(value, bool)
      and 0 <= value < n_messages
  )


def _coerce_severity(value: Any, warnings: list[str]) -> int:
  """Returns ``value`` as an int within the severity scale.

  Args:
    value: The model-supplied severity.
    warnings: Receives a note for every correction made.
  """
  try:
    severity = int(value)
  except (TypeError, ValueError):
    warnings.append(f"non-integer severity {value!r}; set to 1")
    return schema.SEVERITY_MIN
  if not schema.SEVERITY_MIN <= severity <= schema.SEVERITY_MAX:
    warnings.append(f"severity {severity} outside 1-10; clamped")
    severity = max(schema.SEVERITY_MIN, min(schema.SEVERITY_MAX, severity))
  return severity


def _validate_finding(
    raw: dict[str, Any], traj: data.Trajectory
) -> schema.Finding:
  """Checks one model-supplied finding against the trajectory.

  Args:
    raw: A finding as returned by the model.
    traj: The trajectory it refers to.

  Returns:
    The finding with evidence resolved and warnings attached.
  """
  warnings: list[str] = []
  n = traj.n_messages

  category = raw.get("category")
  if category not in schema.CATEGORIES:
    warnings.append(
        f"unknown category {category!r}; mapped to"
        f" {schema.FALLBACK_CATEGORY!r}"
    )
    category = schema.FALLBACK_CATEGORY

  indices: list[int] = []
  for index in raw.get("evidence_indices") or []:
    if not _is_valid_index(index, n):
      warnings.append(f"evidence index {index!r} out of range (n={n}); dropped")
    elif index not in indices:
      indices.append(index)

  quotes_by_index: dict[int, list[schema.QuoteCheck]] = {}
  for raw_quote in raw.get("quotes") or []:
    if not isinstance(raw_quote, dict):
      warnings.append(f"quote is not an object; dropped: {raw_quote!r}")
      continue
    index = raw_quote.get("index")
    text = str(raw_quote.get("quote", ""))
    if not _is_valid_index(index, n):
      warnings.append(f"quote index {index!r} out of range; dropped")
      continue
    if index not in indices:
      indices.append(index)
      warnings.append(
          f"quote cites index {index} not listed in evidence_indices; added"
      )
    found = quote_found(text, traj.message(index).content)
    if not found:
      warnings.append(f"quote for index {index} not found verbatim in message")
    quotes_by_index.setdefault(index, []).append(
        {"quote": text, "found": found}
    )

  # Provenance is recomputed from the log. Whatever the model said about
  # who wrote a message is discarded here.
  evidence: list[schema.EvidenceRef] = []
  roles: dict[str, int] = {}
  for index in sorted(indices):
    message = traj.message(index)
    roles[message.role] = roles.get(message.role, 0) + 1
    evidence.append(
        {
            "index": index,
            "role": message.role,
            "origin": message.origin,
            "tool_call_id": message.tool_call_id,
            "quotes": quotes_by_index.get(index, []),
        }
    )
  if not evidence:
    warnings.append("finding has no valid evidence indices; unverifiable")

  return {
      "category": category,
      "evidence_indices": sorted(indices),
      "evidence": evidence,
      "evidence_roles": roles,
      "observed_behavior": str(raw.get("observed_behavior", "")),
      "severity": _coerce_severity(
          raw.get("severity", schema.SEVERITY_MIN), warnings
      ),
      "limitations": [str(x) for x in (raw.get("limitations") or [])],
      "validation_warnings": warnings,
  }


def _validate_verdict(raw: Any, warnings: list[str]) -> str:
  """Returns ``raw`` if it is a known verdict, else a flagged fallback."""
  if raw in schema.VERDICTS:
    return str(raw)
  warnings.append(f"unknown verdict {raw!r}; treated as {_FALLBACK_VERDICT!r}")
  return _FALLBACK_VERDICT


def build_result(
    traj: data.Trajectory,
    *,
    run_id: str,
    coverage: schema.ReviewCoverage,
    model_info: schema.ModelInfo,
    raw_response_path: str | None,
    parsed: dict[str, Any] | None,
    status: str,
    error: str | None = None,
    prompt_sha256: str | None = None,
) -> schema.Result:
  """Builds the persisted result for one trajectory.

  Args:
    traj: The trajectory that was evaluated.
    run_id: The run this result belongs to.
    coverage: What the model saw, from ``render``.
    model_info: Provider, model, effort and cost.
    raw_response_path: Path of the verbatim response, relative to the run
      directory, or ``None`` if nothing was stored.
    parsed: The model's JSON output, or ``None`` if the provider produced
      nothing usable.
    status: The execution status before validation; ``ok`` may be
      demoted to ``ok_with_warnings`` here.
    error: Provider error text, if any.
    prompt_sha256: Digest of the exact prompt sent.

  Returns:
    The validated result, ready to persist.
  """
  warnings: list[str] = []
  findings: list[schema.Finding] = []
  verdict: str | None = None
  summary = ""

  if parsed is not None:
    verdict = _validate_verdict(parsed.get("verdict"), warnings)
    summary = str(parsed.get("summary", ""))
    for raw_finding in parsed.get("findings") or []:
      if not isinstance(raw_finding, dict):
        warnings.append(
            f"finding is not an object ({type(raw_finding).__name__});"
            " dropped"
        )
        continue
      findings.append(_validate_finding(raw_finding, traj))

    if verdict == "clean" and findings:
      warnings.append("verdict 'clean' but findings present")
    flagged = schema.VERDICT_RANK[verdict] >= schema.VERDICT_RANK["concerning"]
    if flagged and not findings:
      warnings.append(f"verdict '{verdict}' but no findings")

  has_warnings = bool(warnings) or any(
      f["validation_warnings"] for f in findings
  )
  if status == schema.EXEC_STATUS_OK and has_warnings:
    status = schema.EXEC_STATUS_OK_WITH_WARNINGS

  return {
      "traj_id": traj.id,
      "run_id": run_id,
      "execution_status": status,
      "error": error,
      "review_coverage": coverage,
      "verdict": verdict,
      "threat_classes": sorted({f["category"] for f in findings}),
      "max_severity": max((f["severity"] for f in findings), default=0),
      "summary": summary,
      "findings": findings,
      "validation_warnings": warnings,
      "model": model_info,
      "prompt_sha256": prompt_sha256,
      "raw_response_path": raw_response_path,
      "metadata": traj.metadata,
      "completed_at": clock.now_iso(),
  }
