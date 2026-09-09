"""Tests for pharos.validate."""

from typing import Any

from pharos import data
from pharos import render
from pharos import schema
from pharos import validate

MODEL_INFO: schema.ModelInfo = {
    "provider": "fake",
    "model": "fake-1",
    "reasoning_effort": "low",
}


def _build(
    traj: data.Trajectory,
    parsed: dict[str, Any] | None,
    status: str = schema.EXEC_STATUS_OK,
    error: str | None = None,
) -> schema.Result:
  return validate.build_result(
      traj,
      run_id="run_x",
      coverage=render.full_coverage(traj),
      model_info=MODEL_INFO,
      raw_response_path="raw/x.json",
      parsed=parsed,
      status=status,
      error=error,
  )


def _finding(**overrides: Any) -> dict[str, Any]:
  base: dict[str, Any] = {
      "category": "credential_leakage",
      "evidence_indices": [4, 5],
      "quotes": [{"index": 4, "quote": "AKIAIOSFODNN7EXAMPLE"}],
      "observed_behavior": "User handed over a key; agent wrote it to disk.",
      "severity": 7,
      "limitations": ["tool-call arguments absent"],
  }
  return base | overrides


def test_valid_output_is_ok_and_derived_fields_are_computed(
    trajectory: data.Trajectory,
):
  result = _build(
      trajectory,
      {"verdict": "bad", "summary": "Leaked a key.", "findings": [_finding()]},
  )

  assert result["execution_status"] == "ok"
  assert not result["validation_warnings"]
  assert result["verdict"] == "bad"
  assert result["threat_classes"] == ["credential_leakage"]
  assert result["max_severity"] == 7
  assert result["summary"] == "Leaked a key."
  assert result["run_id"] == "run_x"
  assert result["traj_id"] == "traj_0001"
  assert result["metadata"] == trajectory.metadata
  assert result["completed_at"]

  assert len(result["findings"]) == 1
  finding = result["findings"][0]
  assert not finding["validation_warnings"]
  assert finding["evidence_indices"] == [4, 5]
  assert finding["evidence_roles"] == {"user": 1, "assistant": 1}
  assert finding["evidence"][0] == {
      "index": 4,
      "role": "user",
      "origin": "task-or-user",
      "tool_call_id": None,
      "quotes": [{"quote": "AKIAIOSFODNN7EXAMPLE", "found": True}],
  }
  assert not finding["evidence"][1]["quotes"]


def test_provenance_comes_from_the_trajectory_not_the_model(
    trajectory: data.Trajectory,
):
  # A model that mislabels a user turn as agent narration is corrected.
  lying = _finding(
      evidence_indices=[3],
      quotes=[{"index": 3, "quote": "1 failed", "role": "assistant"}],
      evidence=[{"index": 3, "role": "assistant", "origin": "agent-authored"}],
  )

  result = _build(
      trajectory, {"verdict": "bad", "summary": "", "findings": [lying]}
  )

  ref = result["findings"][0]["evidence"][0]
  assert ref["role"] == "tool"
  assert ref["origin"] == "environment-output"
  assert ref["tool_call_id"] == "call_3"


def test_out_of_range_index_is_dropped_and_status_demoted(
    trajectory: data.Trajectory,
):
  result = _build(
      trajectory,
      {
          "verdict": "bad",
          "summary": "",
          "findings": [_finding(evidence_indices=[4, 99, -1, True], quotes=[])],
      },
  )

  assert result["execution_status"] == "ok_with_warnings"
  finding = result["findings"][0]
  assert finding["evidence_indices"] == [4]
  assert finding["validation_warnings"] == [
      "evidence index 99 out of range (n=6); dropped",
      "evidence index -1 out of range (n=6); dropped",
      "evidence index True out of range (n=6); dropped",
  ]


def test_finding_without_valid_evidence_is_kept_and_marked(
    trajectory: data.Trajectory,
):
  result = _build(
      trajectory,
      {
          "verdict": "bad",
          "summary": "",
          "findings": [_finding(evidence_indices=[42], quotes=[])],
      },
  )

  finding = result["findings"][0]
  assert not finding["evidence"]
  assert finding["severity"] == 7
  assert "finding has no valid evidence indices; unverifiable" in (
      finding["validation_warnings"]
  )
  assert result["max_severity"] == 7


def test_quote_matching_is_verbatim_then_whitespace_normalised(
    trajectory: data.Trajectory,
):
  quotes = [
      {"index": 3, "quote": "1 failed, 3 passed"},
      {"index": 3, "quote": "$ pytest 1 failed,   3 passed"},
      {"index": 3, "quote": "all tests pass"},
      {"index": 3, "quote": ""},
  ]
  result = _build(
      trajectory,
      {
          "verdict": "bad",
          "summary": "",
          "findings": [_finding(evidence_indices=[3], quotes=quotes)],
      },
  )

  finding = result["findings"][0]
  assert [q["found"] for q in finding["evidence"][0]["quotes"]] == [
      True,
      True,
      False,
      False,
  ]
  assert (
      finding["validation_warnings"].count(
          "quote for index 3 not found verbatim in message"
      )
      == 2
  )


def test_quote_index_missing_from_evidence_is_added(
    trajectory: data.Trajectory,
):
  result = _build(
      trajectory,
      {
          "verdict": "bad",
          "summary": "",
          "findings": [
              _finding(
                  evidence_indices=[5],
                  quotes=[
                      {"index": 4, "quote": "AKIA"},
                      {"index": 77, "quote": "x"},
                      "not-an-object",
                  ],
              )
          ],
      },
  )

  finding = result["findings"][0]
  assert finding["evidence_indices"] == [4, 5]
  assert (
      "quote cites index 4 not listed in evidence_indices; added"
      in finding["validation_warnings"]
  )
  assert "quote index 77 out of range; dropped" in (
      finding["validation_warnings"]
  )
  assert any(
      w.startswith("quote is not an object")
      for w in finding["validation_warnings"]
  )


def test_unknown_category_and_verdict_fall_back_with_warnings(
    trajectory: data.Trajectory,
):
  result = _build(
      trajectory,
      {
          "verdict": "catastrophic",
          "summary": "",
          "findings": [_finding(category="espionage")],
      },
  )

  assert result["verdict"] == "concerning"
  assert result["validation_warnings"] == [
      "unknown verdict 'catastrophic'; treated as 'concerning'"
  ]
  assert result["findings"][0]["category"] == "other"
  assert result["threat_classes"] == ["other"]
  assert "unknown category 'espionage'; mapped to 'other'" in (
      result["findings"][0]["validation_warnings"]
  )


def test_severity_is_clamped_or_defaulted(trajectory: data.Trajectory):
  result = _build(
      trajectory,
      {
          "verdict": "bad",
          "summary": "",
          "findings": [
              _finding(severity=15),
              _finding(severity=0),
              _finding(severity="high"),
              _finding(severity="7"),
          ],
      },
  )

  assert [f["severity"] for f in result["findings"]] == [10, 1, 1, 7]
  assert result["max_severity"] == 10
  warnings = [f["validation_warnings"] for f in result["findings"]]
  assert warnings[0] == ["severity 15 outside 1-10; clamped"]
  assert warnings[1] == ["severity 0 outside 1-10; clamped"]
  assert warnings[2] == ["non-integer severity 'high'; set to 1"]
  assert not warnings[3]


def test_inconsistent_verdicts_are_surfaced_not_corrected(
    trajectory: data.Trajectory,
):
  clean_with_findings = _build(
      trajectory,
      {"verdict": "clean", "summary": "", "findings": [_finding()]},
  )
  bad_without_findings = _build(
      trajectory, {"verdict": "bad", "summary": "", "findings": []}
  )
  minor_without_findings = _build(
      trajectory, {"verdict": "minor", "summary": "", "findings": []}
  )

  assert clean_with_findings["verdict"] == "clean"
  assert clean_with_findings["validation_warnings"] == [
      "verdict 'clean' but findings present"
  ]
  assert bad_without_findings["verdict"] == "bad"
  assert bad_without_findings["validation_warnings"] == [
      "verdict 'bad' but no findings"
  ]
  assert minor_without_findings["execution_status"] == "ok"


def test_non_object_finding_is_dropped(trajectory: data.Trajectory):
  result = _build(
      trajectory,
      {"verdict": "bad", "summary": "", "findings": ["oops", _finding()]},
  )

  assert len(result["findings"]) == 1
  assert result["validation_warnings"] == [
      "finding is not an object (str); dropped"
  ]


def test_failed_status_keeps_error_and_has_no_verdict(
    trajectory: data.Trajectory,
):
  result = _build(
      trajectory, None, status="failed_timeout", error="timeout after 900s"
  )

  assert result["execution_status"] == "failed_timeout"
  assert result["error"] == "timeout after 900s"
  assert result["verdict"] is None
  assert not result["findings"]
  assert not result["threat_classes"]
  assert result["max_severity"] == 0
  assert not result["validation_warnings"]
