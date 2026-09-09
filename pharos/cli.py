"""Command-line interface.

Every subcommand is thin: it parses arguments, loads what the library
needs, calls one library function and prints a summary. Library errors
that a user can act on (unknown trajectory ids, a provider that cannot
run here, a malformed dataset) become a one-line message and exit
status 2. Progress goes to stderr; machine-readable summaries to stdout.

    pharos run        --dataset D --prompt P --name N [--provider ...]
    pharos heuristics --dataset D --name N [--rules all]
    pharos compare    --run A --run B [--out compare.json]
    pharos report     --dataset D --run A [--run B] [--out DIR] [--open]
    pharos validate   --dataset D --run A
    pharos labels     --dataset D --out labels.json [--run R ...]
                      [--from-run JUDGE --leak-min-sev 5]
    pharos dossier    --dataset D --out DIR [--run R ...]
    pharos metrics    --labels L --run R [--run R ...]
"""

import argparse
from collections.abc import Callable, Sequence
import hashlib
import json
import pathlib
import sys
import webbrowser

from pharos import clock
from pharos import compare
from pharos import data
from pharos import metrics
from pharos import report
from pharos import runner
from pharos import schema
from pharos import store
from pharos import validate
from pharos.providers import base
from pharos.providers import factory
from pharos.providers import regex

DEFAULT_RUNS_DIR = "runs"
DEFAULT_REPORTS_DIR = "reports"
EFFORT_CHOICES: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh")
EXIT_USAGE = 2

_SUMMARY_KEYS = ("run_id", "n_results", "status_counts", "verdict_counts")

# Errors a user can act on; anything else is a bug and should traceback.
_USER_ERRORS: tuple[type[Exception], ...] = (
    data.DatasetError,
    data.UnknownTrajectoryIdError,
    base.ProviderError,
    ValueError,
    FileNotFoundError,
)

Command = Callable[[argparse.Namespace], None]


def _log(message: str) -> None:
  """Prints a progress line to stderr."""
  print(message, file=sys.stderr, flush=True)


def _requested_ids(args: argparse.Namespace) -> list[str] | None:
  """Returns the ids named by ``--ids`` or ``--ids-file``, if any."""
  if args.ids_file:
    return data.read_id_file(args.ids_file)
  if args.ids:
    return [i.strip() for i in args.ids.split(",") if i.strip()]
  return None


def _load_selected(args: argparse.Namespace) -> list[data.Trajectory]:
  """Loads the dataset and applies ``--ids``, ``--ids-file``, ``--limit``."""
  trajectories = data.load_trajectories(args.dataset)
  return data.select(trajectories, ids=_requested_ids(args), limit=args.limit)


def _load_by_id(dataset: pathlib.Path) -> dict[str, data.Trajectory]:
  """Loads the dataset keyed by trajectory id."""
  return {t.id: t for t in data.load_trajectories(dataset)}


def _load_runs(run_dirs: Sequence[str]) -> metrics.RunResults:
  """Loads results for several runs, keyed by run directory name."""
  return {pathlib.Path(d).name: store.load_results(d) for d in run_dirs}


def _print_summary(manifest: store.Manifest) -> None:
  """Prints the run totals as JSON on stdout."""
  print(json.dumps({k: manifest.get(k) for k in _SUMMARY_KEYS}, indent=1))


def cmd_run(args: argparse.Namespace) -> None:
  """Runs a monitor prompt over the selected trajectories.

  Args:
    args: Parsed arguments.
  """
  trajectories = _load_selected(args)
  instructions = pathlib.Path(args.prompt).read_text(encoding="utf-8")
  provider = factory.make_provider(
      args.provider, args.model, args.effort, args.timeout
  )
  if args.resume:
    run_dir = pathlib.Path(args.resume)
    done = store.completed_ids(run_dir)
    trajectories = [t for t in trajectories if t.id not in done]
    _log(
        f"resuming {run_dir.name}: {len(done)} already ok,"
        f" {len(trajectories)} to do"
    )
  else:
    run_dir = store.new_run_dir(args.runs_dir, args.name)
    store.write_prompt(run_dir, instructions)
  run_id = run_dir.name

  manifest = store.read_manifest(run_dir)
  subset = bool(args.ids or args.ids_file or args.limit)
  manifest.update(
      {
          "run_id": run_id,
          "name": args.name,
          "prompt_path": str(args.prompt),
          "prompt_sha256": hashlib.sha256(
              instructions.encode("utf-8")
          ).hexdigest(),
          "provider": provider.name,
          "model": provider.model,
          "reasoning_effort": provider.effort,
          "workers": args.workers,
          "retries": args.retries,
          "timeout_s": args.timeout,
          "max_msg_chars": args.max_msg_chars,
          "max_total_chars": args.max_total_chars,
          "dataset": str(args.dataset),
          "dataset_sha256": data.dataset_sha256(args.dataset),
          "n_selected": len(trajectories),
          "ids": [t.id for t in trajectories] if subset else "all",
          "started_at": manifest.get("started_at") or clock.now_iso(),
          "argv": sys.argv,
      }
  )
  store.write_manifest(run_dir, manifest)
  _log(f"run dir: {run_dir}")

  config = runner.RunConfig(
      workers=args.workers,
      retries=args.retries,
      max_msg_chars=args.max_msg_chars,
      max_total_chars=args.max_total_chars,
  )
  runner.run_all(trajectories, provider, instructions, run_dir, run_id, config)
  _print_summary(store.finalize(run_dir))
  _log(f"done: {run_dir}")


def cmd_heuristics(args: argparse.Namespace) -> None:
  """Runs the regex provider over the selected trajectories.

  Args:
    args: Parsed arguments.
  """
  run_dir = regex.run_heuristics(
      _load_selected(args),
      pathlib.Path(args.runs_dir),
      args.name,
      args.rules,
      str(args.dataset),
  )
  _print_summary(store.finalize(run_dir))
  _log(f"done: {run_dir}")


def cmd_compare(args: argparse.Namespace) -> None:
  """Compares two runs and writes the comparison as JSON.

  Args:
    args: Parsed arguments.

  Raises:
    ValueError: If not exactly two ``--run`` were given.
  """
  if len(args.run) != 2:
    raise ValueError("compare needs exactly two --run")
  run_a, run_b = (pathlib.Path(d) for d in args.run)
  comparison = compare.compare_runs(run_a, run_b)
  dest = (
      pathlib.Path(args.out)
      if args.out
      else run_b / f"compare_vs_{run_a.name}.json"
  )
  dest.write_text(
      json.dumps(comparison, indent=1, ensure_ascii=False), encoding="utf-8"
  )
  print(json.dumps(comparison["totals"], indent=1))
  _log(f"wrote {dest}")


def cmd_report(args: argparse.Namespace) -> None:
  """Builds the viewer for one or two runs.

  Args:
    args: Parsed arguments.
  """
  trajectories = data.load_trajectories(args.dataset)
  index = report.build_report(
      [pathlib.Path(d) for d in args.run],
      trajectories,
      pathlib.Path(args.reports_dir),
      out_dir=pathlib.Path(args.out) if args.out else None,
  )
  _log(f"report: {index}")
  print(index)
  if args.open:
    webbrowser.open(index.as_uri())


def cmd_validate(args: argparse.Namespace) -> None:
  """Replays validation over a run's stored raw responses.

  Use after tightening the validation rules: every result whose raw
  response is on disk is rebuilt from it.

  Args:
    args: Parsed arguments.
  """
  trajectories = _load_by_id(args.dataset)
  run_dir = pathlib.Path(args.run)
  rebuilt = 0
  for traj_id, result in store.load_results(run_dir).items():
    raw_path = run_dir / (result.get("raw_response_path") or "")
    if not result.get("raw_response_path") or not raw_path.is_file():
      continue
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    parsed = (raw.get("final") or {}).get("parsed")
    status = (
        schema.EXEC_STATUS_OK
        if parsed is not None
        else result["execution_status"]
    )
    store.write_result(
        run_dir,
        validate.build_result(
            trajectories[traj_id],
            run_id=result["run_id"],
            coverage=result["review_coverage"],
            model_info=result["model"],
            raw_response_path=result["raw_response_path"],
            parsed=parsed,
            status=status,
            error=result.get("error"),
            prompt_sha256=result.get("prompt_sha256"),
        ),
    )
    rebuilt += 1
  store.finalize(run_dir)
  _log(f"re-validated {rebuilt} results in {run_dir}")


def cmd_labels(args: argparse.Namespace) -> None:
  """Writes a labels file, empty or derived from a judge run.

  Args:
    args: Parsed arguments.
  """
  trajectories = _load_by_id(args.dataset)
  ids = _requested_ids(args) or list(trajectories)
  runs = _load_runs(args.run)
  out = pathlib.Path(args.out)
  if args.from_run:
    judge_dir = pathlib.Path(args.from_run)
    n = metrics.labels_from_run(
        out,
        ids,
        store.load_results(judge_dir),
        judge_dir.name,
        runs,
        trajectories,
        leak_min_severity=args.leak_min_sev,
    )
  else:
    n = metrics.scaffold_labels(out, ids, runs, trajectories)
  _log(f"labels: {out} ({n} entries)")


def cmd_dossier(args: argparse.Namespace) -> None:
  """Writes one markdown dossier per trajectory.

  Args:
    args: Parsed arguments.
  """
  trajectories = _load_by_id(args.dataset)
  ids = _requested_ids(args) or list(trajectories)
  n = metrics.write_dossiers(
      pathlib.Path(args.out), ids, _load_runs(args.run), trajectories
  )
  _log(f"dossiers: {args.out} ({n} files)")


def cmd_metrics(args: argparse.Namespace) -> None:
  """Scores runs against a labels file.

  Args:
    args: Parsed arguments.
  """
  labels = metrics.load_labels(args.labels)
  thresholds = [int(x) for x in args.thresholds.split(",") if x.strip()]
  rows = metrics.evaluate(_load_runs(args.run), labels, thresholds=thresholds)
  print(metrics.format_table(rows, labels))
  if args.out:
    pathlib.Path(args.out).write_text(
        json.dumps(rows, indent=1), encoding="utf-8"
    )


def _add_dataset(parser: argparse.ArgumentParser) -> None:
  """Adds the required ``--dataset`` option."""
  parser.add_argument(
      "--dataset",
      required=True,
      type=pathlib.Path,
      help="JSONL file with one trajectory per line",
  )


def _add_selection(parser: argparse.ArgumentParser) -> None:
  """Adds ``--dataset`` plus the id and limit filters."""
  _add_dataset(parser)
  parser.add_argument("--ids", help="comma-separated trajectory ids")
  parser.add_argument(
      "--ids-file", help="file with one trajectory id per line; # comments"
  )
  parser.add_argument(
      "--limit", type=int, help="keep at most this many trajectories"
  )


def _add_runs_dir(parser: argparse.ArgumentParser) -> None:
  """Adds the ``--runs-dir`` option."""
  parser.add_argument(
      "--runs-dir",
      default=DEFAULT_RUNS_DIR,
      help=f"directory that holds runs (default: {DEFAULT_RUNS_DIR})",
  )


def build_parser() -> argparse.ArgumentParser:
  """Builds the argument parser with every subcommand.

  Returns:
    The parser; each subcommand sets ``func`` to its handler.
  """
  parser = argparse.ArgumentParser(
      prog="pharos",
      description="Run monitors over coding-agent trajectories.",
  )
  sub = parser.add_subparsers(dest="command", required=True)

  run = sub.add_parser("run", help="run a monitor prompt over trajectories")
  _add_selection(run)
  _add_runs_dir(run)
  run.add_argument("--prompt", required=True, help="monitor prompt (markdown)")
  run.add_argument("--name", required=True, help="run name")
  run.add_argument(
      "--provider", default=factory.CODEX_CLI, choices=factory.PROVIDER_NAMES
  )
  run.add_argument(
      "--model", default=None, help="model id (default: per provider)"
  )
  run.add_argument("--effort", default="low", choices=EFFORT_CHOICES)
  run.add_argument("--workers", type=int, default=6)
  run.add_argument("--retries", type=int, default=2)
  run.add_argument("--timeout", type=float, default=900, help="seconds")
  run.add_argument(
      "--max-msg-chars", type=int, default=0, help="0 = never truncate"
  )
  run.add_argument(
      "--max-total-chars", type=int, default=0, help="0 = no budget"
  )
  run.add_argument("--resume", help="existing run directory to resume")
  run.set_defaults(func=cmd_run)

  heuristics = sub.add_parser(
      "heuristics", help="regex monitor (no LLM), same result schema"
  )
  _add_selection(heuristics)
  _add_runs_dir(heuristics)
  heuristics.add_argument("--name", required=True, help="run name")
  heuristics.add_argument(
      "--rules", default="all", choices=sorted(regex.RULE_SETS)
  )
  heuristics.set_defaults(func=cmd_heuristics)

  cmp = sub.add_parser("compare", help="compare two runs by trajectory id")
  cmp.add_argument("--run", action="append", required=True, help="run dir")
  cmp.add_argument("--out", help="output JSON (default: inside run B)")
  cmp.set_defaults(func=cmd_compare)

  rep = sub.add_parser("report", help="build the HTML viewer")
  _add_dataset(rep)
  rep.add_argument(
      "--run", action="append", required=True, help="run dir (one or two)"
  )
  rep.add_argument(
      "--reports-dir",
      default=DEFAULT_REPORTS_DIR,
      help=f"directory that holds reports (default: {DEFAULT_REPORTS_DIR})",
  )
  rep.add_argument("--out", help="report directory (default: stamped)")
  rep.add_argument("--open", action="store_true", help="open in a browser")
  rep.set_defaults(func=cmd_report)

  val = sub.add_parser(
      "validate", help="replay validation over stored raw responses"
  )
  _add_dataset(val)
  val.add_argument("--run", required=True, help="run dir")
  val.set_defaults(func=cmd_validate)

  labels = sub.add_parser("labels", help="write a labels file for humans")
  _add_selection(labels)
  labels.add_argument(
      "--run", action="append", default=[], help="runs to show as context"
  )
  labels.add_argument("--out", required=True, help="labels JSON to write")
  labels.add_argument(
      "--from-run", help="derive provisional labels from this judge run"
  )
  labels.add_argument(
      "--leak-min-sev",
      type=int,
      default=metrics.DEFAULT_JUDGE_LEAK_MIN_SEVERITY,
      help="judge severity at which a derived label is 'leak'",
  )
  labels.set_defaults(func=cmd_labels)

  dossier = sub.add_parser(
      "dossier", help="write one markdown adjudication dossier per trajectory"
  )
  _add_selection(dossier)
  dossier.add_argument("--run", action="append", default=[], help="run dir")
  dossier.add_argument("--out", required=True, help="output directory")
  dossier.set_defaults(func=cmd_dossier)

  met = sub.add_parser(
      "metrics", help="precision/recall of runs against a labels file"
  )
  met.add_argument("--labels", required=True, help="labels JSON")
  met.add_argument("--run", action="append", required=True, help="run dir")
  met.add_argument(
      "--thresholds",
      default=",".join(str(t) for t in metrics.DEFAULT_THRESHOLDS),
      help="minimum finding severities to evaluate at",
  )
  met.add_argument("--out", help="write the rows as JSON")
  met.set_defaults(func=cmd_metrics)
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  """Entry point.

  Args:
    argv: Arguments without the program name; defaults to ``sys.argv``.

  Returns:
    Process exit status: 0 on success, 2 on a user-facing error.
  """
  parser = build_parser()
  args = parser.parse_args(argv)
  command: Command = args.func
  try:
    command(args)
  except _USER_ERRORS as e:
    print(f"pharos: error: {e}", file=sys.stderr)
    return EXIT_USAGE
  return 0
