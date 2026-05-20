"""Run the main-text analysis workflow.

This is the lightweight runner for refreshing the core manuscript results
without running the sensitivity experiments. It regenerates the main Rousseau
event-timing analyses and, by default, refreshes the Overleaf-ready PDF copies
under ``monsoon_paper/figures``.

Excluded by design:
- extra-forcing sensitivity;
- bin-width sensitivity;
- KS-window sensitivity;
- composite-age uncertainty sensitivity;
- Barker et al. independent-catalogue checks;
- parametric-bootstrap diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class WorkflowStep:
    """One script in the lightweight paper-refresh workflow."""

    script: str
    description: str


MAIN_ANALYSIS_STEPS: tuple[WorkflowStep, ...] = (
    WorkflowStep(
        "Predictive_information_model.py",
        "main predictive-information models and main-text Figures 1 and 3",
    ),
    WorkflowStep(
        "Orbital_phase_rayleigh.py",
        "Rayleigh phase diagnostics and main-text Figure 2",
    ),
    WorkflowStep(
        "Lagged_predictive_information.py",
        "lagged predictive-information scan and main-text Figure 4",
    ),
)

EXPORT_STEP = WorkflowStep(
    "paper_figure_export.py",
    "copy paper-referenced PDFs to monsoon_paper/figures",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the workflow runner."""

    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the main-text analyses without running sensitivity "
            "experiments."
        )
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Do not run paper_figure_export.py after the analysis scripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run, but do not execute them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later steps if one step fails.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "run_logs" / "main_paper_workflow",
        help="Directory for timestamped workflow logs.",
    )
    return parser.parse_args()


def workflow_steps(skip_export: bool) -> list[WorkflowStep]:
    """Return the ordered workflow, optionally omitting figure export."""

    steps = list(MAIN_ANALYSIS_STEPS)
    if not skip_export:
        steps.append(EXPORT_STEP)
    return steps


def run_step(step: WorkflowStep, log_file) -> int:
    """Run one workflow step while teeing its output to the log file."""

    command = [sys.executable, step.script]
    print(f"\n==> {step.script}", flush=True)
    print(f"    {step.description}", flush=True)
    log_file.write(f"\n==> {' '.join(command)}\n")
    log_file.write(f"    {step.description}\n")
    log_file.flush()

    start = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_file.write(line)
    return_code = process.wait()
    elapsed = time.perf_counter() - start
    message = f"<== {step.script} finished with code {return_code} in {elapsed:.1f} s\n"
    print(message, end="", flush=True)
    log_file.write(message)
    log_file.flush()
    return return_code


def main() -> None:
    """Command-line entry point."""

    args = parse_args()
    steps = workflow_steps(args.skip_export)

    if args.dry_run:
        print("Main paper workflow dry run:")
        for step in steps:
            print(f"  {sys.executable} {step.script}  # {step.description}")
        return

    args.log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.log_dir / f"main_paper_workflow_{timestamp}.log"

    failures: list[tuple[str, int]] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"Main paper workflow started at {timestamp}\n")
        log_file.write(f"Project root: {PROJECT_ROOT}\n")
        for step in steps:
            return_code = run_step(step, log_file)
            if return_code != 0:
                failures.append((step.script, return_code))
                if not args.continue_on_error:
                    break

    print(f"\nWorkflow log: {log_path.relative_to(PROJECT_ROOT)}")
    if failures:
        print("Failed steps:")
        for script, return_code in failures:
            print(f"  {script}: exit code {return_code}")
        raise SystemExit(1)
    print("Main paper workflow completed successfully.")


if __name__ == "__main__":
    main()
