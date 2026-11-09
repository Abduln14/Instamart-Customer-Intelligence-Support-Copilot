#!/usr/bin/env python3
"""
Instamart Copilot — End-to-End Pipeline Runner
==============================================
Executes all 10 stages of the Instamart Copilot pipeline sequentially,
streaming live terminal output, logging to disk, timing each step, and
summarizing results.

Usage:
  python run_project.py                  # Run all stages (1 to 10)
  python run_project.py --only 6 7       # Run only stages 6 and 7
  python run_project.py --from-stage 5   # Run stages 5 through 10
  python run_project.py --keep-going     # Continue pipeline on errors
  python run_project.py --list           # List stages without running
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

# Ensure stdout/stderr handles UTF-8 cleanly on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Pipeline Stages Definition ────────────────────────────────────────────────
# (Stage Number, Relative Script Path, Human-Readable Stage Name)
STAGES: List[Tuple[int, str, str]] = [
    (1, "generate_data.py", "Generate synthetic data"),
    (2, "src/part_a_clean.py", "Part A: Data cleaning"),
    (3, "src/part_a_validate.py", "Part A: Validate checkpoints"),
    (4, "src/part_a_sql.py", "Part A: SQL queries"),
    (5, "src/part_a_pandas.py", "Part A: Pandas aggregations"),
    (6, "src/part_b_rfm_cluster.py", "Part B: RFM clustering"),
    (7, "src/part_b_lapse.py", "Part B: Lapse prediction model"),
    (8, "src/part_c_triage.py", "Part C: Ticket triage"),
    (9, "src/part_d_rag.py", "Part D: Policy RAG"),
    (10, "src/part_d_agent.py", "Part D: Autonomous agent scenarios"),
]


# ── ANSI Terminal Styling & Fallback ─────────────────────────────────────────
def _supports_color() -> bool:
    """Check if the active terminal supports ANSI escape sequences."""
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                if kernel32.SetConsoleMode(handle, mode.value | 0x0004):
                    return True
        except Exception:
            pass
        return bool(os.environ.get("ANSICON") or os.environ.get("WT_SESSION"))
    return True


USE_COLOR = _supports_color()


def colorize(text: str, code: str) -> str:
    """Wrap text in ANSI escape code if color is supported."""
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def bold(text: str) -> str:
    return colorize(text, "1")


def green(text: str) -> str:
    return colorize(text, "1;32")


def red(text: str) -> str:
    return colorize(text, "1;31")


def yellow(text: str) -> str:
    return colorize(text, "1;33")


def cyan(text: str) -> str:
    return colorize(text, "1;36")


def gray(text: str) -> str:
    return colorize(text, "90")


# ── Root Path Discovery ──────────────────────────────────────────────────────
def find_project_root() -> Path:
    """
    Locate the folder containing `generate_data.py` and `src/`.
    Works whether run_project.py is at the root repo or inside a subfolder.
    """
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "generate_data.py").is_file() and (script_dir / "src").is_dir():
        return script_dir
    if (script_dir / "instamart-copilot" / "generate_data.py").is_file():
        return script_dir / "instamart-copilot"
    cwd = Path.cwd()
    if (cwd / "generate_data.py").is_file() and (cwd / "src").is_dir():
        return cwd
    if (cwd / "instamart-copilot" / "generate_data.py").is_file():
        return cwd / "instamart-copilot"
    return script_dir


# ── Stage Validation ─────────────────────────────────────────────────────────
def validate_scripts(project_root: Path) -> None:
    """Check that all required pipeline script files exist up-front."""
    missing: List[Tuple[int, str, Path]] = []
    for num, script_rel, name in STAGES:
        target_path = project_root / script_rel
        if not target_path.is_file():
            missing.append((num, script_rel, target_path))

    if missing:
        print("\n" + red("=" * 78))
        print(red(" [!] ERROR: Missing Required Pipeline Script(s)"))
        print(red("=" * 78))
        for num, rel, full in missing:
            print(f"  Stage {num:2d} | Missing file: {rel}")
            print(f"           Expected at: {full}")
        print("\nPlease restore the missing files before running the pipeline.\n")
        sys.exit(1)


# ── Execution Logic ──────────────────────────────────────────────────────────
def run_stage(
    num: int,
    total: int,
    script_rel: str,
    name: str,
    project_root: Path,
    logs_dir: Path,
) -> Tuple[str, float, List[str], Path]:
    """
    Run a single pipeline stage script, streaming live output to stdout
    and logging to outputs/logs/<script_name>.txt.

    Returns:
        (status_string, duration_seconds, output_lines, log_path)
    """
    script_path = project_root / script_rel
    log_name = f"{Path(script_rel).stem}.txt"
    log_path = logs_dir / log_name

    # Clear header
    header_title = f"[{num}/{total}] {name}"
    print("\n" + cyan("=" * 78))
    print(cyan(f" >>> {header_title}"))
    print(gray(f"     Script: {script_rel}  |  Log: outputs/logs/{log_name}"))
    print(cyan("=" * 78))

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [sys.executable, str(script_path)]
    start_time = time.perf_counter()

    output_lines: List[str] = []
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout stream
            cwd=str(project_root),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,  # Line-buffered
        )

        with open(log_path, "w", encoding="utf-8") as log_file:
            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_file.write(line)
                    log_file.flush()
                    output_lines.append(line.rstrip("\r\n"))

        return_code = process.wait()
    except Exception as exc:
        duration = time.perf_counter() - start_time
        err_msg = f"Failed to execute process: {exc}"
        print(red(err_msg))
        output_lines.append(err_msg)
        return "FAILED", duration, output_lines, log_path

    duration = time.perf_counter() - start_time
    status = "PASSED" if return_code == 0 else "FAILED"

    # Status announcement
    status_colored = green("[PASSED]") if status == "PASSED" else red("[FAILED]")
    print(cyan("-" * 78))
    print(f" {status_colored} {header_title} in {duration:.2f}s")
    print(cyan("-" * 78))

    return status, duration, output_lines, log_path


# ── Summary & Results Presentation ───────────────────────────────────────────
def print_summary_table(
    results: List[Dict[str, object]],
    total_time: float,
) -> None:
    """Print the final formatted tabular summary of all stages."""
    print("\n" + bold("=" * 78))
    print(bold(" PIPELINE EXECUTION SUMMARY"))
    print(bold("=" * 78))
    print(f" {'#':<4} {'Stage Name':<38} {'Status':<12} {'Duration':<10}")
    print("-" * 78)

    passed_count = 0
    failed_count = 0
    skipped_count = 0

    for r in results:
        num = r["num"]
        name = r["name"]
        status = r["status"]
        duration = r["duration"]

        if status == "PASSED":
            status_str = green("PASSED")
            passed_count += 1
        elif status == "FAILED":
            status_str = red("FAILED")
            failed_count += 1
        else:
            status_str = yellow("SKIPPED")
            skipped_count += 1

        dur_str = f"{duration:.2f}s" if duration is not None else "--"
        print(f" {num:<4} {name:<38} {status_str:<21} {dur_str:<10}")

    print("-" * 78)
    summary_line = (
        f" Total Stages: {len(results)}  |  "
        f"Passed: {green(str(passed_count))}  |  "
        f"Failed: {red(str(failed_count)) if failed_count else '0'}  |  "
        f"Skipped: {yellow(str(skipped_count)) if skipped_count else '0'}  |  "
        f"Total Runtime: {bold(f'{total_time:.2f}s')}"
    )
    print(summary_line)
    print(bold("=" * 78) + "\n")


def print_key_results(results: List[Dict[str, object]]) -> None:
    """Print a clean 'Key Results' section showing the last 3 lines from each run stage."""
    executed = [r for r in results if r["status"] in ("PASSED", "FAILED") and r["lines"]]
    if not executed:
        return

    print(bold("=" * 78))
    print(bold(" KEY RESULTS HIGHLIGHTS (Last 3 output lines per stage)"))
    print(bold("=" * 78))

    for r in executed:
        num = r["num"]
        name = r["name"]
        lines = r["lines"]
        status_tag = green("[OK]") if r["status"] == "PASSED" else red("[FAIL]")

        print(f"\n{status_tag} {bold(f'Stage [{num}/{len(STAGES)}] {name}')}:")
        last_3 = lines[-3:] if len(lines) >= 3 else lines
        for line in last_3:
            clean_line = line.strip()
            if clean_line:
                print(f"   {gray('|')} {clean_line}")

    print("\n" + bold("=" * 78) + "\n")


# ── Main Entrypoint ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Instamart Copilot — End-to-End Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_project.py                   # Run all 10 stages in order
  python run_project.py --only 6 7        # Run only stages 6 and 7
  python run_project.py --from-stage 5    # Run stages 5 through 10
  python run_project.py --keep-going      # Continue executing on errors
  python run_project.py --list            # List all stages without running
""",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        type=int,
        metavar="STAGE",
        help="Run only these stage numbers (e.g. --only 6 7)",
    )
    parser.add_argument(
        "--from-stage",
        type=int,
        metavar="N",
        help="Start pipeline from stage N (e.g. --from-stage 5)",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Don't stop on failure; continue executing remaining stages",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all pipeline stages and exit",
    )

    args = parser.parse_args()

    # If --list requested, print stage inventory and exit
    if args.list:
        print("\n" + bold("INSTAMART COPILOT - PIPELINE STAGES"))
        print("-" * 65)
        for num, script, name in STAGES:
            print(f"  Stage {num:2d}: {name:<35} ({script})")
        print("-" * 65)
        print("Run all stages:  python run_project.py")
        print("Run specific:    python run_project.py --only 6 7\n")
        sys.exit(0)

    # Resolve project root and change cwd
    project_root = find_project_root()
    os.chdir(project_root)

    # Ensure outputs/logs/ directory exists
    logs_dir = project_root / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Validate that all required scripts exist before running anything
    validate_scripts(project_root)

    # Determine which stages to execute
    target_stages = STAGES.copy()
    if args.only:
        target_set = set(args.only)
        target_stages = [s for s in STAGES if s[0] in target_set]
        if not target_stages:
            print(red(f"Error: No valid stages matched --only {args.only}. Available: 1-10"))
            sys.exit(1)
    elif args.from_stage:
        target_stages = [s for s in STAGES if s[0] >= args.from_stage]
        if not target_stages:
            print(red(f"Error: Stage {args.from_stage} is out of bounds (1-10)."))
            sys.exit(1)

    # Welcome banner
    print("\n" + bold(cyan("+" + "=" * 76 + "+")))
    print(bold(cyan("|")) + bold("  INSTAMART COPILOT - AUTOMATED END-TO-END PIPELINE RUNNER".center(76)) + bold(cyan("|")))
    print(bold(cyan("+" + "=" * 76 + "+")))
    print(gray(f"  Working Directory: {project_root}"))
    print(gray(f"  Python Binary:     {sys.executable}"))
    print(gray(f"  Log Storage:       {logs_dir}"))
    print(gray(f"  Stages to Run:     {[s[0] for s in target_stages]} of {len(STAGES)}"))
    if args.keep_going:
        print(yellow("  Flag Active:       --keep-going (pipeline will not halt on error)"))

    pipeline_start = time.perf_counter()
    results: List[Dict[str, object]] = []
    has_failure = False

    for num, script_rel, name in STAGES:
        # Check if stage was selected for execution
        if (num, script_rel, name) not in target_stages:
            results.append({
                "num": num,
                "name": name,
                "status": "SKIPPED",
                "duration": None,
                "lines": [],
            })
            continue

        status, duration, lines, log_path = run_stage(
            num=num,
            total=len(STAGES),
            script_rel=script_rel,
            name=name,
            project_root=project_root,
            logs_dir=logs_dir,
        )

        results.append({
            "num": num,
            "name": name,
            "status": status,
            "duration": duration,
            "lines": lines,
        })

        if status == "FAILED":
            has_failure = True
            print("\n" + red("!" * 78))
            print(red(f" [X] STAGE {num} FAILED: {name}"))
            print(red("!" * 78))
            print(red(f"Last 20 lines of output from {script_rel}:"))
            print(gray("-" * 78))
            tail = lines[-20:] if len(lines) >= 20 else lines
            for tline in tail:
                print(f"  {tline}")
            print(gray("-" * 78))
            print(f"Full execution log: {bold(str(log_path.resolve()))}")

            if not args.keep_going:
                print("\n" + red("Halting pipeline due to error. (Use --keep-going to override)"))
                total_time = time.perf_counter() - pipeline_start
                print_summary_table(results, total_time)
                print_key_results(results)
                sys.exit(1)

    total_time = time.perf_counter() - pipeline_start
    print_summary_table(results, total_time)
    print_key_results(results)

    if has_failure:
        sys.exit(1)
    else:
        print(green("All pipeline stages completed successfully!\n"))
        sys.exit(0)


if __name__ == "__main__":
    main()
