#!/usr/bin/env bash
# reproduce.sh — Runs the complete Instamart Copilot pipeline.
# Activate your virtual environment first, then run from the repo root:
#   source .venv/bin/activate
#   bash reproduce.sh
#
# This is a thin wrapper around run_project.py which provides live progress,
# per-stage logs, timing, and a summary table.
set -euo pipefail
python run_project.py "$@"
