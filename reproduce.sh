#!/usr/bin/env bash
# Runs the whole Copilot end to end (same commands as the README Reproduce section). Run from the repo root with the venv active.
set -euo pipefail
mkdir -p outputs/logs
python generate_data.py
python src/part_a_clean.py   | tee outputs/logs/part_a_clean.txt
python src/part_a_validate.py | tee outputs/logs/part_a_validate.txt
python src/part_a_sql.py     | tee outputs/logs/part_a_sql.txt
python src/part_a_pandas.py  | tee outputs/logs/part_a_pandas.txt
python src/part_b_rfm_cluster.py | tee outputs/logs/part_b_rfm_cluster.txt
python src/part_b_lapse.py   | tee outputs/logs/part_b_lapse.txt
python src/part_c_triage.py  | tee outputs/logs/part_c_triage.txt
python src/part_d_rag.py     | tee outputs/logs/part_d_rag.txt
python src/part_d_agent.py   | tee outputs/logs/part_d_agent.txt
