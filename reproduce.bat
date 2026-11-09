@echo off
:: reproduce.bat — Runs the complete Instamart Copilot pipeline on Windows.
:: Activate your virtual environment first, then run from the repo root:
::   .venv\Scripts\activate
::   reproduce.bat
::
:: This is a thin wrapper around run_project.py which provides live progress,
:: per-stage logs, timing, and a summary table.
cd /d "%~dp0"
python run_project.py %*
