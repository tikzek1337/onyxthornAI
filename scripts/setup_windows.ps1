# Run from the project root in PowerShell.
# If execution policy blocks this script, run:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python scripts\verify_install.py
