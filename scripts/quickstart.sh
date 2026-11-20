#!/usr/bin/env bash
# Set up everything and open the dashboard. Run from the project folder: bash scripts/quickstart.sh
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "1/3 Generating demo machine sounds..."
python scripts/make_demo_data.py
echo "2/3 Training (a few seconds)..."
python scripts/train.py
echo "3/3 Evaluating..."
python scripts/evaluate.py
echo
echo "Starting the dashboard on http://localhost:8000  (Ctrl+C to stop)"
uvicorn msad.app:app
