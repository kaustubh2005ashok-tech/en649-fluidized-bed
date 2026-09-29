#!/usr/bin/env bash
# Full pipeline on the demo bed. For the production run replace --demo/--tag demo
# with the full-bed commands in README.md.
set -e
python verify.py
python run_settle.py --demo
python run_ramp.py   --tag demo --Ug_max 2.0 --rate 1.0
python analysis.py   --tag demo
