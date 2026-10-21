#!/bin/zsh
# Quick reproducibility gate (about 1 minute): the raw file hash, the shared split and every frozen test scoring.
#   zsh code/check_repro.sh            from the repository root
# Full regeneration of every number: zsh code/run_all.sh (40-60 minutes), then this script again.
set -e
cd "$(dirname "$0")"
python3 fetch_data.py          # verifies the SHA-256 of the UCI .xls and rewrites data/raw/*.csv byte-identically
python3 common.py              # rebuilds data/processed/train.csv, test.csv, split_meta.json (same seed, same rows)
python3 freeze.py --check      # every configuration scored on the test set still matches results/frozen_manifest.json
echo "check_repro: OK"
