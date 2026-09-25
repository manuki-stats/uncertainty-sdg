#!/bin/bash
# Pack results for the laptop into one archive in your home directory.
#   slurm/pack_results.sh           results/raw only
#   slurm/pack_results.sh --logs    results/raw and results/logs
# Then, on the laptop (cmd, repo root):
#   scp ra95xov2@login.ai.lrz.de:results_latest.tgz %USERPROFILE%\Downloads\
#   tar -xzf %USERPROFILE%\Downloads\results_latest.tgz
# Chunk files never change once written, so unpacking over older copies is safe.
set -euo pipefail
cd "$(dirname "$0")/.."
DIRS=(results/raw); [ "${1:-}" = "--logs" ] && DIRS+=(results/logs)
OUT=$HOME/results_$(date +%Y%m%d_%H%M).tgz
tar -czf "$OUT" "${DIRS[@]}"
ln -sf "$OUT" "$HOME/results_latest.tgz"
echo "$OUT ($(du -h "$OUT" | cut -f1)); also linked as ~/results_latest.tgz"
