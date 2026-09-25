#!/bin/bash
# One benchmark job: src.harness.check_generator at n_org N_ORG, for every
# thread count in T_LIST and every r in R_LIST, one after another ON THE SAME
# NODE, so thread counts are compared on identical hardware.
# Submitted by slurm/bench_all.sh, which exports:
#   LABEL   name of this benchmark (also the job name)
#   GEN     generator name, or WINNER = path to a tuning winner.yaml
#   KWARGS  JSON generator kwargs (merged over WINNER's if both are given)
#   T_LIST  e.g. "1 2 4 8"      R_LIST  e.g. "1 2"      N_ORG  default 5000
#SBATCH --output=results/bench/%x_%j.out

: "${LABEL:?}" "${T_LIST:?}" "${R_LIST:?}"
[ -n "${GEN:-}${WINNER:-}" ] || { echo "set GEN or WINNER"; exit 1; }
N_ORG=${N_ORG:-5000}; KWARGS=${KWARGS:-"{}"}

cd "${REPO:-$HOME/uncertainty-sdg}" || exit 1
source "${CONDA_SH:-$HOME/mf/etc/profile.d/conda.sh}" && conda activate synth2 || exit 1
export TABPFN_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

echo "benchmark $LABEL  job ${SLURM_JOB_ID:-local}  $(date -Is)"
echo "node $(hostname)  cpu: $(lscpu | sed -n 's/^Model name: *//p')  allocated cpus ${SLURM_CPUS_PER_TASK:-?}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null; then
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
fi
echo "code $(git describe --always --dirty)  n_org $N_ORG  generator ${WINNER:-$GEN}  kwargs $KWARGS"

if [ -n "${WINNER:-}" ]; then SRC=(--winner "$WINNER"); else SRC=(--generator "$GEN"); fi
fail=0
for T in $T_LIST; do
  export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
  for r in $R_LIST; do
    echo; echo "=== $LABEL  threads=$T  r=$r  $(date +%T)"
    python -m src.harness.check_generator "${SRC[@]}" --kwargs "$KWARGS" --n-org "$N_ORG" --r "$r" \
      || { echo "=== FAILED (threads=$T r=$r)"; fail=1; }
  done
done
echo; echo "finished $(date -Is)"
exit $fail
