#!/bin/bash
# Tuning with W parallel workers in one allocation (protocol: src/harness/tune.py).
#   sbatch -J NAME [resources] slurm/tune.sbatch configs/tune_X.yaml W
# Phases: (1) the --smoke run of the same config, which stops the job if any fit
# fails; (2) stage 1 with W workers; (3) stage 2 with W workers; (4) one plain pass
# that computes anything still missing, selects the winner and writes winner.yaml.
# Every fit is its own JSON file, so a job killed at its walltime resumes when
# resubmitted. Threads per worker: THREADS, default CPUs / W.
# Output: results/bench/NAME_JOBID.out, worker lines prefixed [s<stage> w<worker>].
#SBATCH --output=results/bench/%x_%j.out

cd "${REPO:-$HOME/uncertainty-sdg}" || exit 1
source "${CONDA_SH:-$HOME/mf/etc/profile.d/conda.sh}" && conda activate synth2 || exit 1
PY=${PY:-python}
CONFIG=$1; W=${2:-4}
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG"; exit 1; }
CPUS=${SLURM_CPUS_PER_TASK:-$W}
T=${THREADS:-$(( CPUS / W > 0 ? CPUS / W : 1 ))}
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
export LOKY_MAX_CPU_COUNT=$T  # caps joblib pools (ctgan's DataTransformer uses n_jobs=-1)
export TABPFN_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "job ${SLURM_JOB_ID:-local} ${SLURM_JOB_NAME:-} $(date -Is)  node $(hostname)" \
     "cpu: $(lscpu | sed -n 's/^Model name: *//p')  code $(git describe --always --dirty)"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null; then
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
fi
echo "config $CONFIG  workers $W  threads per worker $T"; echo

if [ "${SMOKE:-1}" = 1 ]; then
  echo "== smoke run"
  $PY -m src.harness.tune --config "$CONFIG" --smoke || { echo "smoke run failed: stopping"; exit 1; }
  echo
fi
for STAGE in 1 2; do
  echo "== stage $STAGE: $W workers  $(date -Is)"
  for i in $(seq 1 "$W"); do
    $PY -m src.harness.tune --config "$CONFIG" --worker "$STAGE" 2>&1 \
      | sed -u "s/^/[s$STAGE w$i] /" &
  done
  wait
done
echo; echo "== final pass  $(date -Is)"
$PY -m src.harness.tune --config "$CONFIG" --no-retry-failed
status=$?
echo; echo "finished $(date -Is), exit $status"
exit $status
