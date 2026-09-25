#!/bin/bash
# One array task = WORKERS chunks of replications, run as parallel processes on
# one node. Submit through slurm/submit.sh, which sets partition, QOS and
# resources, and exports CONFIG, R0, R1, CHUNK, WORKERS, THREADS.
#
# Array task t runs chunks t*WORKERS .. t*WORKERS+WORKERS-1 of the range
# R0:R1, chunk c covering replications R0+c*CHUNK .. R0+(c+1)*CHUNK-1. Each
# chunk is one CSV, written atomically by src.harness.run, which skips chunks
# that already exist: resubmitting the same range only runs what is missing.
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/logs/%x_%A_%a.out

: "${CONFIG:?}" "${R0:?}" "${R1:?}" "${CHUNK:?}" "${WORKERS:?}" "${THREADS:?}"
TASK=${SLURM_ARRAY_TASK_ID:-0}
JOB=${SLURM_ARRAY_JOB_ID:-local}

cd "${REPO:-$HOME/uncertainty-sdg}" || exit 1
source "${CONDA_SH:-$HOME/mf/etc/profile.d/conda.sh}" && conda activate synth2 || exit 1

# One thread pool per worker, sized explicitly; no network from compute jobs.
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
export TABPFN_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

name=$(basename "$CONFIG" .yaml)
mkdir -p results/logs
echo "task $TASK on $(hostname) $(date -Is): $WORKERS workers x $THREADS threads," \
     "code $(git describe --always --dirty)"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

pids=(); ranges=()
for ((w = 0; w < WORKERS; w++)); do
  c=$(( TASK * WORKERS + w ))
  a=$(( R0 + c * CHUNK ));  (( a >= R1 )) && break
  b=$(( a + CHUNK ));       (( b > R1 )) && b=$R1
  log=results/logs/${name}_${JOB}_t${TASK}_w${w}.log
  python -m src.harness.run --config "$CONFIG" --reps "$a:$b" --chunk-size "$CHUNK" \
    > "$log" 2>&1 &
  pids+=("$!"); ranges+=("$a:$b")
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "  reps ${ranges[$i]}  ok"
  else echo "  reps ${ranges[$i]}  FAILED, see results/logs/${name}_${JOB}_t${TASK}_w${i}.log"; fail=1
  fi
done
echo "task $TASK finished $(date -Is)"
exit $fail
