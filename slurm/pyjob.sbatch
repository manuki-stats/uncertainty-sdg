#!/bin/bash
# Run one Python script or module on a compute node with the thesis env.
#   sbatch -J NAME [resources] slurm/pyjob.sbatch scripts/some_script.py [args]
#   sbatch -J NAME [resources] slurm/pyjob.sbatch -m src.harness.check_generator [args]
# Threads default to the CPUs allocated. Output: results/bench/NAME_JOBID.out
#SBATCH --output=results/bench/%x_%j.out

cd "${REPO:-$HOME/uncertainty-sdg}" || exit 1
source "${CONDA_SH:-$HOME/mf/etc/profile.d/conda.sh}" && conda activate synth2 || exit 1
T=${THREADS:-${SLURM_CPUS_PER_TASK:-1}}
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
export TABPFN_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "job ${SLURM_JOB_ID:-local} ${SLURM_JOB_NAME:-} $(date -Is)  node $(hostname)" \
     "cpu: $(lscpu | sed -n 's/^Model name: *//p')  threads $T  code $(git describe --always --dirty)"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null; then
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
fi
echo "python $*"; echo
python "$@"
status=$?
echo; echo "finished $(date -Is), exit $status"
exit $status
