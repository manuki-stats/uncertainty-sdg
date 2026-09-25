#!/bin/bash
# Submit the S5 throughput benchmark: every generator at n_org 5000 on the node
# type it would run on. Thread scans run inside one job, so they share a node.
#   slurm/bench_all.sh            submit
#   slurm/bench_all.sh --dry-run  print the sbatch commands only
# Output: results/bench/<label>_<jobid>.out
set -euo pipefail
cd "$(dirname "$0")/.."
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
mkdir -p results/bench
CTGAN=results/tuning/tune_ctgan__350ba10a74/winner.yaml
CPU=(-p lrz-cpu -q cpu)
GPU=(-p lrz-hgx-h100-94x4 -q gpu --gres=gpu:1)

submit() {  # label  time  cpus  "resources"  VAR=value...
  # Variables go through the environment (--export=ALL), not through
  # --export=A=b,C=d, which would split JSON kwargs at their commas.
  local label=$1 time=$2 cpus=$3; local -a res=($4); shift 4
  local cmd=(sbatch --parsable -J "$label" "${res[@]}" --cpus-per-task="$cpus"
             --mem-per-cpu=4G --time="$time" --export=ALL slurm/bench.sbatch)
  echo "$label: $* ; ${cmd[*]}"
  if (( ! DRY )); then echo "  -> job $(env LABEL="$label" "$@" "${cmd[@]}")"; fi
}

# CPU partition. synthpop is single-threaded R: one thread, both r.
submit bench_synthpop_proper 01:00:00 1 "${CPU[*]}" GEN=synthpop 'KWARGS={"proper": true}'  T_LIST=1 'R_LIST=1 2'
submit bench_synthpop_plugin 01:00:00 1 "${CPU[*]}" GEN=synthpop 'KWARGS={"proper": false}' T_LIST=1 'R_LIST=1 2'
# Neural nets: fit cost dominates and does not depend on r, so scan threads at r = 2.
submit bench_tvae_cpu  03:00:00 8 "${CPU[*]}" GEN=tvae 'KWARGS={"enable_gpu": false}' 'T_LIST=1 2 4 8' R_LIST=2
submit bench_ctgan_cpu 12:00:00 8 "${CPU[*]}" WINNER=$CTGAN 'KWARGS={"enable_gpu": false}' 'T_LIST=1 2 4 8' R_LIST=2
# GPU partition: the same networks on an H100, and TabPFN at its package defaults.
submit bench_tvae_gpu   02:00:00 4 "${GPU[*]}" GEN=tvae 'KWARGS={"enable_gpu": true}' T_LIST=4 R_LIST=2
submit bench_ctgan_gpu  04:00:00 4 "${GPU[*]}" WINNER=$CTGAN 'KWARGS={"enable_gpu": true}' T_LIST=4 R_LIST=2
submit bench_tabpfn_gpu 06:00:00 4 "${GPU[*]}" GEN=tabpfn T_LIST=4 'R_LIST=1 2'
