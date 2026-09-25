#!/bin/bash
# Submit one design cell as a SLURM array job.
#
#   slurm/submit.sh CONFIG START:STOP PROFILE [options]
#
#   PROFILE    cpu       lrz-cpu, QOS cpu
#              gpu       lrz-hgx-h100-94x4, QOS gpu, one H100
#              gpu-mcml  mcml-hgx-h100-94x4, QOS mcml, one H100
#   --chunk N        replications per chunk (= per CSV). Keep it FIXED for a
#                    cell: chunk files are named by their range.     [10]
#   --workers W      parallel processes per array task                [8]
#   --threads T      threads per process; the job asks W*T CPUs      [2]
#   --time T         walltime per task, at most 2-00:00:00       [12:00:00]
#   --mem-per-cpu M                                                   [4G]
#   --dry-run        print the sbatch command, submit nothing
#
# Limits here: 10 running and 50 queued jobs per QOS, and every array task
# counts as a job. Size chunk x workers so the array has at most ~40 tasks.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() { sed -n '2,20p' "$0"; exit 1; }
[ $# -ge 3 ] || usage
CONFIG=$1; RANGE=$2; PROFILE=$3; shift 3
CHUNK=10; WORKERS=8; THREADS=2; TIME=12:00:00; MEM=4G; DRY=0
while [ $# -gt 0 ]; do
  case $1 in
    --chunk) CHUNK=$2; shift 2 ;;
    --workers) WORKERS=$2; shift 2 ;;
    --threads) THREADS=$2; shift 2 ;;
    --time) TIME=$2; shift 2 ;;
    --mem-per-cpu) MEM=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown option $1"; usage ;;
  esac
done
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG"; exit 1; }
R0=${RANGE%:*}; R1=${RANGE#*:}

case $PROFILE in
  cpu)      RES=(-p lrz-cpu -q cpu) ;;
  gpu)      RES=(-p lrz-hgx-h100-94x4 -q gpu --gres=gpu:1) ;;
  gpu-mcml) RES=(-p mcml-hgx-h100-94x4 -q mcml --gres=gpu:1) ;;
  *) echo "unknown profile $PROFILE"; usage ;;
esac

NCHUNK=$(( (R1 - R0 + CHUNK - 1) / CHUNK ))
NTASK=$(( (NCHUNK + WORKERS - 1) / WORKERS ))
if (( NTASK > 50 )); then
  echo "$NTASK array tasks exceed the 50-job queue limit: raise --chunk or --workers"; exit 1
fi
NAME=$(basename "$CONFIG" .yaml)
mkdir -p results/logs

CMD=(sbatch --parsable -J "$NAME" "${RES[@]}" --array=0-$((NTASK - 1))
     --cpus-per-task=$((WORKERS * THREADS)) --mem-per-cpu="$MEM" --time="$TIME"
     --export=ALL,CONFIG="$CONFIG",R0="$R0",R1="$R1",CHUNK="$CHUNK",WORKERS="$WORKERS",THREADS="$THREADS"
     slurm/run_array.sbatch)
echo "$NAME: reps $R0:$R1 -> $NCHUNK chunks of $CHUNK, $NTASK tasks x $WORKERS workers x $THREADS threads ($PROFILE)"
echo "${CMD[*]}"
if (( DRY )); then echo "(dry run, nothing submitted)"; exit 0; fi
JOBID=$("${CMD[@]}")
echo "submitted job $JOBID; logs in results/logs/${NAME}_${JOBID}_*"
