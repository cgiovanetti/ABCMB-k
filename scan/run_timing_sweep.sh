#!/bin/bash
set -e
JID=$(cat bench/.jobid)
for B in 32 64 128; do
  echo "=========== B=$B start $(date +%H:%M:%S) ==========="
  srun --jobid=$JID --ntasks=1 --cpus-per-task=32 --gpus-per-task=4 bash -c \
    "module load conda && conda activate actdr6 && export PYTHONPATH=/pscratch/sd/c/carag/ABCMB-k:\$PYTHONPATH && GPS_LMAX=2508 GPS_B=$B GPS_KCHUNK=100 GPS_LENSING=1 GPS_SHARD=1 python -u scan/grad_prod_shape.py" \
    > scan/results/grad_timing_B${B}.log 2>&1
  echo "=========== B=$B done $(date +%H:%M:%S) ==========="
  grep -E "RESULT|COLD|WARM|Traceback|RESOURCE_EXHAUSTED" scan/results/grad_timing_B${B}.log || true
done
echo "ALL DONE $(date +%H:%M:%S)"
