#!/bin/bash
# Driver: loop (B kchunk shard) configs, one fresh python process each (clean
# peak memory + OOM isolation). APPENDS JSON lines to round2_sweep.jsonl.
# Reordered: headline big-B 4-GPU curve first (128/256/512), then kchunk
# sensitivity + OOM checks + single-GPU memory points. (B=64/100/1 already done.)
set -u
cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1

OUT=bench/round2_sweep.jsonl   # append (keep prior B=64/100/1 result)

# B kchunk shard
CONFIGS=(
  "256 100 1"
  "512 48 1"
  "128 100 1"
  "384 100 1"
  "192 100 1"
  "256 48 1"
  "512 100 1"
  "64 100 0"
  "128 100 0"
)

for cfg in "${CONFIGS[@]}"; do
  read -r B KC SH <<< "$cfg"
  echo ">>> config B=$B kchunk=$KC shard=$SH  ($(date +%H:%M:%S))"
  python bench/mem_throughput_sweep.py --B "$B" --kchunk "$KC" --shard "$SH" \
    2> >(tail -3 >&2) | grep '^RESULT' | sed 's/^RESULT //' >> "$OUT"
  echo "    done ($(date +%H:%M:%S))"
done
echo "ALL DONE"
