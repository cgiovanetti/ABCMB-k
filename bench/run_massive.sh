#!/bin/bash
# Massive-neutrino memory probe (single GPU). Ny ~46 -> ~250, so persistent mem
# (∝ Ny×B_local) and transient Jacobian (∝ Ny²×k_chunk×B_local) both grow; the
# k_chunk sweep at fixed B=16 reveals whether the transient is now a real memory
# knob (peak should DROP with smaller k_chunk if so). Appends to round2_massive.jsonl.
set -u
cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT=bench/round2_massive.jsonl
: > "$OUT"

# B kchunk shard
CONFIGS=(
  "8 100 0"
  "16 100 0"
  "16 50 0"
  "16 25 0"
  "32 50 0"
  "32 25 0"
)
for cfg in "${CONFIGS[@]}"; do
  read -r B KC SH <<< "$cfg"
  echo ">>> massive B=$B kchunk=$KC shard=$SH ($(date +%H:%M:%S))"
  python bench/mem_throughput_sweep.py --B "$B" --kchunk "$KC" --shard "$SH" --massive 1 \
    2> >(tail -3 >&2) | grep '^RESULT' | sed 's/^RESULT //' >> "$OUT"
  echo "    done ($(date +%H:%M:%S))"
done
echo "ALL DONE"
