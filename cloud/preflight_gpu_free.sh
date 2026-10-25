#!/usr/bin/env bash
# GPU 空闲门：被其他项目占用时拒绝开跑（防止干扰在跑实验 / 抢显存）
set -euo pipefail
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
MAX=${MAX_USED_MB:-2000}
echo "GPU memory used: ${USED} MiB (gate: ${MAX})"
if [ "$USED" -gt "$MAX" ]; then
  echo "GPU BUSY — another project is using it. Abort." >&2
  exit 3
fi
echo "GPU free — safe to run."
