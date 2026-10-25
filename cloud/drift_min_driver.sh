#!/usr/bin/env bash
# drift-k7 最小复现驱动：按优先序 S4 > S1 > S2 > S3 > S0 逐例跑（每例独立子壳，
# 防单例崩溃带崩全程）。S0 的跨 boot 验证由后续单独一次调用完成（不在本脚本内）。
set -u
cd /root/specsafe
source /root/autodl-tmp/venv/bin/activate
export PATH=/root/autodl-tmp/venv/bin:$PATH
export VLLM_LOGGING_LEVEL=WARNING
export PYTHONPATH=/root/specsafe${PYTHONPATH:+:$PYTHONPATH}

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1
mkdir -p results/drift_min

run_one() {  # tag [extra args...]
  local tag=$1; shift
  echo "=== $(date +%H:%M:%S) CASE $tag $* ===" | tee -a results/drift_min/run.log
  ( nohup /root/autodl-tmp/venv/bin/python cloud/drift_min_run.py \
      --case "$tag" "$@" > "results/drift_min/${tag}.log" 2>&1 < /dev/null & )
  # 等该例完成（进程退出且 FINAL 行出现）
  while ! grep -q "FINAL=" "results/drift_min/${tag}.log" 2>/dev/null; do
    if ! pgrep -f "drift_min_run.py --case ${tag}" > /dev/null; then
      sleep 10   # 进程退了但没出 FINAL —— 等日志落盘再查一次
      grep -q "FINAL=" "results/drift_min/${tag}.log" 2>/dev/null && break
      echo "!! CASE $tag died without FINAL -- see ${tag}.log" | tee -a results/drift_min/run.log
      return 1
    fi
    sleep 20
  done
  grep "FINAL=" "results/drift_min/${tag}.log" | tail -1 | tee -a results/drift_min/run.log
}

# 优先序：判别步最先
run_one S4
run_one S1
run_one S2
run_one S3
run_one S0
echo "=== ALL CASES DONE $(date +%H:%M:%S) ===" | tee -a results/drift_min/run.log
