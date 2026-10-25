#!/usr/bin/env bash
# 持续观察 GPU 是否"真空闲"。
# 背景：邻项目是多个小任务串行，GPU 呈"满载约3分钟、空约1分钟"的锯齿形，
# 单次采样极易误判任务间隙为结束。本脚本连续采样 WATCH_MINUTES 分钟
# （默认 8 分钟 ≈ 覆盖两个完整周期），任一样本看到占用即判"仍在跑"。
# 双条件：显存占用 < THRESH_MB 且无 compute 进程。
# 退出码：0 = SUSTAINED-FREE（持续空闲，可信）；1 = STILL-BUSY。
set -u
WATCH_MINUTES=${WATCH_MINUTES:-8}
INTERVAL=${INTERVAL:-20}
THRESH_MB=${THRESH_MB:-2000}

samples=$(( WATCH_MINUTES * 60 / INTERVAL ))
busy_seen=0
echo "[watch] window=${WATCH_MINUTES}min interval=${INTERVAL}s thresh=${THRESH_MB}MiB"
for i in $(seq 1 "$samples"); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  apps=${apps:-0}
  ts=$(date +%H:%M:%S)
  if [ "${used:-99999}" -gt "$THRESH_MB" ] || [ "$apps" -gt 0 ]; then
    echo "[watch] $ts sample $i/$samples BUSY (used=${used}MiB compute_apps=$apps)"
    busy_seen=1
    break
  fi
  echo "[watch] $ts sample $i/$samples free (used=${used}MiB compute_apps=$apps)"
  [ "$i" -lt "$samples" ] && sleep "$INTERVAL"
done
if [ "$busy_seen" -eq 0 ]; then
  echo "[watch] VERDICT: SUSTAINED-FREE"
  exit 0
fi
echo "[watch] VERDICT: STILL-BUSY (task gap detected, not a real end)"
exit 1
