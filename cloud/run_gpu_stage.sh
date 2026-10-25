#!/usr/bin/env bash
# 云机一键阶段脚本（v3）：邻项目队列门 -> GPU 门 -> manifest -> 原子切 stock
# -> 依序跑阶段 -> 退出时恢复到启动前记录的 symlink 状态
# 用法（在 /root/specsafe 下）：
#   bash cloud/run_gpu_stage.sh smoke|static32|drift|all
set -euo pipefail
cd "$(dirname "$0")/.."      # /root/specsafe

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARNING}
export PYTHONPATH=/root/specsafe${PYTHONPATH:+:$PYTHONPATH}
export PATH="/root/autodl-tmp/venv/bin:$PATH"   # torch.compile 子进程需在 PATH 找到 ninja

PY=${PY:-/root/autodl-tmp/venv/bin/python}
MODEL=${MODEL:-/root/autodl-tmp/models/qwen3.8-27b-fp8}
METHOD=${METHOD:-mtp}
MAXLEN=${MAXLEN:-4096}
GPUMEM=${GPUMEM:-0.85}
TAG=${TAG:-$(basename "$MODEL")}
STAGE=${1:?stage: smoke|static32|drift|all}
# 用活跃解释器自报 site-packages（避免误绑到未在用的旧 venv symlink 树）
VLLM_DIR=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
SWAP=/root/autodl-tmp/bench/swap_vllm.sh

QUEUE_PAT="run_adaptiv[e]|phase_adaptiv[e]|g0[b]|run_al[l]|matrix_da[y]|matrix_v[3]|confirm_matri[x]|setup_adaptiv[e]"

# ---- 门 1：邻项目队列进程存活 = 项目仍在跑（哪怕 GPU 空）——不启动、不碰 symlink
if pgrep -f "$QUEUE_PAT" >/dev/null 2>&1; then
  echo "GATE1-FAIL: neighbor queue processes alive:"
  pgrep -af "$QUEUE_PAT" | head -3
  exit 4
fi
# ---- 门 2：GPU 显存空闲
bash cloud/preflight_gpu_free.sh

mkdir -p results
bash cloud/env_manifest.sh > "results/manifest-$TAG.txt" || true

# ---- symlink 管理：记录启动前状态，退出恢复原状 ----
# 三种情形：symlink->vllm.stock（直接用）/ symlink->其他（切 stock，退出恢复）
# / 实体目录（全新实例，无 symlink 方案）：验证原版 0.32.0 后直接用，不碰
PRE_LINK=$(readlink "$VLLM_DIR/vllm" 2>/dev/null || echo "")
restore_link() {
  if [ -n "$PRE_LINK" ] && [ "$PRE_LINK" != "vllm.stock" ]; then
    bash "$SWAP" adaptive >/dev/null 2>&1 || true
    echo "[restore] vllm symlink -> $PRE_LINK"
  else
    echo "[restore] leaving vllm as-is (pre-run: ${PRE_LINK:-solid-dir})"
  fi
}
trap restore_link EXIT

if [ -z "$PRE_LINK" ]; then
  VER=$("$PY" -m pip show vllm 2>/dev/null | sed -n "s/^Version: //p")
  if [ "$VER" != "0.32.0" ]; then
    echo "FRESH-BOX GATE: vllm=$VER (expected 0.32.0 stock); install first:"
    echo "  $PY -m pip install vllm==0.32.0 ninja"
    exit 5
  fi
  echo "[env] fresh box: solid vllm 0.32.0 directory, using as-is"
elif [ "$PRE_LINK" != "vllm.stock" ]; then
  bash "$SWAP" stock || { echo "swap to stock failed"; exit 5; }
fi
# 切换后复核：邻项目没有恰好在此秒启动（若其启动器自行 flip 也无害，我们保守退出）
if pgrep -f "$QUEUE_PAT" >/dev/null 2>&1; then
  echo "GATE1B-FAIL: neighbor queue started during swap -- abort"
  exit 4
fi

run() {  # matrix filter outdir
  local matrix=$1 filter=$2 out=$3
  if [ -n "$PRE_LINK" ]; then   # symlink 方案才有 per-stage 复核；实体目录免检
    local cur
    cur=$(readlink "$VLLM_DIR/vllm" 2>/dev/null || true)
    if [ "$cur" != "vllm.stock" ]; then
      echo "[guard] vllm symlink = ${cur:-?} (expected vllm.stock) -- re-flip before $matrix"
      bash "$SWAP" stock >/dev/null || true
    fi
  fi
  local args=(--engine vllm --matrix "$matrix" --model "$MODEL"
              --spec-method "$METHOD" --max-model-len "$MAXLEN"
              --gpu-mem "$GPUMEM" --out "$out")
  [ -n "$filter" ] && args+=(--filter "$filter")
  "$PY" -m specsafe.cli run "${args[@]}" || true   # 单阶段失败不中断后续
}

case "$STAGE" in
  smoke)    run smoke      k1  "results/smoke-vllm-$TAG" ;;
  static32) run static32  ""  "results/static32-vllm-$TAG" ;;
  drift)    run drift35150 "" "results/drift35150-vllm-$TAG" ;;
  all)      run smoke      k1  "results/smoke-vllm-$TAG"
            run static32  ""  "results/static32-vllm-$TAG"
            run drift35150 "" "results/drift35150-vllm-$TAG" ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac

echo "== stage $STAGE done; summaries =="
for f in results/*/summary.md; do echo "--- $f"; head -8 "$f"; done
