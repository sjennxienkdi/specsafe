#!/usr/bin/env bash
# 云机环境指纹：与结果一并归档，报告可追溯（docs/08 §环境 manifest 要求）
set -euo pipefail
PY=${PY:-/root/autodl-tmp/venv/bin/python}
[ -x "$PY" ] || PY=python3
echo "== time ==";        date -Is
echo "== gpu ==";         nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
echo "== python ==";      "$PY" --version
echo "== vllm ==";        "$PY" -m pip show vllm 2>/dev/null | sed -n '1,3p' || echo "vllm NOT installed"
echo "== vllm patch check =="
"$PY" - <<'EOF'
import vllm, pathlib, time
p = pathlib.Path(vllm.__file__).parent
print("vllm path:", p)
for f in ("v1/core/sched/processor.py", "config/speculative.py"):
    fp = p / f
    if fp.exists():
        print(f, "mtime:", time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime(fp.stat().st_mtime)))
EOF
echo "== torch ==";       "$PY" -c "import torch; print(torch.__version__, torch.version.cuda)"
echo "== models =="
for d in /root/autodl-tmp/models/*/; do
  [ -f "$d/.download_complete" ] && echo "OK  $d" || echo "??  $d"
done 2>/dev/null || true
echo "== disk ==";        df -h / /root/autodl-tmp 2>/dev/null || df -h /
echo "== specsafe ==";    "$PY" -m specsafe.cli --version 2>/dev/null || echo "specsafe not on path"
