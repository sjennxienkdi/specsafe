#!/usr/bin/env bash
# 本机打包：产出可直接上传云机的 specsafe tar（源码布局，PYTHONPATH 即用，免安装）
set -euo pipefail
cd "$(dirname "$0")/.."                      # projects/03_specsafe
OUT=${1:-../dist}
mkdir -p "$OUT"
NAME=specsafe-0.2.0.tar.gz
tar czf "$OUT/$NAME" \
    --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='results' --exclude='.pytest_cache' \
    specsafe tests pyproject.toml README.md WORKLOG.md cloud scenarios
echo "bundled -> $OUT/$NAME"
echo "upload:  python ../../remote.py upload $OUT/$NAME /root/$NAME"
