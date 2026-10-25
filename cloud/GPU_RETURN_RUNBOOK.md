# 凌晨 GPU 回来后的执行手册（阶段 2：floor 定标 + v2 改判）

> 背景：10-24 晚实例关机，v2 bundle（specsafe-0.2.0.tar.gz，已在本机
> `projects/dist/`）未及上传。本手册覆盖从租到卡到出结果的全流程。
> 预计 GPU 时间：floor ~15 分钟 + static32×3 ~2-2.5 小时 + drift×3 ~30-45 分钟
> ≈ **3 小时**（嫌长可先跑 floor + static32 的 `--filter i1-`，约 50 分钟出核心结论）。

## 1. 租卡与连接更新（人工）

- [ ] AutoDL 租卡（RTX PRO 6000 / 5090 / 4090 均可，≥48GB 显存；
      96GB 可复现同条件）
- [ ] 把新的 SSH 地址/端口/密码写进仓库根 `remote.py` 的
      `HOST, PORT, USER, PASS`
- [ ] `MSYS_NO_PATHCONV=1 python remote.py check` 验证连通

## 2. 环境判断（两种情形）

### 情形 A：同一实例关机重启（数据盘还在）
`ls /root/autodl-tmp/models/ /root/autodl-tmp/venv/` 都在 → 跳到 §3。

### 情形 B：全新实例（数据盘丢了）——完整重建 ~40-60 分钟
```bash
# 本机遥控执行（每条都加 MSYS_NO_PATHCONV=1 前缀）
python remote.py cmd 'bash -lc "df -h /root/autodl-tmp"'                # 数据盘确认
python remote.py upload bench/setup.sh /root/autodl-tmp/setup.sh       # 复用现有脚手架
python remote.py cmd 'bash -lc "MODELS_DIR=/root/autodl-tmp/models bash /root/autodl-tmp/setup.sh"'  # 权重下载（后台，看 bg_download.log）
python remote.py cmd 'bash -lc "python3 -m venv /root/autodl-tmp/venv && /root/autodl-tmp/venv/bin/pip install vllm==0.32.0 ninja"'
```
注意（白天踩过的坑，脚本已适配但环境要装全）：**ninja 必装**；
全新实例 vllm 是实体目录（非 symlink），run_gpu_stage.sh v2 已支持
（验证 0.32.0 后直接用，退出不动它）。

## 3. 部署 v2（两种情形都要）

```bash
MSYS_NO_PATHCONV=1 python remote.py upload projects/dist/specsafe-0.2.0.tar.gz /root/specsafe-0.2.0.tar.gz
MSYS_NO_PATHCONV=1 python remote.py cmd 'mkdir -p /root/specsafe && tar xzf /root/specsafe-0.2.0.tar.gz -C /root/specsafe'
MSYS_NO_PATHCONV=1 python remote.py cmd 'bash -lc "cd /root/specsafe && python -m specsafe.cli selftest | tail -4"'   # 应见 SELFTEST PASS（CPU 即可）
```

## 4. 阶段 2 三连（云机 /root/specsafe 下）

```bash
export PATH=/root/autodl-tmp/venv/bin:$PATH
PY=/root/autodl-tmp/venv/bin/python
MODEL=/root/autodl-tmp/models/qwen3.8-27b-fp8

# 4a. 地板定标（~15 分钟）：重建对 n=3 + 批组成，产出 margin 分布与 suggested_tau
$PY -m specsafe.cli run --engine vllm --matrix floor \
    --model $MODEL --max-model-len 4096 --out results/v2-floor

# 4b. static32 v2 改判 ×3 重复（~2-2.5 小时；37 个 FAIL 的正式改判）
$PY -m specsafe.cli run --engine vllm --matrix static32 --repeats 3 \
    --margin-tau 0.5 --model $MODEL --max-model-len 4096 --out results/v2-static32

# 4c. drift35150 v2 判别 ×3（~30-45 分钟；#35150 形状的最终定性）
$PY -m specsafe.cli run --engine vllm --matrix drift35150 --repeats 3 \
    --margin-tau 0.5 --model $MODEL --max-model-len 4096 --out results/v2-drift
```

判读（无需人工，报告自带）：
- v2-floor 的 `suggested_tau`：若与 0.5 偏差大，用新 τ 重跑 4b/4c
- 4b 预期：绝大多数原 FAIL → BOUNDARY（地板内）；任何稳定 FAIL
  （3 次重复多数）= 噪声解释不了的真分歧 → issue 素材
- 4c 若出现 margin 大的长程漂移 → #35150 类 bug 在 vLLM stock 的实锤；
  全 BOUNDARY → 升级 state-digest 探针（W4）

## 5. 回收

```bash
MSYS_NO_PATHCONV=1 python remote.py cmd 'cd /root/specsafe && tar czf /root/v2-results.tar.gz results/v2-*'
# 小文件逐个 cat 拉回，大包 base64 或 AutoDL 网盘
```

结果写 WORKLOG + README 版本兼容表（boundary map 首版）。
