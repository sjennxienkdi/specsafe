# SpecSafe 云机执行手册（W3：vLLM v0.32.0 静态路径实测）

> 目标：GPU 空闲窗口内，用 SpecSafe 对 vLLM 0.32.0 静态 K 路径跑全套六不变量 +
> #35150 形状强制拒绝漂移测试，产出 `results/` 报告矩阵。
> 预计 GPU 时间：smoke 15 分钟 / static32 全量 1.5-2.5 小时（27B-FP8）或 20 分钟（0.8B）。
> 本机（无 GPU）已完成：31 单测 + selftest 6/6 + 参照引擎全矩阵绿。

## 0. 前置条件与注意（云机状态已于 2026-10-24 核实）

- [ ] `nvidia-smi` 确认 GPU 空闲。**⚠️ 邻项目是多个小任务串行，GPU 呈
      "满载约 3 分钟、空约 1 分钟"锯齿形——单次采样必误判**。判定真空闲用
      `bash cloud/watch_gpu_free.sh`（连续 8 分钟、每 20 秒采样、显存<2000MiB
      且无 compute 进程双条件；输出 SUSTAINED-FREE / STILL-BUSY）
- [ ] **vLLM 位置**：`/root/autodl-tmp/venv`（v0.32.0，python 3.12）。
      非交互 SSH 里 `python3` 不在 PATH——一律用全路径
      `/root/autodl-tmp/venv/bin/python` 或 `bash -lc`
- [ ] **⚠️ vLLM 疑似带补丁**：`vllm/config/speculative.py` mtime=2026-10-22
      （L1/GDN adaptive 补丁时期）。正确性基线必须先恢复原版：
      `bash /root/autodl-tmp/bench/swap_vllm.sh`（或
      `/root/autodl-tmp/venv/bin/pip install --force-reinstall --no-deps vllm==0.32.0`），
      跑完 SpecSafe 再按需打回。manifest 会留档 mtime 佐证
- [ ] 模型在 `/root/autodl-tmp/models/`：qwen3.8-27b-fp8 / -int4 / -nvfp4 / drafter-dspark
- [ ] 磁盘：`df -h /root/autodl-tmp`（结果与日志 < 100MB）

## 1. 上传与 CPU 预检（GPU 忙时也可先做，零 GPU 占用——已完成 ✅ 2026-10-24）

> 2026-10-24 已执行：bundle 上传至 `/root/specsafe`，云机 CPU selftest 6/6 OK、
> 参照引擎 smoke 12/12。此后只需在改代码后重新上传。

本机（Git Bash，在 `projects/03_specsafe/` 下）——**注意 MSYS 路径转换，加
`MSYS_NO_PATHCONV=1`**：

```bash
bash cloud/bundle.sh                                  # 产出 projects/dist/specsafe-0.1.0.tar.gz
cd ../..                                              # 仓库根
MSYS_NO_PATHCONV=1 python remote.py upload projects/dist/specsafe-0.1.0.tar.gz /root/specsafe-0.1.0.tar.gz
MSYS_NO_PATHCONV=1 python remote.py cmd 'mkdir -p /root/specsafe && tar xzf /root/specsafe-0.1.0.tar.gz -C /root/specsafe'
MSYS_NO_PATHCONV=1 python remote.py cmd 'bash -lc "cd /root/specsafe && python -m specsafe.cli selftest"'
```

## 2. 环境清单（先采集再开跑）

```bash
MSYS_NO_PATHCONV=1 python remote.py cmd 'bash -lc "cd /root/specsafe && bash cloud/env_manifest.sh | tee results/manifest.txt"'
```

核对：vllm = 0.32.0 且 **speculative.py mtime 应为安装日（非 2026-10-22）**、
torch/CUDA、驱动、`/root/autodl-tmp/models/` 各目录 `.download_complete` 标记。

## 3. GPU 实测（按序，每步产出独立 results 目录）

在云机 `/root/specsafe` 下执行（推荐用一键脚本，自带 GPU 空闲门 + manifest）：

```bash
bash cloud/run_gpu_stage.sh smoke      # 3a 冒烟（K=1，~15 分钟）
bash cloud/run_gpu_stage.sh static32   # 3b 全量 34 用例（1.5-2.5 小时）
bash cloud/run_gpu_stage.sh drift      # 3c #35150 形状强制拒绝漂移
bash cloud/run_gpu_stage.sh all        # 或一次全跑
# 环境变量可覆盖：MODEL=.../qwen3.8-27b-int4 TAG=int4 bash cloud/run_gpu_stage.sh static32
```

手动等价命令（不走脚本时）：

### 3a. 冒烟（~15 分钟，27B-FP8，只跑 K=1）

```bash
MODEL=/root/autodl-tmp/models/qwen3.8-27b-fp8
PY=/root/autodl-tmp/venv/bin/python
cd /root/specsafe
"$PY" -m specsafe.cli run --engine vllm --matrix smoke --filter k1 \
    --model "$MODEL" --spec-method mtp --max-model-len 4096 --gpu-mem 0.85 \
    --out results/smoke-vllm-fp8
```

判读：12→6 个用例。任何 fail/error 先看 `summary.md` 的 mismatch 行与
`resolved_spec_configs`（确认 speculative_config 真被解析成 mtp+synthetic/standard）。
常见坑：`synthetic_acceptance_rates` 需非递减且长度==K（框架已本地校验同款规则）。

### 3b. 全量矩阵（static32，34 用例）

```bash
"$PY" -m specsafe.cli run --engine vllm --matrix static32 \
    --model "$MODEL" --spec-method mtp --max-model-len 4096 --gpu-mem 0.85 \
    --out results/static32-vllm-fp8
```

约 1.5-2.5 小时（每个 K 需要构造 4 个引擎实例：target-only / spec-standard /
spec-synthetic / eager，LRU 缓存上限 2，见适配器说明）。若时间紧，分两段：
先 `--filter i1-`（强制拒绝族，最重要），再跑其余。

### 3c. #35150 形状：强制拒绝长程漂移

```bash
"$PY" -m specsafe.cli run --engine vllm --matrix drift35150 \
    --model "$MODEL" --spec-method mtp --max-model-len 4096 --gpu-mem 0.85 \
    --out results/drift35150-vllm-fp8
```

这是 SGLang #35150（forced-reject 不无损）在 vLLM+GDN 栈上的对应测试：
rates=[0]*K 强制每步全拒、连续 256-512 token，与 target-only 逐 token 对照。
**任何分歧 = 复现了同类 bug，直接截 summary.md 提 vLLM issue。**

### 3d. 可选扩展臂（时间富余时）

```bash
# INT4 量化对照（同一矩阵，差异定位用）
MODEL=/root/autodl-tmp/models/qwen3.8-27b-int4 TAG=int4 bash cloud/run_gpu_stage.sh static32
# dspark 臂（drafter 已在盘）：需在 scenarios JSON 里给 spec_opts.extra 加
#   ["spec_model","/root/autodl-tmp/models/drafter-dspark"] 并 --spec-method dspark
# 小模型快速全量（需先下载 Qwen/Qwen3.5-0.8B，~1.6GB，hf-mirror）
```

## 4. 结果回收（本机）

```bash
MSYS_NO_PATHCONV=1 python remote.py cmd 'cd /root/specsafe && tar czf /root/specsafe-results.tar.gz results'
MSYS_NO_PATHCONV=1 python remote.py cmd 'base64 /root/specsafe-results.tar.gz' > /tmp/b64.txt
base64 -d /tmp/b64.txt > specsafe-results.tar.gz    # Git Bash 本机解码落地
```

回收后核对 `summary.md` 全部矩阵，把结论登记进 WORKLOG.md 与 docs/00 §四。

## 5. 判读与上报规则（与 docs/08 G0 门一致）

- 单用例 fail 且 margin（logprobs top1-top2）在分歧位极小（<0.01）→ 标记
  "数值边界待复核"，重复 3 次取多数
- fail 稳定复现 → 保留 report.json + 场景 JSON，最小化请求数做 reproducer，
  提 vLLM issue（模板见 repo 后续 ISSUE_TEMPLATE）
- 全绿 → 版本兼容表加一行：`vLLM 0.32.0 / mtp1,mtp3 / Qwen3.8-27B-FP8 / PASS`
