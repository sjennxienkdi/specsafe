# SpecSafe

[English](README.md) | **简体中文**

**推测解码正确性差分测试框架** —— 面向混合 GDN/Attention 架构
（Qwen3.5 / Qwen3.8 / Qwen3-Next 系）在 vLLM / SGLang 上的 speculative decoding，
自动生成接受/拒绝/混合场景，验证六条行为不变量，检测状态漂移、
graph 几何错配与 batch 重排错误。

## 为什么需要它

推测解码的加速数字满天飞，正确性审计却几乎空白：

- vLLM 社区自己的 [RFC #28135](https://github.com/vllm-project/vllm/issues/28135)
  （2025-11，推测解码 CI 回归测试）承认现有 e2e 测试只做 66-80% 的文本匹配率
  阈值、阈值任意、无接受率跟踪——该 RFC 挂 stale 关闭，从未实现
- [SGLang #35150](https://github.com/sgl-project/sglang/issues/35150)（open，
  2026-08）：Qwen3.8 + DSpark **forced-reject 不无损**——强制拒绝草稿时输出仍随
  累积 GDN 状态漂移，报告者未能定位根因
- [vLLM #53051](https://github.com/vllm-project/vllm/issues/53051)（open）：
  prefill 误入 spec-decode CUDA graph → **静默** GDN 状态丢失、输出乱码
- [vLLM #53929](https://github.com/vllm-project/vllm/issues/53929)（open PR）：
  captured uniform layout 被 ragged replay 覆盖的 graph 几何坑

这些 bug 的共同点：**不是单测能覆盖的功能错误，而是状态机/布局不变量被破坏**。
SpecSafe 把这类不变量做成可复用的差分测试。

## 首个实战捕获：vLLM #63593（2026-10-25）

SpecSafe 的 margin-aware 判定在 vLLM 0.32.0 + Qwen3.8-27B-FP8 上从 36 个数值噪声报警中
分离出 **1 个真实状态污染 bug** 并完成最小化：

- **现象**：MTP 推测解码在强制全拒（forced-reject）下偏离 target-only，双侧置信
  margin 0.6-5.9 nat（该栈数值地板 ≤0.5 nat），分歧后 100% 走散——GDN 状态被
  被拒草稿污染的实锤
- **最小复现**：K≥5, n≥8, L≥192（单轴贪心约简，S0-S8 全网格 3 重复 × 双 boot 稳定）；
  受害请求确定（同一 prompt 可二分）、分歧位置随批量移动（n=8→pos154, n=32→pos181）
- **定位**：[#39273](https://github.com/vllm-project/vllm/issues/39273) GDN 拒绝回滚
  bug 家族的 MTP 新成员——该家族此前明确认为 MTP 路径干净
- **报告**：[vllm#63593](https://github.com/vllm-project/vllm/issues/63593)（附一条命令
  ~3 分钟的回归用例，跨 boot 稳定，比家族长 #39273 的跨重启不重现更适合做回归测试）

没有 margin 判定，这个 bug 与数值噪声不可区分——这正是本工具存在的理由。

## 六条不变量

| # | 不变量 | 抓的 bug 类 |
|---|---|---|
| 1 | **all-reject equivalence**：全部草稿被拒 ⇒ 输出 == target-only 解码 | 拒绝路径状态回滚（#35150） |
| 2 | **accept-path equivalence**：无损验证下任意接受模式 ⇒ 输出 == target-only | 接受路径状态折叠/前缀错位 |
| 3 | **permutation invariance**：batch 重排不改任何请求输出 | 槽位索引、compaction 重排 |
| 4 | **split/merge equivalence**：packed 执行 == 逐请求独立执行 | ragged/padded 布局（#53051） |
| 5 | **state isolation**：A 的接受/拒绝不影响 B 的输出与状态 | recurrent/conv state 串槽 |
| 6 | **graph geometry consistency**：graph 制度切换（captured/eager）不改输出 | capture/replay 布局错配（#53929） |

## 快速开始（本地，零依赖零 GPU）

```bash
python -m specsafe.cli selftest     # 双防线证明（见下）
python -m specsafe.cli run --engine reference --matrix smoke --repeats 3 --out results/smoke-ref
python -m specsafe.cli gen --matrix static32 --out scenarios/static32.json
```

`selftest` 是 oracle 的**双防线证明**：
1. 六个变异引擎（各模拟一类真实 bug：#35150 拒绝路径漂移、#53051 graph
   静默损坏等）在 margin-aware 判定下**全部 FAIL**——oracle 不洗白真 bug
2. NoisyFlip 测试替身（只在近平局位翻转 token，模拟硬件 tie-break 方差）
   **判 BOUNDARY**（τ=0.5）且 τ=0 时恢复 FAIL——阈值确实在做区分

## 判定语义（oracle v2，三态）

| 判定 | 含义 |
|---|---|
| **PASS** | 各臂 token 流完全一致 |
| **BOUNDARY** | 存在分歧，但每个分歧位至少一侧的 top1-top2 logprob margin ≤ τ（数值边界，噪声地板可解释，不计罪） |
| **FAIL** | 存在双侧 margin > τ 的自信分歧——噪声解释不了的语义差异 |

无 margin 证据的分歧保守判 FAIL。每个用例可 `--repeats N` 重复取多数
（fail > boundary > pass，最差者赢平票）。τ 默认 0.5 nat，由重建对实验
（`--matrix floor`）的 margin 分布定标。

## 云机（vLLM 实测）

见 `cloud/RUNBOOK.md`（上传/预检/矩阵/#35150 形状/回收，全流程命令化）：

```bash
python3 -m specsafe.cli run --engine vllm --matrix static32 \
    --model ~/models/qwen3.8-27b-fp8 --spec-method mtp \
    --max-model-len 4096 --out results/static32-vllm-fp8
```

## 语义说明（重要）

- **standard（无损）验证**：greedy 下 spec 输出**必须**逐 token 等于 target-only
  输出，与接受/拒绝模式无关——这是不变量 1/2 的判定依据
- **synthetic rates**（vLLM 0.32 `rejection_sample_method="synthetic"`）：
  `rates[i]` = 前 i+1 个草稿全被接受的边际概率（非递减约束镜像自
  `vllm/config/speculative.py`）。`[0.0]*K` 强制每步全拒，此时每个发射 token
  仍是 target 的 greedy 选择，因此输出等价性仍是硬要求（#35150 性质）；
  `[1.0]*K` 会接受草稿 token（≠ target 选择），不用于输出等价判定
- 参照引擎支持 `draft_mode=oracle/adversarial` 精确构造全接受/全拒绝；
  vLLM 公开 API 无法强制草稿质量，检查器自动降级到 synthetic 臂

## 架构

```
specsafe/
  scenario.py    # 版本化场景（JSON 可序列化，本地生成 == 云端回放）
  generator.py   # 确定性矩阵生成：selftest / smoke / static32 / drift35150
  invariants.py  # 六不变量检查器（行为层，引擎无关）
  engine/
    reference.py  # 纯 Python 确定性玩具 GDN 引擎（含状态折叠+拒绝回滚）
    mutants.py    # 六个变异引擎，每个对应一类真实 bug
    vllm_offline.py # vLLM 0.32 离线 API 适配器（云机执行，字段名经源码核实）
  report.py      # report.json + summary.md 矩阵报告
  cli.py         # selftest / run / gen
```

分级设计：行为层（输出对照）先行；state 层探针（GDN state norm/max-diff，
需 vLLM 插桩）按需启用——行为层出现漂移才升级。

## 版本兼容

| 引擎 | 版本 | 状态 |
|---|---|---|
| vLLM | 0.32.0 纯净原版 + torch 2.13.0（2026-10-25 夜测） | 六不变量在噪声地板之上全部成立（static32: 8 pass/26 boundary/0 fail ×3 重复）；**forced-reject 高并发存在 1 例稳定真分歧，已最小化为 K≥5/n≥8/L≥192 → [#63593](https://github.com/vllm-project/vllm/issues/63593)** |
| vLLM | 0.32.0 vllm.stock + torch 2.12（2026-10-24 白天） | 发现并定标数值非确定性地板（探针），v1 token 精确相等不可用 |
| SGLang | 对应版本 | 计划中（#35150 原样复现需 SGLang+NVFP4+DSpark 环境） |

**Boundary map 首版**（RTX PRO 6000 Blackwell 96GB × Qwen3.8-27B-FP8）：
重建对地板 margin ≤0.1875（50% 完全并列）；批组成地板 ≤0.125；
静态路径全部分歧 ≤τ=0.5。τ=0.5 对实测地板保持 2.7× 安全余量。

## 实测发现：vLLM 0.32.0 的数值非确定性地板（2026-10-24）

在 RTX PRO 6000 Blackwell 96GB × Qwen3.8-27B-FP8 × vLLM 0.32.0（stock，
Mamba 块约束下 max_num_seqs=64）上，43 用例 token 精确相等判定 37 个
FAIL（smoke 0/6、static32 6/34、drift35150 0/3）。**分歧探针的控制实验
给出了根因**（`results/gpu-20261024/probe_pos0.json`）：

- **target-only 同配置两次独立重建，自身就分歧**（r003@pos17，该位
  margin=0.0——logits 完全并列被任意打破）
- spec 臂 vs target 臂的分歧（margin 0.0-0.375）全部落在该噪声地板内
- 结论：**近平局 argmax 随引擎构建/批组成翻转的地板存在，与投机解码
  无关**；token 精确相等在这类栈上不是可用 oracle，这正是 v2 三态判定
  和 `floor` 矩阵（重建对定标 + 批组成定位）的由来

这个数据本身没有公开先例（vLLM 社区 RFC #28135 批评的"66% 匹配率阈值"
正是没有地板概念的表现）。带 margin 分布的 boundary map 将随 v2 改判
结果一起发布。

## 已知边界

- 离线 API 无法做运行中插入/取消/compaction 生命周期事件——计划中的
  server 模式适配器（连续提交 + 动态到达）覆盖该项
- state 层（final_state）仅参照引擎暴露；真实引擎需插桩构建（W4 分级
  设计：行为层出真分歧才升级）
- τ 的普适性：0.5 nat 起步值来自单卡单模型的重建对，跨硬件/精度需用
  `floor` 矩阵重新定标（这是特性不是缺陷——地板本身是可测量对象）
