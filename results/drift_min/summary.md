# drift-k7 最小复现结果（2026-10-25 13:42-15:12，westb 实例）

环境：vLLM 0.32.0（纯净）+ torch 2.13.0+cu130 × Qwen3.8-27B-FP8 ×
RTX PRO 6000 Blackwell 96GB（driver 580.82.09）× SpecSafe v0.2.0
（tau=0.5；每例 3 重复；S0 跨 2 个独立 boot）。

## 约简表（S0-S8）

| 步 | K | n | L | 判定 | fail 类分歧 |
|---|---|---|---|---|---|
| S0 | 7 | 32 | 256 | **FAIL 3/3**（boot ×2） | r006@154, m 5.875/1.375, ratio .265 |
| S1 | 7 | 8 | 256 | **FAIL 3/3** | r006@154, m 0.875/5.75, ratio 1.0 |
| S2 | 7 | 32 | 128 | boundary×3 | —（L 杀死） |
| S3 | 5 | 32 | 256 | **FAIL 3/3** | r006@181, m 3.875/5.25, ratio 1.0 |
| S4 | 7 | 1 | 256 | boundary（1/3 fail） | r006@181, m 3.5/5.31, ratio 1.0 |
| S5 | 5 | 8 | 256 | **FAIL 3/3** | r006@154, m 0.875/5.75, ratio 1.0 |
| S6 | 5 | 4 | 256 | boundary×3 | —（n 杀死） |
| S7 | 5 | 8 | 192 | **FAIL 3/3** | r006@154, m 0.875/5.75, ratio 1.0 |
| S8 | 3 | 8 | 256 | boundary | —（K 杀死） |

## 结论

- **最小稳定组合：K=5, n=8, L=192**；杀死边界 K≤3 / n≤4 / L≤128
- **S4 混合型判别**：单请求偶发（1/3）+ n=8 必然（3/3）——batch 是放大器，
  根因在拒绝回滚路径的 GDN 状态处理
- **受害者确定性**：全部 fail 均为 r006（prompt 见 scenarios/drift35150.json）
- **走散型**（ratio 1.0）+ 双侧自信 margin（0.6-5.9 nat，地板 0.5）——
  状态污染实锤，非近平局翻转
- 一条命令复现：`python cloud/drift_min_run.py --case S7 --k 5 --n 8 --len 192 --selector first8`

文件：S*__K*_n*_L*__boot*.json ×9（含双方完整 token IDs、每重复分歧明细、
manifest/boot uuid）。
