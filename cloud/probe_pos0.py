#!/usr/bin/env python
"""pos-0 分歧判别探针：分辨 vLLM spec 臂 vs target 臂的 pos-0 分歧是
(a) 引擎构建数值噪声（graph capture 路径不同、近平局 argmax 翻转）
(b) 真实的草稿状态污染（#35150 类 bug）

三步：
  A. target 引擎构建 #1 -> 4 条固定 prompt 各 48 token + 每 position margin
  B. target 引擎构建 #2（同配置重建） -> 同样生成，对照 A = 构建噪声基线
  C. spec 引擎（mtp K=1, synthetic rates=[0.0] 强制全拒）-> 对照 A

判读：
  A vs B 就有分歧      -> build 噪声存在，矩阵 FAIL 先按"数值边界待复核"
  A == B 且 C 在 pos0 分歧且 margin 大 -> 疑似真实 bug，最小复现成立
"""
import json
import sys

sys.path.insert(0, "/root/specsafe")

from specsafe.engine.vllm_offline import VllmEngine   # noqa: E402
from specsafe.scenario import EngineOptions, ReqSpec  # noqa: E402

PROMPTS = [
    "Explain the difference between momentum and inertia in one paragraph.",
    "Write a Python function that checks if a string of brackets is balanced.",
    "Summarize the causes of the 1929 economic crisis briefly.",
    "List three practical uses of linear attention models in inference.",
]


def arm(model: str, opts: EngineOptions, tag: str) -> list[dict]:
    eng = VllmEngine(model=model, max_model_len=4096, max_num_seqs=64)
    reqs = [ReqSpec(f"r{i:03d}", p, 48) for i, p in enumerate(PROMPTS)]
    res = eng.generate(reqs, opts)
    eng.shutdown()
    out = [{"rid": r.rid, "tokens": r.token_ids, "margins": r.margins} for r in res]
    print(f"[probe] arm {tag} done", flush=True)
    return out


def first_div(a: dict, b: dict):
    for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])):
        if x != y:
            return i
    return None if len(a["tokens"]) == len(b["tokens"]) else min(
        len(a["tokens"]), len(b["tokens"]))


def compare(tag: str, armA: list, armB: list):
    rows = []
    for a, b in zip(armA, armB):
        d = first_div(a, b)
        if d is not None:
            ma = a["margins"][d] if a["margins"] and d < len(a["margins"]) else None
            mb = b["margins"][d] if b["margins"] and d < len(b["margins"]) else None
            rows.append({"rid": a["rid"], "pos": d,
                         "tok_a": a["tokens"][d], "tok_b": b["tokens"][d],
                         "margin_a_at_div": ma, "margin_b_at_div": mb})
    print(f"[probe] {tag}: {len(rows)}/{len(armA)} requests diverge", flush=True)
    return rows


def main():
    model = sys.argv[1]
    out_path = sys.argv[2]
    none = EngineOptions(method="none", seed=0)
    forced = EngineOptions(method="spec", num_speculative_tokens=1, seed=0,
                           rejection_sample_method="synthetic",
                           synthetic_acceptance_rates=(0.0,))

    A = arm(model, none, "target_build1")
    B = arm(model, none, "target_build2")
    C = arm(model, forced, "spec_forced_reject")

    report = {
        "model": model,
        "ab_same_config_rebuild": compare("A vs B (target rebuild)", A, B),
        "ac_spec_vs_target": compare("A vs C (spec vs target)", A, C),
        "verdict_hint": None,
    }
    if report["ab_same_config_rebuild"]:
        report["verdict_hint"] = ("build-noise: identical-config target rebuilds "
                                  "already diverge -> treat matrix FAILs as "
                                  "numeric-boundary, pending 3x repeat")
    elif report["ac_spec_vs_target"]:
        report["verdict_hint"] = ("suspect-real: target deterministic across "
                                  "rebuilds, spec arm diverges -> #35150-class "
                                  "minimal reproducer candidate")
    else:
        report["verdict_hint"] = "all-consistent in probe (48-token window)"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[probe] verdict_hint: {report['verdict_hint']}", flush=True)
    print(f"[probe] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
