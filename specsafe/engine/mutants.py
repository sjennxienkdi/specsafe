"""Mutant engines: one deliberate bug class per invariant.

Each mutant subclasses ReferenceEngine and corrupts exactly one behavior,
emulating a real reported bug. `specsafe selftest` proves that the matching
invariant checker catches each mutant and that the clean engine passes all
six -- this is the local proof that the invariant framework has teeth before
any GPU is involved.
"""
from __future__ import annotations

from typing import Sequence

from ..scenario import EngineOptions
from .reference import VOCAB, ReferenceEngine, _ReqRun


class RejectBonusDrop(ReferenceEngine):
    """SGLang #35150 class: on draft rejection the target's bonus token is
    emitted to the user but NOT folded into the recurrent state. Output
    diverges from target-only under forced reject, drifting over time."""

    name = "mutant:reject_bonus_drop"

    def _commit(self, run: _ReqRun, new_tokens: Sequence[int]) -> None:
        run.emitted.extend(new_tokens)
        if run.last_event[0] == "reject":
            run.model_tokens.extend(new_tokens[:-1])   # bonus token lost
        else:
            run.model_tokens.extend(new_tokens)


class AcceptPathSkip(ReferenceEngine):
    """Accept-path state fold skips one token on full acceptance: the state
    accumulates a per-step offset only along all-accept trajectories."""

    name = "mutant:accept_path_skip"

    def _commit(self, run: _ReqRun, new_tokens: Sequence[int]) -> None:
        run.emitted.extend(new_tokens)
        if run.last_event[0] == "full_accept":
            run.model_tokens.extend(new_tokens[1:])    # first draft lost
        else:
            run.model_tokens.extend(new_tokens)


class BatchOrderLeak(ReferenceEngine):
    """Each request's state is polluted by whatever request was processed
    immediately before it in the batch -- reordering the batch changes each
    request's output (vLLM batch-reorder/compaction bug class)."""

    name = "mutant:batch_order_leak"

    def _pre_round(self, batch: list[_ReqRun]) -> None:
        for prev, cur in zip(batch, batch[1:]):
            if prev.emitted:
                cur.model_tokens.append(prev.emitted[-1] & 0xFF)


class PackedPaddingOffset(ReferenceEngine):
    """Packed/ragged execution computes positions against a padded layout:
    requests in a multi-request batch see a shifted position (vLLM #53051 /
    #53929 packed-vs-independent class). Single-request runs are unaffected."""

    name = "mutant:packed_padding_offset"

    def _position(self, run: _ReqRun, batch_len: int) -> int:
        pos = len(run.model_tokens)
        if batch_len > 1:
            pos += 0x100 + batch_len    # uniform padded-layout offset
        return pos


class StateBleedOnCoexist(ReferenceEngine):
    """When two requests coexist, slot-0's last token is folded into every
    other request's state: request A's dynamics change request B's state
    (GDN state isolation bug class)."""

    name = "mutant:state_bleed"

    def _pre_round(self, batch: list[_ReqRun]) -> None:
        if len(batch) > 1 and batch[0].emitted:
            for other in batch[1:]:
                other.model_tokens.append(batch[0].emitted[-1] & 0xFF)


class GraphRegimeCorrupt(ReferenceEngine):
    """CUDA-graph replay path corrupts state, eager path is correct: outputs
    depend on the graph regime (vLLM #53051 silent GDN state loss /
    #53929 captured-uniform-vs-ragged-replay class)."""

    name = "mutant:graph_regime"

    def _post_step(self, run: _ReqRun, batch_len: int, opts: EngineOptions) -> None:
        if opts.graph_regime == "captured" and run.last_event[0] == "full_accept":
            run.model_tokens.append(run.model_tokens[-1] ^ 0x55)   # replay-corrupt fold


class NoisyFlip(ReferenceEngine):
    """Oracle honesty double, NOT a bug: identical model, but at near-tie
    positions (margin <= 0.25) it emits the runner-up token -- exactly what
    hardware tie-break variance does on real GPUs. A margin-aware oracle
    MUST classify the resulting divergences as BOUNDARY (not FAIL), or it
    is washing real signal away; with tau=0 it must FAIL them."""

    name = "double:noisy_flip"

    def _target_token(self, run: _ReqRun, batch_len: int,
                      prefix_extra: Sequence[int] = ()) -> int:
        tgt = super()._target_token(run, batch_len, prefix_extra)
        idx = len(run.emitted) + len(prefix_extra)
        if self._margin_for(run, idx) <= 0.25:
            return (tgt + 1) % VOCAB
        return tgt


# invariant key -> mutant that only that invariant (plus documented overlaps) must catch
MUTANTS: dict[str, type[ReferenceEngine]] = {
    "all_reject_equivalence": RejectBonusDrop,
    "accept_path_equivalence": AcceptPathSkip,
    "permutation_invariance": BatchOrderLeak,
    "split_merge_equivalence": PackedPaddingOffset,
    "state_isolation": StateBleedOnCoexist,
    "graph_geometry_consistency": GraphRegimeCorrupt,
}
