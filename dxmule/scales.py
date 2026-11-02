"""Edge-retention scales {k_m}.

This is the only scale axis in DXMulE. A scale k_m in (0, 1] means "keep the
top fraction k_m of edges," ranked by one shared edge score. The same score
vector is reused at every scale (shared scorer, as in MSExplainer). Hard
top-k masks are nested: a smaller ratio keeps a subset of the edges kept by
a larger ratio, because every ratio reads one stable argsort.

Soft masks are the training relaxation. They are not a second scale
definition, and they are not guaranteed to be nested.

Retention is per graph and per stored edge. Directions are not symmetrized.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor


def parse_scales(text: str) -> tuple[float, ...]:
    """Parse a comma-separated list of retention ratios."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("scales must list at least one retention ratio")
    return tuple(float(p) for p in parts)


def edges_to_keep(num_edges: int, retention: float) -> int:
    """How many edges a ratio keeps on a single graph.

    ``ceil(k * n)`` meets the requested fraction, with a floor of one edge
    when the graph is non-empty so the masked subgraph still has a message.
    """
    if num_edges <= 0:
        return 0
    keep = math.ceil(float(retention) * num_edges - 1e-12)
    return max(1, min(num_edges, keep))


def edge_graph_ids(edge_index: Tensor, batch: Tensor | None, num_nodes: int) -> Tensor:
    """Graph id of each edge, using the source node of ``edge_index``."""
    if batch is None:
        return edge_index.new_zeros(edge_index.size(1))
    if batch.numel() != num_nodes:
        raise ValueError(
            f"batch has {batch.numel()} entries for {num_nodes} nodes"
        )
    return batch[edge_index[0]]


def hard_retention_mask(scores: Tensor, edge_batch: Tensor, retention: float) -> Tensor:
    """Hard top-fraction mask for one retention ratio.

    Same keep-count and stable argsort as ``EdgeRetentionScales`` at that
    ratio. ``retention`` is in (0, 1]. A single ratio is allowed here;
    ``EdgeRetentionScales`` still requires two ratios because DXMulE links
    adjacent scales.
    """
    if scores.dim() != 1:
        raise ValueError(f"scores must be rank-1, got {tuple(scores.shape)}")
    if edge_batch.shape != scores.shape:
        raise ValueError(
            f"edge_batch shape {tuple(edge_batch.shape)} != scores {tuple(scores.shape)}"
        )
    ratio = float(retention)
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"retention ratio must lie in (0, 1], got {ratio}")
    out = torch.zeros_like(scores)
    graph_ids = torch.unique(edge_batch.detach()).tolist()
    for graph_id in graph_ids:
        index = (edge_batch == graph_id).nonzero(as_tuple=False).view(-1)
        chosen = _retain_group(
            scores.index_select(0, index),
            ratio,
            hard=True,
            temperature=1.0,
        )
        out = out.index_copy(0, index, chosen)
    return out


class EdgeRetentionScales:
    """Ordered retention ratios and the masks derived from them.

    Ratios are stored strictly increasing (fine subgraph, then coarser
    subgraphs). Callers may pass any order; it is sorted here.
    """

    def __init__(self, ratios: Sequence[float]):
        if len(ratios) < 2:
            raise ValueError(
                "DXMulE needs at least two retention ratios so adjacent "
                "scales can be linked"
            )
        cleaned = tuple(float(k) for k in ratios)
        if any(k <= 0.0 or k > 1.0 for k in cleaned):
            raise ValueError(f"retention ratios must lie in (0, 1], got {cleaned}")
        ordered = tuple(sorted(set(cleaned)))
        if len(ordered) < 2:
            raise ValueError(
                "retention ratios collapse to fewer than two distinct scales"
            )
        self.ks: tuple[float, ...] = ordered

    def __len__(self) -> int:
        return len(self.ks)

    def masks(
        self,
        scores: Tensor,
        edge_batch: Tensor,
        *,
        hard: bool,
        temperature: float = 1.0,
    ) -> list[Tensor]:
        """One mask per ratio, each shaped like ``scores`` (``[num_edges]``).

        ``hard=True`` is an exact top-k (eval). ``hard=False`` is a sigmoid
        around the detached k-th score (train). ``temperature`` is only used
        for the soft mask.
        """
        if scores.dim() != 1:
            raise ValueError(f"scores must be rank-1, got {tuple(scores.shape)}")
        if edge_batch.shape != scores.shape:
            raise ValueError(
                f"edge_batch shape {tuple(edge_batch.shape)} != scores "
                f"{tuple(scores.shape)}"
            )
        if temperature <= 0:
            raise ValueError("retention temperature must be positive")

        built = [
            self._mask_one(scores, edge_batch, k, hard=hard, temperature=temperature)
            for k in self.ks
        ]
        return built

    def _mask_one(
        self,
        scores: Tensor,
        edge_batch: Tensor,
        retention: float,
        *,
        hard: bool,
        temperature: float,
    ) -> Tensor:
        out = torch.zeros_like(scores)
        # unique() on a long tensor does not need to be differentiated.
        graph_ids = torch.unique(edge_batch.detach()).tolist()
        for graph_id in graph_ids:
            index = (edge_batch == graph_id).nonzero(as_tuple=False).view(-1)
            chosen = _retain_group(
                scores.index_select(0, index),
                retention,
                hard=hard,
                temperature=temperature,
            )
            out = out.index_copy(0, index, chosen)
        return out


def _retain_group(
    scores: Tensor,
    retention: float,
    *,
    hard: bool,
    temperature: float,
) -> Tensor:
    num_edges = scores.numel()
    keep = edges_to_keep(num_edges, retention)
    if keep == 0:
        return torch.zeros_like(scores)

    # Stable argsort so a smaller keep-count is a prefix of a larger one,
    # including when several edges share a score.
    order = torch.argsort(scores.detach(), descending=True, stable=True)
    if hard:
        mask = torch.zeros_like(scores)
        return mask.index_fill(0, order[:keep], 1.0)

    threshold = scores.detach().index_select(0, order[keep - 1:keep])
    return torch.sigmoid((scores - threshold) / temperature)
