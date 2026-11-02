"""Hard-mask fidelity of an edge explanation against the frozen GNN.

ACC@p and Fidelity@p below are computed by this module. They are a
protocol for these entry points, not a transcription of a published table.

For one graph, let ``yhat`` be the frozen GNN's argmax on the unmasked
graph, and let ``p(yhat)`` be the softmax probability of that class.

- ACC@p is 1 when the argmax on the hard top-p subgraph equals ``yhat``.
  The mean over graphs is prediction preservation. It is not ground-truth
  edge accuracy (that is ROC-AUC, precision, and recall).
- Fidelity+@p (necessity) is ``p_full(yhat) - p_drop(yhat)``, where the
  drop graph keeps the complement of the hard top-p mask.
- Fidelity-@p (sufficiency) is ``p_full(yhat) - p_keep(yhat)``, where the
  keep graph retains only the hard top-p edges.

Hard top-p uses ``dxmule.scales.edges_to_keep`` (``ceil(p * n_edges)``,
at least one edge when the graph is non-empty) and a stable descending
argsort of the explanation score. Higher score means more important.

``GCNConv`` adds self-loops that are not rows of the stored ``edge_index``.
PyG's explain hook leaves those loop messages at weight 1, so a mask never
removes them. Fidelity forwards use that hook with ``apply_sigmoid=False``
on a 0/1 mask.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks

DEFINITION = (
    "ACC@p is the fraction of graphs whose frozen-GNN argmax on the hard "
    "top-p subgraph equals the unmasked argmax. "
    "Fidelity+@p (necessity) is the mean drop in softmax probability of that "
    "unmasked class when the hard top-p edges are removed. "
    "Fidelity-@p (sufficiency) is the mean drop when only the hard top-p "
    "edges are kept. "
    "Hard top-p keeps ceil(p * n_edges) edges, at least 1 if the graph has "
    "edges, ranked by the shared explanation score with a stable argsort, "
    "per graph. This is dxmule.scales.edges_to_keep. "
    "GCNConv self-loops are not part of the stored edge_index; the explain "
    "hook leaves those loop messages at weight 1. "
    "Ground-truth edge labels are not used. "
    "IDEA and DXMulE training objectives are different and are not fidelity."
)


def _as_row(logits: Tensor) -> Tensor:
    if logits.dim() == 1:
        return logits.unsqueeze(0)
    if logits.dim() == 2 and logits.size(0) == 1:
        return logits
    raise ValueError(f"expected logits for one graph, got {tuple(logits.shape)}")


def graph_fidelity(
    full_logits: Tensor,
    keep_logits: Tensor,
    drop_logits: Tensor,
) -> dict[str, float]:
    """Prediction preservation and both fidelity directions for one graph."""
    full = _as_row(full_logits)
    keep = _as_row(keep_logits)
    drop = _as_row(drop_logits)
    full_prob = full.softmax(dim=-1)
    pred = int(full_prob.argmax(dim=-1).item())
    p_full = float(full_prob[0, pred])
    p_keep = float(keep.softmax(dim=-1)[0, pred])
    p_drop = float(drop.softmax(dim=-1)[0, pred])
    keep_pred = int(keep.softmax(dim=-1).argmax(dim=-1).item())
    return {
        "acc": 1.0 if keep_pred == pred else 0.0,
        "fidelity_plus": p_full - p_drop,
        "fidelity_minus": p_full - p_keep,
    }


def _mask_dtype(reference: Tensor) -> torch.dtype:
    if reference.is_floating_point():
        return reference.dtype
    return torch.float32


def logits_under_mask(
    model,
    x: Tensor,
    edge_index: Tensor,
    batch,
    edge_mask: Tensor,
) -> Tensor:
    """GNN logits with a hard 0/1 edge mask. Self-loops stay weight 1."""
    if edge_mask.dim() != 1 or edge_mask.numel() != edge_index.size(1):
        raise ValueError(
            f"edge mask length {edge_mask.numel()} != {edge_index.size(1)} edges"
        )
    try:
        set_masks(model, edge_mask, edge_index, apply_sigmoid=False)
        return model(x, edge_index, batch)
    finally:
        clear_masks(model)


def edge_mask_affects_logits(model, x: Tensor, edge_index: Tensor, batch) -> bool:
    """Whether an all-zero edge mask changes logits relative to an all-one mask."""
    num_edges = int(edge_index.size(1))
    if num_edges == 0:
        return False
    ones = torch.ones(num_edges, device=x.device, dtype=_mask_dtype(x))
    zeros = torch.zeros_like(ones)
    with_ones = logits_under_mask(model, x, edge_index, batch, ones)
    with_zeros = logits_under_mask(model, x, edge_index, batch, zeros)
    return not torch.allclose(with_ones, with_zeros)


def score_hard_mask(model, x, edge_index, batch, full_logits: Tensor, hard_mask: Tensor):
    """Fidelity of one hard mask. ``None`` when the graph has no edges."""
    if edge_index.size(1) == 0 or hard_mask.numel() == 0:
        return None
    mask = hard_mask.detach().to(device=x.device, dtype=_mask_dtype(x)).view(-1)
    keep = logits_under_mask(model, x, edge_index, batch, mask)
    drop = logits_under_mask(model, x, edge_index, batch, 1.0 - mask)
    return graph_fidelity(full_logits, keep, drop)


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


class FidelityAccumulator:
    """Means of ACC@p, Fidelity+@p, and Fidelity-@p at fixed ratios."""

    def __init__(self, ratios):
        self.ratios = tuple(float(r) for r in ratios)
        self._acc = {ratio: [] for ratio in self.ratios}
        self._plus = {ratio: [] for ratio in self.ratios}
        self._minus = {ratio: [] for ratio in self.ratios}
        self.skipped_no_edges = 0
        self.skipped_nonfinite = 0

    def add(self, ratio: float, stats: dict[str, float]) -> None:
        self._acc[ratio].append(float(stats["acc"]))
        self._plus[ratio].append(float(stats["fidelity_plus"]))
        self._minus[ratio].append(float(stats["fidelity_minus"]))

    def lines(self) -> list[str]:
        rows = []
        for ratio in self.ratios:
            rows.append(
                "FIDELITY p={:.2f} n={} acc={:.6f} fid_plus={:.6f} fid_minus={:.6f}".format(
                    ratio,
                    len(self._acc[ratio]),
                    _mean(self._acc[ratio]),
                    _mean(self._plus[ratio]),
                    _mean(self._minus[ratio]),
                )
            )
        return rows
