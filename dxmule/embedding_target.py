"""Unmasked graph embedding used as the IDEA / DXMulE alignment target.

The target has to be the unmasked embedding of the batch being trained.
``DataLoader(shuffle=True)`` draws a new order every iteration, so a list
cached on the first pass and indexed by batch position pairs a graph with
another graph's embedding. Both entry points call ``unmasked_embedding_target``
on the current batch instead.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.explain.algorithm.utils import clear_masks
from torch_geometric.nn import global_mean_pool


def unmasked_embedding_target(model, x: Tensor, edge_index: Tensor, batch: Tensor | None) -> Tensor:
    """Pooled unmasked embedding of the graphs in this batch.

    Clears an explain hook left on ``model`` so the pool is not taken under
    a training mask. The module is set to eval for this forward and restored
    afterward. The returned tensor is detached.
    """
    clear_masks(model)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            node = model.embedding(x, edge_index)
            if batch is None:
                pooled = node.mean(dim=0, keepdim=True)
            else:
                pooled = global_mean_pool(node, batch)
    finally:
        if was_training:
            model.train()
    return pooled.detach()
