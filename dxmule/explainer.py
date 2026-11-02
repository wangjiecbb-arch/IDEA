"""DXMulE explainer.

Subclasses IDEA's PGExplainer-style ``Idea`` so the shared MLP, concrete
edge sampler, and ``RV_model`` (Wasserstein prototype alignment) stay the
upstream objects. ``train`` and ``forward`` are replaced.

Where {k_m} enters
------------------
One forward of the frozen GNN (no edge mask) feeds the shared MLP. That
produces a single score per edge. ``EdgeRetentionScales`` turns those scores
into a mask per retention ratio. During training the scores are concrete
samples, matching IDEA's sampler. At explanation time the scores are the
raw MLP logits. Each mask is written with PyG ``set_masks`` and the GNN
embedding of that subgraph is pooled. Those pooled vectors are what the
codebooks see.

What is optimized
-----------------
For every scale, IDEA's ``RV_model`` aligns the masked embedding with the
unmasked embedding of the same batch. The scale losses are averaged, so the
``emd`` weight stays on the same scale as a single IDEA call.

C2 (``CrossScalePrototypeLink``) then links adjacent scales on codebook B.
An optional complement term aligns the dropped edges with codebook A.
Label-space contrastive losses from MSExplainer are not part of this
objective. IDEA's move from label space to prototype space is kept.

Node-level explanations are left to the original ``Idea`` class.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import ModelTaskLevel
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import get_embeddings

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dxmule.cross_scale import CrossScalePrototypeLink, complement_base_alignment
from dxmule.scales import EdgeRetentionScales, edge_graph_ids
from explainer.idea import Idea


class DXMulE(Idea):
    """Multi-scale IDEA. Graph-level only."""

    def __init__(
        self,
        epochs: int,
        lr: float = 0.003,
        codebook_size: int = 32,
        codebook_channels: int = 32,
        scales: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
        cross_scale_weight: float = 0.1,
        complement_weight: float = 0.1,
        shell_weight: float = 0.1,
        retention_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            epochs=epochs,
            lr=lr,
            codebook_size=codebook_size,
            codebook_channels=codebook_channels,
            **kwargs,
        )
        self.retention = EdgeRetentionScales(scales)
        self.retention_temperature = float(retention_temperature)
        self.cross_scale_weight = float(cross_scale_weight)
        self.complement_weight = float(complement_weight)
        self.cross_scale = CrossScalePrototypeLink(
            codebook_size=codebook_size,
            shell_weight=shell_weight,
        )
        # Rebuild the optimizer so the new affinity matrix is included.
        # Codebook learning rate matches Idea (0.1); the scorer and the
        # cross-scale map use the explainer learning rate.
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.mlp.parameters(), "lr": lr},
                {"params": self.rv_emd_model.parameters(), "lr": 0.1},
                {"params": self.cross_scale.parameters(), "lr": lr},
            ]
        )

    def reset_parameters(self):
        super().reset_parameters()
        eye = torch.eye(self.cross_scale.codebook_size, device=self.cross_scale.affinity.device)
        self.cross_scale.affinity.data.copy_(eye * 4.0)

    def train(
        self,
        epoch: int,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        emd_target: Tensor,
        index: int | Tensor | None = None,
        **kwargs,
    ):
        del target  # prototype-space objective; label target is unused
        self._require_graph_level()
        if index is not None:
            raise NotImplementedError(
                "DXMulE does not take a node index. Node-level explanations "
                "remain on Idea. TODO: node-level retention."
            )

        batch = kwargs.get("batch", None)
        self.optimizer.zero_grad(set_to_none=True)
        try:
            # Scores come from the unmasked graph, then every scale reuses them.
            node_emb = get_embeddings(model, x, edge_index, **kwargs)[-1]
            temperature = self._get_temperature(epoch)
            logits = self.mlp(self._get_inputs(node_emb, edge_index, index=None)).view(-1)
            scores = self._concrete_sample(logits, temperature)
            edge_batch = edge_graph_ids(edge_index, batch, x.size(0))
            masks = self.retention.masks(
                scores,
                edge_batch,
                hard=False,
                temperature=self.retention_temperature,
            )

            scale_embs = []
            rv_terms = []
            dist_terms = []
            rec_terms = []
            for mask in masks:
                set_masks(model, mask, edge_index, apply_sigmoid=False)
                explained = _pooled_embedding(model, x, edge_index, batch)
                scale_embs.append(explained)
                rv_loss, dist_loss, rec_loss = self.rv_emd_model(explained, emd_target)
                rv_terms.append(rv_loss)
                dist_terms.append(dist_loss)
                rec_terms.append(rec_loss)

            rv_mean = torch.stack(rv_terms).mean()
            dist_mean = torch.stack(dist_terms).mean()
            rec_mean = torch.stack(rec_terms).mean()
            cross = self.cross_scale(
                scale_embs,
                self.rv_emd_model.exp_codebook.embedding.weight,
            )

            if self.complement_weight > 0:
                complements = []
                for mask in masks:
                    set_masks(model, 1.0 - mask, edge_index, apply_sigmoid=False)
                    complements.append(_pooled_embedding(model, x, edge_index, batch))
                complement = complement_base_alignment(
                    complements,
                    emd_target,
                    self.rv_emd_model.base_codebook.embedding.weight,
                )
            else:
                complement = rv_mean.new_zeros(())

            total = (
                self.coeffs["emd"] * rv_mean
                + self.cross_scale_weight * cross
                + self.complement_weight * complement
            )
            total.backward()
            self.optimizer.step()
            self._curr_epoch = epoch
            stats = {
                "loss": float(total.detach()),
                "rqvae_loss": float(rv_mean.detach()),
                "distribution_loss": float(dist_mean.detach()),
                "reconstruction_loss": float(rec_mean.detach()),
                "cross_scale_loss": float(cross.detach()),
                "complement_loss": float(complement.detach()),
            }
            stats.update(self.cross_scale.last_stats)
            return stats
        finally:
            clear_masks(model)

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: int | Tensor | None = None,
        **kwargs,
    ) -> Explanation:
        del target
        self._require_graph_level()
        if self._curr_epoch < self.epochs - 1:
            raise ValueError(
                f"DXMulE has run {self._curr_epoch + 1} of {self.epochs} "
                "epochs. Call algorithm.train(...) through the requested "
                "number of epochs before explaining."
            )

        clear_masks(model)
        try:
            node_emb = get_embeddings(model, x, edge_index, **kwargs)[-1]
            logits = self.mlp(self._get_inputs(node_emb, edge_index, index=None)).view(-1)
            # Same single-edge interface as Idea: sigmoid of the shared logit.
            edge_mask = self._post_process_mask(logits, None, apply_sigmoid=True)

            edge_batch = edge_graph_ids(edge_index, kwargs.get("batch", None), x.size(0))
            soft = self.retention.masks(
                logits,
                edge_batch,
                hard=False,
                temperature=self.retention_temperature,
            )
            hard = self.retention.masks(
                logits,
                edge_batch,
                hard=True,
                temperature=self.retention_temperature,
            )
            soft_mat = torch.stack(soft, dim=0)
            hard_mat = torch.stack(hard, dim=0)

            # Coarsest scale (largest k) is the subgraph closest to the input.
            set_masks(model, soft[-1], edge_index, apply_sigmoid=False)
            coarsest = _pooled_embedding(model, x, edge_index, kwargs.get("batch", None))
        finally:
            clear_masks(model)

        ratios = torch.tensor(self.retention.ks, dtype=edge_mask.dtype, device=edge_mask.device)
        return Explanation(
            edge_mask=edge_mask,
            subgraph_embedding=coarsest,
            scale_edge_masks=soft_mat,
            scale_hard_masks=hard_mat,
            retention_ratios=ratios,
        )

    def _require_graph_level(self) -> None:
        task_level = self.model_config.task_level
        if task_level != ModelTaskLevel.graph:
            raise NotImplementedError(
                "DXMulE retention masks are implemented for graph-level "
                f"explanations (got {task_level}). Use Idea for node-level. "
                "TODO: node-level multi-scale retention."
            )


def _pooled_embedding(
    model: torch.nn.Module,
    x: Tensor,
    edge_index: Tensor,
    batch: Tensor | None,
) -> Tensor:
    node = model.embedding(x, edge_index)
    if batch is None:
        return node.mean(dim=0, keepdim=True)
    return global_mean_pool(node, batch)
