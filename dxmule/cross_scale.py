"""Cross-scale prototype links (C2).

IDEA already quantizes a graph and its explanation with a shared pair of
codebooks and matches them with a Wasserstein loss (``RV_model``). That loss
is per scale inside DXMulE. It does not, by itself, relate the prototypes
used at one retention ratio to those used at another.

C2 adds that relation. Adjacent scales (fine k, then the next coarser k)
produce soft assignments on the *same* explanatory codebook. A single
learned row-stochastic matrix ``W`` (shape ``[K, K]``) maps a fine-scale
assignment onto the coarser scale:

    p_fine @ W  ≈  p_coarse

``W`` is global across graphs. It is a codebook-level interaction, not a
per-graph attention and not another edge-retention rule.

Two KL terms use that map. The top-down term treats the coarser assignment
as a detached target (the larger subgraph is the prior for the smaller one).
The reverse term treats the transported fine assignment as a detached target
so the coarser assignment is pulled back toward the same map.

A third term, the shell, is a hypothesis: the embedding added between two
nested scales should land on the prototype mass that the coarser assignment
has and the transported fine assignment does not. It is implemented so the
path can be trained and ablated. It is not an established result.

Assignments here are a softmax over negative squared distance to codebook
vectors. Upstream ``RV_model`` still applies its own softmax to raw
distances inside the Wasserstein term; that path is unchanged.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def prototype_assignment(embedding: Tensor, codebook: Tensor) -> Tensor:
    """Softmax over prototypes. ``embedding`` is ``[B, D]``, codebook ``[K, D]``."""
    if embedding.dim() != 2 or codebook.dim() != 2:
        raise ValueError("assignment expects embedding [B, D] and codebook [K, D]")
    if embedding.size(-1) != codebook.size(-1):
        raise ValueError(
            f"embedding dim {embedding.size(-1)} != codebook dim {codebook.size(-1)}"
        )
    # Squared Euclidean distance. Clamp keeps a slightly negative value,
    # from a numerical inner product, out of the softmax.
    emb_sq = (embedding * embedding).sum(dim=-1, keepdim=True)
    code_sq = (codebook * codebook).sum(dim=-1).unsqueeze(0)
    distance = (emb_sq + code_sq - 2.0 * (embedding @ codebook.t())).clamp(min=0.0)
    scale = distance.detach().median().clamp(min=1e-4)
    return torch.softmax(-distance / scale, dim=-1)


def _kl(target: Tensor, pred: Tensor) -> Tensor:
    """KL(target || pred), averaged over the batch. ``pred`` is trained."""
    pred = pred.clamp(min=1e-8)
    pred = pred / pred.sum(dim=-1, keepdim=True)
    target = target.clamp(min=0.0)
    target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return F.kl_div(pred.log(), target, reduction="batchmean")


class CrossScalePrototypeLink(nn.Module):
    """Global fine-to-coarse map on one shared explanatory codebook."""

    def __init__(self, codebook_size: int, shell_weight: float = 0.1):
        super().__init__()
        if codebook_size < 1:
            raise ValueError("codebook_size must be positive")
        if shell_weight < 0:
            raise ValueError("shell_weight must be non-negative")
        self.codebook_size = codebook_size
        self.shell_weight = float(shell_weight)
        # A peaked identity says "the same prototype persists across scales"
        # before any gradient has moved W. Softmax(4 * I) is diagonal-dominant.
        self.affinity = nn.Parameter(torch.eye(codebook_size) * 4.0)
        self.last_stats: dict[str, float] = {}

    def coupling(self) -> Tensor:
        """Row-stochastic ``[K, K]`` map. Row i is the image of prototype i."""
        return torch.softmax(self.affinity, dim=-1)

    def forward(
        self,
        scale_embeddings: Sequence[Tensor],
        codebook: Tensor,
    ) -> Tensor:
        """Link adjacent scales. Embeddings are ordered fine → coarse.

        ``codebook`` is the explanatory codebook weight ``[K, D]`` (IDEA
        codebook B). Gradients flow into that weight, into ``affinity``, and
        into the scale embeddings.
        """
        if len(scale_embeddings) < 2:
            raise ValueError("cross-scale link needs at least two scales")
        if codebook.size(0) != self.codebook_size:
            raise ValueError(
                f"codebook has {codebook.size(0)} rows, link was built for "
                f"{self.codebook_size}"
            )

        assigns = [prototype_assignment(emb, codebook) for emb in scale_embeddings]
        transport = self.coupling()
        topdown_terms = []
        reverse_terms = []
        shell_terms = []

        for finer, coarser, emb_fine, emb_coarse in zip(
            assigns, assigns[1:], scale_embeddings, scale_embeddings[1:]
        ):
            predicted = finer @ transport
            # Coarser assignment is the top-down target.
            topdown_terms.append(_kl(coarser.detach(), predicted))
            # Transported fine assignment is the reverse target.
            reverse_terms.append(_kl(predicted.detach(), coarser))
            shell_terms.append(
                _shell_loss(emb_fine, emb_coarse, predicted, coarser, codebook)
            )

        topdown = torch.stack(topdown_terms).mean()
        reverse = torch.stack(reverse_terms).mean()
        shell = torch.stack(shell_terms).mean()
        loss = topdown + reverse + self.shell_weight * shell

        with torch.no_grad():
            diag = transport.diag().mean()
            self.last_stats = {
                "topdown_kl": float(topdown.detach()),
                "reverse_kl": float(reverse.detach()),
                "shell_loss": float(shell.detach()),
                "persistence": float(diag),
            }
        return loss


def _shell_loss(
    emb_fine: Tensor,
    emb_coarse: Tensor,
    predicted: Tensor,
    assign_coarse: Tensor,
    codebook: Tensor,
) -> Tensor:
    """Match the between-scale embedding gap to leftover prototype mass.

    The gap uses a detached fine embedding so only the coarser subgraph and
    the codebook are trained by this term. Samples whose leftover mass is
    ~0 contribute nothing (the two assignments already agree).
    """
    delta = emb_coarse - emb_fine.detach()
    leftover = (assign_coarse - predicted).clamp(min=0.0).detach()
    mass = leftover.sum(dim=-1)
    valid = mass > 1e-4
    if not bool(valid.any()):
        return predicted.sum() * 0.0

    target = leftover[valid] / mass[valid].unsqueeze(-1)
    predicted_delta = prototype_assignment(delta[valid], codebook)
    return _kl(target, predicted_delta)


def complement_base_alignment(
    complement_embeddings: Sequence[Tensor],
    full_embedding: Tensor,
    base_codebook: Tensor,
) -> Tensor:
    """Match complement subgraphs to the base codebook's assignment of the full graph.

    Hypothesis, not a measured result: edges left outside the retained
    subgraph line up with IDEA's base prototypes (codebook A), which the
    tokenizer trains as the component removed before explanatory alignment.
    The base codebook is detached, so this term trains the scorer through
    the complement embeddings and leaves codebook A to ``RV_model``.
    """
    if len(complement_embeddings) == 0:
        raise ValueError("complement alignment received no embeddings")
    frozen = base_codebook.detach()
    target = prototype_assignment(full_embedding.detach(), frozen)
    terms = [
        _kl(target, prototype_assignment(emb, frozen))
        for emb in complement_embeddings
    ]
    return torch.stack(terms).mean()
