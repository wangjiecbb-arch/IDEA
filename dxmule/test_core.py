"""CPU checks for retention masks and the cross-scale prototype map.

Run from the repo root: ``python -m dxmule.test_core``
These checks do not load a dataset and do not measure explanation quality.
"""

from __future__ import annotations

import unittest

import torch

from dxmule.cross_scale import CrossScalePrototypeLink, prototype_assignment
from dxmule.scales import EdgeRetentionScales, edges_to_keep


class ScaleTests(unittest.TestCase):
    def test_keep_counts_are_monotonic(self):
        self.assertEqual(edges_to_keep(10, 0.1), 1)
        self.assertEqual(edges_to_keep(10, 1.0), 10)
        self.assertEqual(edges_to_keep(0, 0.5), 0)
        counts = [edges_to_keep(17, k) for k in (0.1, 0.3, 0.5, 0.7, 0.9)]
        self.assertEqual(counts, sorted(counts))

    def test_ratios_are_sorted_unique_and_checked(self):
        scales = EdgeRetentionScales((0.9, 0.1, 0.5, 0.1))
        self.assertEqual(scales.ks, (0.1, 0.5, 0.9))
        with self.assertRaises(ValueError):
            EdgeRetentionScales((0.2,))
        with self.assertRaises(ValueError):
            EdgeRetentionScales((0.0, 0.5))
        with self.assertRaises(ValueError):
            EdgeRetentionScales((0.2, 1.2))

    def test_hard_masks_are_nested_per_graph(self):
        scores = torch.tensor([0.1, 0.4, 0.4, 0.2, 5.0, 4.0, 3.0, 2.0])
        edge_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        scales = EdgeRetentionScales((0.25, 0.5, 1.0))
        masks = scales.masks(scores, edge_batch, hard=True)
        for finer, coarser in zip(masks, masks[1:]):
            self.assertTrue(torch.all(finer <= coarser))
        # Each graph keeps its own top edges. Graph 0's largest score is index 1
        # (tie at 0.4 broken by stable argsort toward the earlier index? )
        # Stable descending argsort keeps the first max when values tie.
        # Graph 0 values: idx 1 and 2 are both 0.4; stable sort puts idx 1 first.
        self.assertEqual(int(masks[0][:4].sum()), 1)
        self.assertEqual(int(masks[0][1].item()), 1)
        self.assertEqual(int(masks[0][5].item()), 0)
        # Graph 1's top edge is the 5.0 at local index 0 → global index 4.
        self.assertEqual(int(masks[0][4].item()), 1)

    def test_soft_masks_keep_the_score_order(self):
        # A soft mask is a strictly increasing function of the shared scores,
        # so every ratio has the same ranking. ROC-AUC cannot separate them.
        scores = torch.tensor([0.2, 1.4, -0.3, 0.7, 0.1])
        edge_batch = torch.zeros(5, dtype=torch.long)
        scales = EdgeRetentionScales((0.4, 0.8))
        order = torch.argsort(scores, descending=True, stable=True)
        for mask in scales.masks(scores, edge_batch, hard=False, temperature=0.7):
            self.assertTrue(torch.equal(torch.argsort(mask, descending=True, stable=True), order))

    def test_soft_mask_reaches_the_scores(self):
        scores = torch.tensor([0.2, -0.4, 1.5, 0.7], requires_grad=True)
        edge_batch = torch.zeros(4, dtype=torch.long)
        scales = EdgeRetentionScales((0.5, 1.0))
        soft = scales.masks(scores, edge_batch, hard=False, temperature=1.0)
        soft[0].sum().backward()
        self.assertIsNotNone(scores.grad)
        self.assertGreater(float(scores.grad.abs().sum()), 0.0)


class CrossScaleTests(unittest.TestCase):
    def test_initial_map_is_diagonal_and_row_stochastic(self):
        link = CrossScalePrototypeLink(codebook_size=8, shell_weight=0.1)
        coupling = link.coupling()
        self.assertTrue(torch.allclose(coupling.sum(dim=-1), torch.ones(8), atol=1e-5))
        self.assertTrue(torch.equal(coupling.argmax(dim=-1), torch.arange(8)))

    def test_link_trains_affinity_and_embeddings(self):
        torch.manual_seed(0)
        batch, dim, k = 4, 6, 5
        link = CrossScalePrototypeLink(codebook_size=k, shell_weight=0.1)
        codebook = torch.randn(k, dim, requires_grad=True)
        fine = torch.randn(batch, dim, requires_grad=True)
        mid = torch.randn(batch, dim, requires_grad=True)
        coarse = fine.detach() + 0.3 * torch.randn(batch, dim)
        coarse.requires_grad_(True)
        loss = link((fine, mid, coarse), codebook)
        loss.backward()
        self.assertGreater(float(link.affinity.grad.abs().sum()), 0.0)
        self.assertGreater(float(fine.grad.abs().sum()), 0.0)
        self.assertGreater(float(codebook.grad.abs().sum()), 0.0)
        self.assertIn("topdown_kl", link.last_stats)
        self.assertTrue(torch.isfinite(loss).item())

    def test_identical_scales_have_finite_loss(self):
        torch.manual_seed(1)
        link = CrossScalePrototypeLink(codebook_size=4, shell_weight=1.0)
        codebook = torch.randn(4, 3)
        emb = torch.randn(3, 3)
        loss = link((emb, emb.clone()), codebook)
        self.assertTrue(torch.isfinite(loss).item())

    def test_assignment_is_a_distribution(self):
        emb = torch.randn(3, 5)
        codebook = torch.randn(7, 5)
        assign = prototype_assignment(emb, codebook)
        self.assertEqual(assign.shape, (3, 7))
        self.assertTrue(torch.allclose(assign.sum(dim=-1), torch.ones(3), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
