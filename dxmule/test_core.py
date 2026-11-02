"""CPU checks for retention masks and the cross-scale prototype map.

Run from the repo root: ``python -m dxmule.test_core``
These checks do not load a dataset and do not measure explanation quality.
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from dxmule.cross_scale import CrossScalePrototypeLink, prototype_assignment
from dxmule.embedding_target import unmasked_embedding_target
from dxmule.fidelity import (
    FidelityAccumulator,
    edge_mask_affects_logits,
    graph_fidelity,
    score_hard_mask,
)
from dxmule.matched_auc import (
    build_matched,
    classify_graph_auc,
    load_json,
    summarize_matched_seeds,
    write_auc_record,
)
from dxmule.scales import EdgeRetentionScales, edges_to_keep, hard_retention_mask


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

    def test_hard_retention_mask_matches_scale_rows(self):
        scores = torch.tensor([0.1, 0.4, 0.4, 0.2, 5.0, 4.0, 3.0, 2.0])
        edge_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        scales = EdgeRetentionScales((0.25, 0.5, 1.0))
        masks = scales.masks(scores, edge_batch, hard=True)
        for ratio, mask in zip(scales.ks, masks):
            self.assertTrue(torch.equal(hard_retention_mask(scores, edge_batch, ratio), mask))

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


class _TinyGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = GCNConv(3, 4)
        self.lin = torch.nn.Linear(4, 2)

    def forward(self, x, edge_index, batch=None):
        hidden = self.conv(x, edge_index).relu()
        if batch is None:
            pooled = hidden.mean(dim=0, keepdim=True)
        else:
            pooled = global_mean_pool(hidden, batch)
        return self.lin(pooled)


class FidelityTests(unittest.TestCase):
    def test_graph_fidelity_matches_softmax_drop(self):
        full = torch.tensor([[4.0, 0.0]])
        keep = torch.tensor([[4.0, 0.0]])
        drop = torch.tensor([[0.0, 4.0]])
        stats = graph_fidelity(full, keep, drop)
        self.assertEqual(stats["acc"], 1.0)
        self.assertAlmostEqual(stats["fidelity_minus"], 0.0)
        p_full = float(torch.softmax(full, dim=-1)[0, 0])
        p_drop = float(torch.softmax(drop, dim=-1)[0, 0])
        self.assertAlmostEqual(stats["fidelity_plus"], p_full - p_drop)
        flipped = graph_fidelity(full, torch.tensor([[0.0, 4.0]]), drop)
        self.assertEqual(flipped["acc"], 0.0)

    def test_accumulator_means_two_graphs(self):
        bucket = FidelityAccumulator((0.1, 0.5))
        bucket.add(0.1, {"acc": 1.0, "fidelity_plus": 0.2, "fidelity_minus": 0.0})
        bucket.add(0.1, {"acc": 0.0, "fidelity_plus": 0.4, "fidelity_minus": 0.2})
        bucket.add(0.5, {"acc": 1.0, "fidelity_plus": 0.1, "fidelity_minus": -0.1})
        lines = bucket.lines()
        self.assertIn("FIDELITY p=0.10 n=2 acc=0.500000 fid_plus=0.300000 fid_minus=0.100000", lines)
        self.assertIn("FIDELITY p=0.50 n=1 acc=1.000000 fid_plus=0.100000 fid_minus=-0.100000", lines)

    def test_empty_ratio_is_nan_not_a_fake_zero(self):
        line = FidelityAccumulator((0.3,)).lines()[0]
        self.assertIn("n=0", line)
        self.assertIn("acc=nan", line)

    def test_hard_mask_changes_tiny_gcn_and_scores_are_finite(self):
        torch.manual_seed(0)
        model = _TinyGCN().eval()
        x = torch.randn(5, 3)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 1], [1, 2, 3, 4, 0, 0]])
        batch = torch.zeros(5, dtype=torch.long)
        self.assertTrue(edge_mask_affects_logits(model, x, edge_index, batch))
        full = model(x, edge_index, batch)
        mask = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        stats = score_hard_mask(model, x, edge_index, batch, full, mask)
        self.assertIsNotNone(stats)
        self.assertIn(stats["acc"], (0.0, 1.0))
        self.assertTrue(math.isfinite(stats["fidelity_plus"]))
        self.assertTrue(math.isfinite(stats["fidelity_minus"]))
        self.assertIsNone(
            score_hard_mask(
                model,
                x,
                edge_index.new_zeros((2, 0)),
                batch,
                full,
                mask.new_zeros(0),
            )
        )


def _graph_record(index, labels, scores):
    record = classify_graph_auc(labels, scores)
    record["index"] = index
    return record


def _payload(method, seed, graphs, max_graphs=None):
    return {
        "method": method,
        "dataset": "Benzene",
        "seed": seed,
        "protocol": {"max_graphs": max_graphs},
        "graphs": graphs,
    }


class MatchedAucTests(unittest.TestCase):
    def test_constant_scores_are_flagged_and_not_given_auc(self):
        record = classify_graph_auc([0, 1, 0, 1], [0.2, 0.2, 0.2, 0.2])
        self.assertFalse(record["constant_label"])
        self.assertTrue(record["constant_score"])
        self.assertTrue(record["score_finite"])
        self.assertIsNone(record["auc"])

    def test_constant_labels_are_flagged(self):
        record = classify_graph_auc([1, 1, 1], [0.1, 0.2, 0.9])
        self.assertTrue(record["constant_label"])
        self.assertIsNone(record["auc"])

    def test_nonfinite_scores_are_not_constant_and_have_no_auc(self):
        record = classify_graph_auc([0, 1], [0.1, float("nan")])
        self.assertFalse(record["score_finite"])
        self.assertFalse(record["constant_score"])
        self.assertIsNone(record["auc"])

    def test_mixed_graph_has_finite_auc(self):
        record = classify_graph_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertGreater(record["auc"], 0.5)
        self.assertEqual(record["n_edges"], 4)

    def test_shared_skip_keeps_the_same_graphs(self):
        # 0 constant label, 1 both vary, 2 IDEA constant score, 3 DXMulE
        # constant score, 4 both constant scores, 5 both vary.
        idea_graphs = [
            _graph_record(0, [0, 0], [0.1, 0.9]),
            _graph_record(1, [0, 1], [0.1, 0.9]),
            _graph_record(2, [0, 1], [0.4, 0.4]),
            _graph_record(3, [0, 1], [0.2, 0.8]),
            _graph_record(4, [0, 1], [0.3, 0.3]),
            _graph_record(5, [1, 0], [0.2, 0.7]),
        ]
        dx_graphs = [
            _graph_record(0, [0, 0], [0.2, 0.2]),
            _graph_record(1, [0, 1], [0.2, 0.8]),
            _graph_record(2, [0, 1], [0.1, 0.9]),
            _graph_record(3, [0, 1], [0.5, 0.5]),
            _graph_record(4, [0, 1], [0.6, 0.6]),
            _graph_record(5, [1, 0], [0.1, 0.4]),
        ]
        result = build_matched(_payload("idea", 2025, idea_graphs), _payload("dxmule", 2025, dx_graphs))
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["idea"]["n"], result["dxmule"]["n"])
        self.assertEqual(result["skips"]["constant_label"], 1)
        self.assertEqual(result["skips"]["idea_only_constant_score"], 1)
        self.assertEqual(result["skips"]["dxmule_only_constant_score"], 1)
        self.assertEqual(result["skips"]["both_constant_score"], 1)
        self.assertEqual(result["inclusion"]["matched_n"], 2)
        self.assertEqual(result["inclusion"]["idea_own_rule_n"], 5)
        self.assertEqual(result["inclusion"]["dxmule_own_rule_n"], 3)
        self.assertAlmostEqual(result["idea"]["mean"], float(np_mean([idea_graphs[1]["auc"], idea_graphs[5]["auc"]])))
        self.assertAlmostEqual(result["dxmule"]["mean"], float(np_mean([dx_graphs[1]["auc"], dx_graphs[5]["auc"]])))

    def test_label_disagreement_raises(self):
        idea = [_graph_record(0, [0, 1], [0.1, 0.9])]
        dx = [_graph_record(0, [0, 0], [0.1, 0.9])]
        with self.assertRaises(ValueError):
            build_matched(_payload("idea", 1, idea), _payload("dxmule", 1, dx))

    def test_truncated_split_is_refused_when_required(self):
        graphs = [_graph_record(0, [0, 1], [0.1, 0.9])]
        idea = _payload("idea", 1, graphs, max_graphs=8)
        dx = _payload("dxmule", 1, graphs, max_graphs=8)
        with self.assertRaises(ValueError):
            build_matched(idea, dx, require_full_split=True)

    def test_summary_uses_per_seed_means_and_omits_std_for_one_seed(self):
        one = build_matched(
            _payload("idea", 2025, [
                _graph_record(0, [0, 1], [0.1, 0.9]),
                _graph_record(1, [1, 0], [0.2, 0.8]),
            ]),
            _payload("dxmule", 2025, [
                _graph_record(0, [0, 1], [0.2, 0.7]),
                _graph_record(1, [1, 0], [0.3, 0.9]),
            ]),
        )
        alone = summarize_matched_seeds([one])
        self.assertEqual(alone["n_seeds"], 1)
        self.assertIsNone(alone["idea"]["std_ddof1"])
        other = dict(one)
        other["seed"] = 2026
        other["idea"] = dict(one["idea"])
        other["dxmule"] = dict(one["dxmule"])
        other["idea"]["mean"] = one["idea"]["mean"] + 0.1
        other["dxmule"]["mean"] = one["dxmule"]["mean"] + 0.2
        both = summarize_matched_seeds([one, other])
        self.assertEqual(both["n_seeds"], 2)
        self.assertAlmostEqual(
            both["idea"]["mean"],
            0.5 * (one["idea"]["mean"] + other["idea"]["mean"]),
        )
        self.assertIsNotNone(both["idea"]["std_ddof1"])
        self.assertEqual(both["per_seed_n"], [2, 2])

    def test_record_roundtrip(self):
        graphs = [_graph_record(0, [0, 1, 0], [0.2, 0.4, 0.9])]
        payload = _payload("idea", 2025, graphs)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "idea.json")
            write_auc_record(path, payload)
            loaded = load_json(path)
        self.assertEqual(loaded["seed"], 2025)
        self.assertEqual(loaded["graphs"][0]["index"], 0)


def np_mean(values):
    return sum(values) / len(values)


class EmbeddingTargetTests(unittest.TestCase):
    def test_target_is_the_mean_of_the_current_batch(self):
        class _FeatureEmbed(torch.nn.Module):
            def embedding(self, x, edge_index, edge_weights=None):
                return x

        model = _FeatureEmbed()
        x = torch.tensor([[1.0], [1.0], [3.0], [3.0]])
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        batch = torch.tensor([0, 0, 1, 1])
        target = unmasked_embedding_target(model, x, edge_index, batch)
        self.assertTrue(torch.allclose(target.view(-1), torch.tensor([1.0, 3.0])))

    def test_cached_index_does_not_follow_a_different_batch(self):
        class _FeatureEmbed(torch.nn.Module):
            def embedding(self, x, edge_index, edge_weights=None):
                return x

        model = _FeatureEmbed()

        def _batch(value):
            x = torch.full((4, 1), float(value))
            edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
            batch = torch.tensor([0, 0, 1, 1])
            return x, edge_index, batch

        cached = unmasked_embedding_target(model, *_batch(1.0))
        current = unmasked_embedding_target(model, *_batch(9.0))
        self.assertTrue(torch.allclose(cached.view(-1), torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.allclose(current.view(-1), torch.tensor([9.0, 9.0])))
        self.assertFalse(torch.allclose(cached, current))

    def test_shuffled_loader_changes_batch_membership(self):
        graphs = []
        for i in range(32):
            graphs.append(
                Data(
                    x=torch.full((2, 1), float(i)),
                    edge_index=torch.tensor([[0, 1], [1, 0]]),
                    y=torch.tensor([i]),
                )
            )
        torch.manual_seed(2025)
        loader = DataLoader(graphs, batch_size=8, shuffle=True)

        def signature():
            return [tuple(int(v) for v in batch.y.view(-1).tolist()) for batch in loader]

        self.assertNotEqual(signature(), signature())

    def test_entry_points_recompute_the_current_batch(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(repo, "explainer", "eval_idea.py"), encoding="utf-8") as handle:
            idea = handle.read()
        with open(os.path.join(repo, "explainer", "eval_dxmule.py"), encoding="utf-8") as handle:
            dx = handle.read()
        self.assertIn("unmasked_embedding_target", idea)
        self.assertIn("unmasked_embedding_target", dx)
        self.assertNotIn("emd_targets", idea)
        self.assertNotIn("global_mean_pool(full, data.batch)", dx)


if __name__ == "__main__":
    unittest.main()
