"""Train and score DXMulE.

The original IDEA entry point is ``explainer/eval_idea.py`` and is unchanged.
This script is the multi-scale path: shared IDEA codebooks, retention ratios
{k_m}, and the cross-scale prototype link.

Example (CPU smoke, not a benchmark):

    python explainer/eval_dxmule.py --dataset Benzene --root_path ./ \\
        --device -1 --epochs 1 --batch_size 4 --max_graphs 8

``--max_graphs`` truncates the split so a machine can check that the path
runs. ROC-AUC on that subset is a sanity log. ACC@p and Fidelity+/-@p are
logged from ``dxmule.fidelity`` on the hard retention masks. Sufficiency
and necessity in any other formula, and complement-deletion curves beyond
Fidelity+@p, are not a separate protocol.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import random
import sys
import time
import zipfile

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.explain import Explainer
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dxmule.explainer import DXMulE
from dxmule.fidelity import (
    DEFINITION,
    FidelityAccumulator,
    edge_mask_affects_logits,
    score_hard_mask,
)
from dxmule.scales import parse_scales


DATA_FILES = {
    "Mutag": "mutag_with_gt.pkl",
    "Benzene": "benzene_with_gt.pkl",
    "Alkane": "alkane_with_gt.pkl",
    "Fluoride": "fluoride_with_gt.pkl",
    "BA2Motif": "ba2motifs_with_gt.pkl",
}


def normalize_root(root_path: str) -> str:
    return os.path.abspath(root_path or ".")


def ensure_pickled_dataset(root: str, filename: str) -> str:
    """Return the pickle path, unzipping ``filename.zip`` when that is all we have."""
    data_dir = os.path.join(root, "data")
    pickle_path = os.path.join(data_dir, filename)
    zip_path = pickle_path + ".zip"
    if os.path.isfile(pickle_path):
        return pickle_path
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(
            f"Missing {pickle_path}. Expected the pickle or {zip_path}."
        )
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if os.path.basename(name) == filename and not name.startswith("__MACOSX")
        ]
        if not members:
            raise FileNotFoundError(f"{zip_path} does not contain {filename}")
        os.makedirs(data_dir, exist_ok=True)
        with archive.open(members[0]) as source, open(pickle_path, "wb") as target:
            target.write(source.read())
    logging.info("Unzipped %s to %s", zip_path, pickle_path)
    return pickle_path


def get_dataset(dataset_name: str, root: str):
    if dataset_name not in DATA_FILES:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented")
    path = ensure_pickled_dataset(root, DATA_FILES[dataset_name])
    with open(path, "rb") as handle:
        return pickle.load(handle)


def split_dataset(dataset_name: str, dataset):
    """Same cuts as ``explainer/eval_idea.py``."""
    if dataset_name == "Mutag":
        return dataset[800:], dataset[:800]
    if dataset_name == "Benzene":
        test_num = int(0.3 * len(dataset))
        return dataset[test_num:], dataset[:test_num]
    if dataset_name in {"Alkane", "Fluoride"}:
        return dataset[500:], dataset[:500]
    if dataset_name == "BA2Motif":
        return dataset[:800], dataset[800:]
    raise NotImplementedError(f"Dataset {dataset_name} is not implemented")


def init_model(dataset_name: str, root: str, device: str):
    path = os.path.join(root, "GNN", f"{dataset_name}.pth")
    model = torch.load(path, weights_only=False, map_location=torch.device(device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def resolve_device(device_id: int) -> str:
    if device_id >= 0 and torch.cuda.is_available():
        return f"cuda:{device_id}"
    return "cpu"


def load_pretrained_codebooks(algorithm: DXMulE, root: str, dataset_name: str, device: str) -> bool:
    """Copy upstream codebook A/B into the shared base and explanatory codebooks.

    The cross-scale affinity is not part of those checkpoints. It stays at
    its identity-peaked initialization when a checkpoint is loaded.
    """
    codebook_dir = os.path.join(root, "codebook")
    path_a = os.path.join(codebook_dir, f"Codebook_A_{dataset_name}.pt")
    path_b = os.path.join(codebook_dir, f"Codebook_B_{dataset_name}.pt")
    if not (os.path.isfile(path_a) and os.path.isfile(path_b)):
        return False
    weight_a = torch.load(path_a, map_location=device, weights_only=True).float()
    weight_b = torch.load(path_b, map_location=device, weights_only=True).float()
    base = algorithm.rv_emd_model.base_codebook.embedding.weight
    exp = algorithm.rv_emd_model.exp_codebook.embedding.weight
    if tuple(weight_a.shape) != tuple(base.shape) or tuple(weight_b.shape) != tuple(exp.shape):
        raise RuntimeError(
            "Pretrained codebook shape does not match this run "
            f"(A {tuple(weight_a.shape)} vs {tuple(base.shape)}, "
            f"B {tuple(weight_b.shape)} vs {tuple(exp.shape)}). "
            "Retrain with --use_ckpt 0, or match --codebook_size to the file."
        )
    base.data.copy_(weight_a)
    exp.data.copy_(weight_b)
    logging.info("Loaded pretrained codebooks from %s", codebook_dir)
    return True


def train_single_scale_tokenizer(
    explained_model,
    train_loader,
    train_size: int,
    codebook_size: int,
    embedding_dim: int,
    num_features: int,
    epochs: int,
    learning_rate: float,
    device: str,
    dataset_name: str,
    root: str,
):
    """Upstream HGTokenizer pass (single graph, not multi-scale).

    TODO: a multi-scale tokenizer that quantizes each retention subgraph
    during codebook learning is not implemented. DXMulE consumes the
    codebooks this function (or the released checkpoints) produce, and
    applies {k_m} only when the explainer is trained.
    """
    import torch.nn.functional as F
    from tqdm import tqdm

    from graph_tokenizer import GraphTokenizer

    tokenizer = GraphTokenizer(
        codebook_size=codebook_size,
        codebook_dims=embedding_dim,
        decoder_dims=num_features,
    ).to(device)
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    codebook_a = None
    codebook_b = None
    best_correct = -1

    for epoch in range(epochs):
        tokenizer.train()
        total_loss = 0.0
        correct = 0
        for data in tqdm(train_loader, desc=f"tokenizer {epoch + 1}/{epochs}"):
            data = data.to(device)
            optimizer.zero_grad()
            embed = explained_model.embedding(data.x, data.edge_index)
            x_res_a, x_res_b, token_loss, *_rest = tokenizer(
                embed, data.x, data.edge_index, use_sk=True
            )
            prediction_a = explained_model.prediction(x_res_a, batch=data.batch)
            prediction_b = explained_model.prediction(x_res_b, batch=data.batch)
            uniform = torch.ones_like(prediction_a) / prediction_a.size(1)
            loss_a = F.kl_div(
                F.log_softmax(prediction_a, dim=1), uniform, reduction="batchmean"
            )
            loss_b = criterion(prediction_b, data.y)
            loss = token_loss + loss_a + loss_b
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            correct += int((prediction_b.argmax(dim=1) == data.y).sum().item())
            if correct > best_correct:
                best_correct = correct
                codebook_a = tokenizer.codebook_a.embedding.weight.detach().cpu().clone()
                codebook_b = tokenizer.codebook_b.embedding.weight.detach().cpu().clone()
        logging.info(
            "Tokenizer epoch %s/%s loss %.4f acc %.4f",
            epoch + 1,
            epochs,
            total_loss / max(len(train_loader), 1),
            correct / max(train_size, 1),
        )

    if codebook_a is None or codebook_b is None:
        raise RuntimeError("Tokenizer produced no codebook weights")
    codebook_dir = os.path.join(root, "codebook")
    os.makedirs(codebook_dir, exist_ok=True)
    torch.save(codebook_a, os.path.join(codebook_dir, f"Codebook_A_{dataset_name}.pt"))
    torch.save(codebook_b, os.path.join(codebook_dir, f"Codebook_B_{dataset_name}.pt"))
    return codebook_a, codebook_b


def _batch_node_counts(data):
    if getattr(data, "batch", None) is None:
        return [int(data.num_nodes)]
    return torch.bincount(data.batch).detach().cpu().tolist()


def _grad_flow(module):
    """Whether the last backward left a finite, non-zero gradient on any parameter."""
    nonzero = 0
    abs_sum = 0.0
    saw_grad = False
    finite = True
    for param in module.parameters():
        if param.grad is None:
            continue
        saw_grad = True
        finite = finite and bool(torch.isfinite(param.grad).all().item())
        mag = float(param.grad.detach().abs().sum().item())
        abs_sum += mag
        if mag > 0.0:
            nonzero += 1
    return bool(saw_grad and finite), nonzero, abs_sum


def train_dxmule(explainer, train_loader, epochs: int, device: str, smoke: bool = False):
    model = explainer.model
    model.eval()
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        totals = {
            "loss": 0.0,
            "rqvae_loss": 0.0,
            "distribution_loss": 0.0,
            "reconstruction_loss": 0.0,
            "cross_scale_loss": 0.0,
            "complement_loss": 0.0,
            "topdown_kl": 0.0,
            "reverse_kl": 0.0,
            "shell_loss": 0.0,
            "persistence": 0.0,
        }
        train_counts = []
        for data in train_loader:
            data = data.to(device)
            if smoke and epoch == 0:
                train_counts.append(_batch_node_counts(data))
            with torch.no_grad():
                # Unmasked embedding of this batch. Recomputed every step so
                # shuffling the loader cannot pair a graph with another graph's target.
                full = model.embedding(data.x, data.edge_index)
                emd_target = global_mean_pool(full, data.batch)
            target = model(data.x, data.edge_index, data.batch)
            stats = explainer.algorithm.train(
                epoch,
                model,
                data.x,
                data.edge_index,
                target=target,
                emd_target=emd_target,
                batch=data.batch,
            )
            for key in totals:
                totals[key] += float(stats[key])
        if smoke and epoch == 0:
            logging.info("SMOKE epoch-1 train batch node counts: %s", train_counts)
        steps = max(len(train_loader), 1)
        averaged = {key: value / steps for key, value in totals.items()}
        history.append(averaged)
        logging.info(
            "Epoch %s/%s loss %.4f rv %.4f dist %.4f recon %.4f "
            "cross %.4f (topdown %.4f reverse %.4f shell %.4f persist %.3f) "
            "complement %.4f",
            epoch + 1,
            epochs,
            averaged["loss"],
            averaged["rqvae_loss"],
            averaged["distribution_loss"],
            averaged["reconstruction_loss"],
            averaged["cross_scale_loss"],
            averaged["topdown_kl"],
            averaged["reverse_kl"],
            averaged["shell_loss"],
            averaged["persistence"],
            averaged["complement_loss"],
        )
        if not np.isfinite(averaged["loss"]):
            raise RuntimeError("DXMulE loss is not finite. Stopped before evaluation.")
    elapsed = time.perf_counter() - started
    logging.info("Train loop seconds: %.3f", elapsed)
    if smoke:
        grad_finite, grad_nonzero, grad_abs = _grad_flow(explainer.algorithm)
        logging.info(
            "SMOKE last-step grad flow: finite=%s nonzero_params=%s grad_abs_sum=%.6f",
            grad_finite,
            grad_nonzero,
            grad_abs,
        )
        logging.info("SMOKE train loop seconds: %.3f", elapsed)
    return history


def _auc_status(labels, scores) -> tuple[str, float | None]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if labels.min() == labels.max():
        return "constant_labels", None
    if np.allclose(scores, scores[0]):
        return "constant_scores", None
    return "ok", float(roc_auc_score(labels, scores))


def evaluate_dxmule(explainer, test_loader, device: str):
    """Shared-score ROC-AUC, hard-mask precision and recall, and fidelity.

    Soft retention masks are strictly increasing functions of the same logits,
    so they induce the same edge ranking and the same ROC-AUC. Per-scale
    precision and recall compare the hard top-k subgraph to ground-truth edges.
    ACC@p and Fidelity+/-@p use those same hard masks against the frozen GNN.
    A constant ground-truth edge label, or a constant explanation score, is
    left out of ROC-AUC and counted separately. Recall is averaged only over
    graphs with at least one positive ground-truth edge.
    """
    ratios = explainer.algorithm.retention.ks
    shared_aucs = []
    n_const_label = 0
    n_const_score = 0
    per_scale = {
        k: {"precision": [], "recall": [], "retention": []} for k in ratios
    }
    fidelity = FidelityAccumulator(ratios)
    mask_ok = None
    model = explainer.model
    model.eval()
    n_nested = 0

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            data = data.to(device)
            target = model(data.x, data.edge_index, data.batch)
            explanation = explainer(
                data.x, data.edge_index, target=target, batch=data.batch
            )
            labels = data.edge_mask.detach().float().cpu()
            status, shared = _auc_status(
                labels.tolist(), explanation.edge_mask.detach().cpu().tolist()
            )
            if status == "constant_labels":
                n_const_label += 1
            elif status == "constant_scores":
                n_const_score += 1
            elif shared is not None:
                shared_aucs.append(shared)

            hard = explanation.scale_hard_masks.detach().cpu()
            if hard.size(0) != len(ratios):
                raise RuntimeError("scale_hard_masks does not match the retention ratios")
            for row in range(hard.size(0) - 1):
                if not bool(torch.all(hard[row] <= hard[row + 1])):
                    raise RuntimeError(
                        "Hard retention masks are not nested. "
                        "Scales must be one shared top-k ranking."
                    )
            n_nested += 1
            positive = labels > 0.5
            n_positive = int(positive.sum())
            for row, ratio in enumerate(ratios):
                kept = hard[row] > 0.5
                n_kept = int(kept.sum())
                hits = int((positive & kept).sum()) if n_kept else 0
                per_scale[ratio]["retention"].append(
                    float(n_kept) / float(kept.numel()) if kept.numel() else float("nan")
                )
                per_scale[ratio]["precision"].append(
                    hits / n_kept if n_kept else float("nan")
                )
                per_scale[ratio]["recall"].append(
                    hits / n_positive if n_positive else float("nan")
                )

            if mask_ok is False:
                continue
            if mask_ok is None:
                mask_ok = bool(edge_mask_affects_logits(
                    model, data.x, data.edge_index, data.batch
                ))
                logging.info("FIDELITY mask_sensitivity=%s graph_index=%s", mask_ok, idx)
                if not mask_ok:
                    continue
            if data.edge_index.size(1) == 0:
                fidelity.skipped_no_edges += 1
                continue
            for row, ratio in enumerate(fidelity.ratios):
                stats = score_hard_mask(
                    model, data.x, data.edge_index, data.batch, target, hard[row]
                )
                if stats is None:
                    fidelity.skipped_no_edges += 1
                    break
                fidelity.add(ratio, stats)

    def _mean(values: list[float]) -> float:
        finite = [v for v in values if np.isfinite(v)]
        if not finite:
            return float("nan")
        return float(np.mean(finite))

    def _finite_count(values: list[float]) -> int:
        return sum(1 for value in values if np.isfinite(value))

    logging.info(
        "Hard retention masks nested: True (n=%s graphs). "
        "A violation raises before this line.",
        n_nested,
    )
    if shared_aucs:
        arr = np.asarray(shared_aucs, dtype=float)
        logging.info(
            "ROC-AUC summary mean=%.6f std_ddof0=%.6f min=%.6f max=%.6f n=%s",
            float(arr.mean()),
            float(arr.std(ddof=0)),
            float(arr.min()),
            float(arr.max()),
            len(shared_aucs),
        )
    else:
        logging.info("ROC-AUC summary mean=nan std_ddof0=nan min=nan max=nan n=0")
    logging.info(
        "AUC skipped_constant_labels=%s skipped_constant_scores=%s scored=%s",
        n_const_label,
        n_const_score,
        len(shared_aucs),
    )
    logging.info(
        "Shared-score ROC-AUC %.4f (n=%s) values=%s. Soft masks reuse this ranking, "
        "so they are not scored separately.",
        _mean(shared_aucs),
        len(shared_aucs),
        [round(float(a), 4) for a in shared_aucs],
    )
    for ratio in ratios:
        bucket = per_scale[ratio]
        logging.info(
            "Hard retention summary k=%.2f realized=%.6f precision=%.6f recall=%.6f "
            "n_retention=%s n_precision=%s n_recall=%s",
            ratio,
            _mean(bucket["retention"]),
            _mean(bucket["precision"]),
            _mean(bucket["recall"]),
            len(bucket["retention"]),
            _finite_count(bucket["precision"]),
            _finite_count(bucket["recall"]),
        )
        logging.info(
            "Hard retention k=%.2f realized %.3f precision %.3f recall %.3f (n=%s)",
            ratio,
            _mean(bucket["retention"]),
            _mean(bucket["precision"]),
            _mean(bucket["recall"]),
            len(bucket["retention"]),
        )
    if mask_ok is not True:
        logging.info("FIDELITY status=UNKNOWN")
        logging.info(
            "FIDELITY gap: an all-zero edge mask did not change GNN logits "
            "relative to an all-one mask on the checked test graph, so ACC@p "
            "and Fidelity@p were not computed."
        )
    else:
        logging.info("FIDELITY status=OK")
        logging.info("FIDELITY definition: %s", DEFINITION)
        logging.info(
            "FIDELITY skipped_no_edges=%s skipped_nonfinite=%s",
            fidelity.skipped_no_edges,
            fidelity.skipped_nonfinite,
        )
        for line in fidelity.lines():
            logging.info("%s", line)
    return {"shared_auc": shared_aucs, "per_scale": per_scale}


def main():
    parser = argparse.ArgumentParser(description="DXMulE multi-scale prototype explainer")
    parser.add_argument("--root_path", type=str, default="./", help="Repo root (this fork has no nested IDEA/)")
    parser.add_argument("--dataset", type=str, default="Benzene")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=int, default=-1, help="GPU id, or -1 for CPU")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--codebook_size", type=int, default=32)
    parser.add_argument("--use_ckpt", type=int, default=1)
    parser.add_argument("--gt_learning_rate", type=float, default=1e-3)
    parser.add_argument("--gt_epochs", type=int, default=10)
    parser.add_argument(
        "--scales",
        type=str,
        default="0.1,0.3,0.5,0.7,0.9",
        help="Comma-separated edge-retention ratios k_m in (0, 1]",
    )
    parser.add_argument("--cross_scale_weight", type=float, default=0.1)
    parser.add_argument("--complement_weight", type=float, default=0.1)
    parser.add_argument("--shell_weight", type=float, default=0.1)
    parser.add_argument("--retention_temperature", type=float, default=1.0)
    parser.add_argument(
        "--max_graphs",
        type=int,
        default=None,
        help="If set, truncate train/test for a smoke run. Metrics are not a benchmark.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = normalize_root(args.root_path)
    device = resolve_device(args.device)
    scales = parse_scales(args.scales)
    logging.info("Device %s root %s scales %s", device, root, scales)
    if args.device >= 0 and device == "cpu":
        logging.info(
            "Requested GPU device %s but CUDA is unavailable. Using CPU.",
            args.device,
        )
    logging.info(
        "PROTOCOL dataset=%s epochs=%s lr=%s batch_size=%s seed=%s "
        "codebook_size=%s use_ckpt=%s max_graphs=%s scales=%s "
        "cross_scale_weight=%s complement_weight=%s shell_weight=%s "
        "retention_temperature=%s device=%s",
        args.dataset,
        args.epochs,
        args.lr,
        args.batch_size,
        args.seed,
        args.codebook_size,
        args.use_ckpt,
        args.max_graphs,
        scales,
        args.cross_scale_weight,
        args.complement_weight,
        args.shell_weight,
        args.retention_temperature,
        device,
    )
    if args.max_graphs is not None:
        logging.info(
            "max_graphs=%s truncates the split. Losses and AUC below are a smoke check.",
            args.max_graphs,
        )

    dataset = get_dataset(args.dataset, root)
    train_set, test_set = split_dataset(args.dataset, dataset)
    if args.max_graphs is not None:
        if args.max_graphs < 2:
            raise ValueError("max_graphs must be at least 2")
        train_set = list(train_set)[: args.max_graphs]
        test_keep = max(1, args.max_graphs // 4)
        test_set = list(test_set)[:test_keep]

    num_features = train_set[0].x.size(1)
    labels = torch.tensor([int(data.y) for data in list(train_set) + list(test_set)])
    num_classes = int(torch.unique(labels).numel())
    logging.info(
        "%s graphs in this run: train %s, test %s, features %s, classes %s",
        args.dataset,
        len(train_set),
        len(test_set),
        num_features,
        num_classes,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    explained_model = init_model(args.dataset, root, device)

    sample = train_set[0]
    with torch.no_grad():
        sample_emb = explained_model.embedding(
            sample.x.to(device), sample.edge_index.to(device)
        )
    embedding_dim = int(sample_emb.size(-1))

    algorithm = DXMulE(
        epochs=args.epochs,
        lr=args.lr,
        codebook_size=args.codebook_size,
        codebook_channels=embedding_dim,
        scales=scales,
        cross_scale_weight=args.cross_scale_weight,
        complement_weight=args.complement_weight,
        shell_weight=args.shell_weight,
        retention_temperature=args.retention_temperature,
    ).to(device)

    loaded = False
    if args.use_ckpt:
        loaded = load_pretrained_codebooks(algorithm, root, args.dataset, device)
    if not loaded:
        logging.info(
            "No usable pretrained codebook. Training the single-scale HGTokenizer "
            "(upstream procedure) before DXMulE."
        )
        codebook_a, codebook_b = train_single_scale_tokenizer(
            explained_model,
            train_loader,
            len(train_set),
            args.codebook_size,
            embedding_dim,
            num_features,
            args.gt_epochs,
            args.gt_learning_rate,
            device,
            args.dataset,
            root,
        )
        algorithm.rv_emd_model.base_codebook.embedding.weight.data.copy_(
            codebook_a.to(device)
        )
        algorithm.rv_emd_model.exp_codebook.embedding.weight.data.copy_(
            codebook_b.to(device)
        )

    model_config = {
        "mode": "multiclass_classification" if num_classes > 2 else "binary_classification",
        "task_level": "graph",
        "return_type": "raw",
    }
    explainer = Explainer(
        model=explained_model,
        algorithm=algorithm,
        explanation_type="phenomenon",
        edge_mask_type="object",
        model_config=model_config,
    )

    train_dxmule(
        explainer,
        train_loader,
        args.epochs,
        device,
        smoke=args.max_graphs is not None,
    )
    eval_started = time.perf_counter()
    evaluate_dxmule(explainer, test_loader, device)
    logging.info("Evaluate seconds: %.3f", time.perf_counter() - eval_started)


if __name__ == "__main__":
    main()
