from tqdm import tqdm
import argparse
import random
import torch
import numpy as np
from time import time
import logging
import os
import sys
import torch.nn.functional as F
import pickle

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm.utils import clear_masks
from gnn_model import *
from idea import Idea
from sklearn.metrics import roc_auc_score

from dxmule.embedding_target import unmasked_embedding_target
from dxmule.fidelity import (
    DEFINITION,
    FidelityAccumulator,
    edge_mask_affects_logits,
    score_hard_mask,
)
from dxmule.matched_auc import classify_graph_auc, write_auc_record
from dxmule.scales import edge_graph_ids, hard_retention_mask, parse_scales

def topk_mask_py(lst, k):
    if k <= 0:
        return [0] * len(lst)
    indexed = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
    mask = [0] * len(lst)
    for idx, _ in indexed[:k]:
        mask[idx] = 1
    return mask

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


def get_dataset(dataset_name="BA2Motif", root_path=".", **kwargs):
    if dataset_name == "Mutag":
        with open(root_path + 'data/mutag_with_gt.pkl', 'rb') as f:
            dataset = pickle.load(f)
    elif dataset_name == "Benzene":
        with open(root_path + 'data/benzene_with_gt.pkl', 'rb') as f:
            dataset = pickle.load(f)
    elif dataset_name == "Alkane":
        with open(root_path + 'data/alkane_with_gt.pkl', 'rb') as f:
            dataset = pickle.load(f)
    elif dataset_name == "Fluoride":
        with open(root_path + 'data/fluoride_with_gt.pkl', 'rb') as f:
            dataset = pickle.load(f)
    elif dataset_name == "BA2Motif":
        with open(root_path + 'data/ba2motifs_with_gt.pkl', 'rb') as f:
            dataset = pickle.load(f)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
    return dataset

def init_model(dataset_name="BA2Motif", root_path = ".", device = None):
    model_path = root_path + f'GNN/{dataset_name}.pth'
    model = torch.load(model_path, weights_only=False, map_location=torch.device(device))
    model.eval()
    return model

def _batch_node_counts(data):
    if getattr(data, "batch", None) is None:
        return [int(data.num_nodes)]
    return torch.bincount(data.batch).detach().cpu().tolist()


def train_explainer(explainer, train_loader, epochs = 20,
                    exp_type = "phenomenon", device = None, smoke = False):
    losses = []
    start_time = time()
    logged_pairing = False

    # The unmasked embedding target is computed inside the epoch loop on the
    # batch just drawn. A list cached before the loop and indexed by position
    # is wrong: shuffle=True changes batch membership every iteration.
    for epoch in range(epochs):
        epoch_loss = {
            'rqvae_loss': 0.0,
            'distribution_loss': 0.0,
            'codebook_loss': 0.0,
        }
        loss_key = list(epoch_loss.keys())
        train_counts = []
        for data in train_loader:
            data = data.to(device)
            if smoke and epoch == 0:
                train_counts.append(_batch_node_counts(data))
            emd_target = unmasked_embedding_target(
                explainer.model, data.x, data.edge_index, data.batch
            )
            n_graphs = int(data.num_graphs)
            if emd_target.size(0) != n_graphs:
                raise RuntimeError(
                    f"embedding target has {emd_target.size(0)} rows for {n_graphs} graphs"
                )
            if not logged_pairing:
                logging.info(
                    "IDEA unmasked embedding target is recomputed on the current "
                    "batch each optimizer step (rows=%s). A cached list indexed by "
                    "DataLoader position is not used.",
                    n_graphs,
                )
                logged_pairing = True
            target = explainer.model(data.x, data.edge_index, data.batch)
            ls = explainer.algorithm.train(epoch, 
                            explainer.model, 
                            data.x, 
                            data.edge_index, 
                            target = target,
                            emd_target = emd_target, 
                            batch = data.batch)
            assert len(ls) == len(loss_key), "Loss keys and values mismatch"
            for i in range(len(loss_key)):
                epoch_loss[loss_key[i]] += ls[i]
        if smoke and epoch == 0:
            logging.info("SMOKE epoch-1 train batch node counts: %s", train_counts)
        avg_loss = {k: v / len(train_loader) for k, v in epoch_loss.items()}
        logging.info(f"Epoch {epoch+1}/{epochs}"
                        f"- RQVAE Loss: {avg_loss['rqvae_loss']:.4f}\n" +
                        f"- Codebook Loss: {avg_loss['codebook_loss']:.4f}\n" +
                        f"- Distribution Loss: {avg_loss['distribution_loss']:.4f}\n" +
                        f"- Time: {time() - start_time:.2f}s")
        losses.append(avg_loss['rqvae_loss'])

    logging.info(
        "Train loop seconds: %.3f (unmasked embedding recomputed on each current batch)",
        time() - start_time,
    )
    if smoke:
        grad_finite, grad_nonzero, grad_abs = _grad_flow(explainer.algorithm)
        logging.info(
            "SMOKE last-step grad flow: finite=%s nonzero_params=%s grad_abs_sum=%.6f",
            grad_finite,
            grad_nonzero,
            grad_abs,
        )
    return losses

def _log_fidelity(fidelity, mask_ok):
    if fidelity is None or mask_ok is not True:
        logging.info("FIDELITY status=UNKNOWN")
        if fidelity is None:
            logging.info("FIDELITY gap: no fidelity ratios were set, so ACC@p was not computed.")
        else:
            logging.info(
                "FIDELITY gap: an all-zero edge mask did not change GNN logits "
                "relative to an all-one mask on the checked test graph, so "
                "ACC@p and Fidelity@p were not computed."
            )
        return
    logging.info("FIDELITY status=OK")
    logging.info("FIDELITY definition: %s", DEFINITION)
    logging.info(
        "FIDELITY skipped_no_edges=%s skipped_nonfinite=%s",
        fidelity.skipped_no_edges,
        fidelity.skipped_nonfinite,
    )
    for line in fidelity.lines():
        logging.info("%s", line)


def _log_own_rule_auc(aucs, constant_label, constant_score_mixed, nonfinite):
    """Phase-1 IDEA inclusion: drop constant ground-truth labels only.

    This count is not the fair comparison. Constant explanation scores stay
    inside it. Matched-subset AUC is computed later from the per-graph record.
    """
    if aucs:
        arr = np.asarray(aucs, dtype=float)
        logging.info(
            "OWN-RULE ROC-AUC summary mean=%.6f std_ddof0=%.6f min=%.6f max=%.6f n=%s",
            float(arr.mean()),
            float(arr.std(ddof=0)),
            float(arr.min()),
            float(arr.max()),
            len(aucs),
        )
    else:
        logging.info("OWN-RULE ROC-AUC summary mean=nan std_ddof0=nan min=nan max=nan n=0")
    logging.info(
        "OWN-RULE skips constant labels only. constant_label=%s "
        "constant_score_mixed_labels=%s nonfinite_scores=%s scored=%s",
        constant_label,
        constant_score_mixed,
        nonfinite,
        len(aucs),
    )
    logging.info(
        "OWN-RULE ROC-AUC is not the fair comparison. "
        "Matched-subset AUC is computed by dxmule.matched_auc from --auc_record."
    )


def evaluate_explainer(explainer, test_loader, model, device=None, gt_mask=None, dataset = None,
                       fidelity_ratios=(), auc_record_path=None, auc_meta=None):
    results = {
        'roc_auc': [],
    }
    skipped_constant_labels = 0
    constant_score_mixed = 0
    nonfinite_scores = 0
    graph_records = []
    fidelity = FidelityAccumulator(fidelity_ratios) if fidelity_ratios else None
    mask_ok = None
    model.eval()

    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_loader, desc = "Processing Test Data", ncols = 80)):
            data = data.to(device)
            if int(data.num_graphs) != 1:
                raise RuntimeError(
                    "AUC records require test batch_size 1, "
                    f"got {int(data.num_graphs)} graphs"
                )
            clear_masks(model)
            target = explainer.model(data.x, data.edge_index, data.batch)
            exp = explainer(data.x, data.edge_index, target = target, batch = data.batch)

            labels = data.edge_mask.detach().float().cpu().numpy()
            scores_np = exp.edge_mask.detach().float().cpu().numpy()
            record = classify_graph_auc(labels, scores_np)
            record["index"] = idx
            graph_records.append(record)
            if record["constant_label"]:
                skipped_constant_labels += 1
            elif not record["score_finite"]:
                nonfinite_scores += 1
            elif record["constant_score"]:
                constant_score_mixed += 1
                results['roc_auc'].append(
                    float(roc_auc_score(labels.astype(int), scores_np))
                )
            else:
                results['roc_auc'].append(float(record["auc"]))

            if fidelity is None or mask_ok is False:
                continue
            scores = exp.edge_mask.detach()
            if scores.numel() == 0 or not torch.isfinite(scores).all():
                fidelity.skipped_nonfinite += 1
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
            edge_batch = edge_graph_ids(data.edge_index, data.batch, data.x.size(0))
            for ratio in fidelity.ratios:
                hard = hard_retention_mask(scores, edge_batch, ratio)
                stats = score_hard_mask(
                    model, data.x, data.edge_index, data.batch, target, hard
                )
                if stats is None:
                    fidelity.skipped_no_edges += 1
                    break
                fidelity.add(ratio, stats)

    _log_own_rule_auc(
        results['roc_auc'],
        skipped_constant_labels,
        constant_score_mixed,
        nonfinite_scores,
    )
    if auc_record_path:
        meta = dict(auc_meta or {})
        payload = {
            "method": "idea",
            "dataset": meta.get("dataset", dataset),
            "seed": meta.get("seed"),
            "protocol": meta.get("protocol", {}),
            "score": "sigmoid of the IDEA edge logit (Explanation.edge_mask)",
            "n_graphs": len(graph_records),
            "graphs": graph_records,
        }
        write_auc_record(auc_record_path, payload)
        logging.info(
            "Wrote per-graph AUC record %s n=%s",
            auc_record_path,
            len(graph_records),
        )
    _log_fidelity(fidelity, mask_ok)

    return results

def main():
    parser = argparse.ArgumentParser(description="PGExplainer Test")
    parser.add_argument("--root_path", type=str, default="", help="Path to IDEA directory")
    parser.add_argument("--dataset", type=str, default="BA2Motif", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type = int, default = 0, help = "GPU ID to use (-1 for CPU)")
    parser.add_argument("--seed", type= int, default=2025, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--codebook_size", type=int, default=32, help="Size of the codebook for enhanced PGExplainer")
    
    parser.add_argument('--use_ckpt', type=int, default=1)
    parser.add_argument("--gt_learning_rate", type=float, default=1e-3, help="GT Learning rate")
    parser.add_argument("--gt_epochs", type=int, default=10, help="Training epochs")
    parser.add_argument(
        "--max_graphs",
        type=int,
        default=None,
        help="If set, truncate train/test the same way as explainer/eval_dxmule.py. "
             "Metrics are a smoke check, not a benchmark.",
    )
    parser.add_argument(
        "--fidelity_scales",
        type=str,
        default="0.1,0.3,0.5,0.7,0.9",
        help="Hard top-p grid for ACC@p and Fidelity+/-@p. "
             "Independent of DXMulE retention. Empty is not allowed.",
    )
    parser.add_argument(
        "--auc_record",
        type=str,
        default="",
        help="Write per-graph AUC flags for matched-subset scoring. Empty skips the file.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.device >= 0 and torch.cuda.is_available():
        device = f"cuda:{args.device}"
    elif args.device >= 0 and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        if args.device >= 0:
            logging.info(
                "Requested GPU device %s but CUDA and MPS are unavailable. Using CPU.",
                args.device,
            )
    logging.info(f"Using device: {device}")
    fidelity_ratios = tuple(sorted(set(parse_scales(args.fidelity_scales))))
    logging.info(
        "PROTOCOL dataset=%s epochs=%s lr=%s batch_size=%s seed=%s "
        "codebook_size=%s use_ckpt=%s max_graphs=%s fidelity_scales=%s "
        "device=%s embedding_target=current_batch",
        args.dataset,
        args.epochs,
        args.lr,
        args.batch_size,
        args.seed,
        args.codebook_size,
        args.use_ckpt,
        args.max_graphs,
        fidelity_ratios,
        device,
    )
    logging.info(
        "NOTE IDEA loss values are not comparable to DXMulE. The objectives differ."
    )

    if args.dataset == "Mutag":
        dataset = get_dataset(args.dataset, args.root_path)
        train_set, test_set = dataset[800:], dataset[:800]
        num_features = train_set[0].x.shape[1]
        num_classes = (torch.unique(torch.tensor([data.y for data in test_set]))).numel()
        logging.info(f"Loaded {args.dataset} dataset with {len(train_set) + len(test_set)} graphs, {num_features} features, {num_classes} classes")
        logging.info(f"Training graphs: {len(train_set)}, Testing graphs: {len(test_set)}")
    elif args.dataset == "Benzene":
        dataset = get_dataset(args.dataset, args.root_path)
        test_num = int(0.3 * len(dataset))
        train_set, test_set = dataset[test_num:], dataset[:test_num]
        num_features = train_set[0].x.shape[1]
        num_classes = (torch.unique(torch.tensor([data.y for data in test_set]))).numel()
        logging.info(f"Loaded {args.dataset} dataset with {len(dataset)} graphs, {num_features} features, {num_classes} classes")
        logging.info(f"Training graphs: {len(train_set)}, Testing graphs: {len(test_set)}")
    elif args.dataset == "Alkane":
        dataset = get_dataset(args.dataset, args.root_path)
        train_set, test_set = dataset[500:], dataset[:500]
        num_features = train_set[0].x.shape[1]
        num_classes = (torch.unique(torch.tensor([data.y for data in test_set]))).numel()
        logging.info(f"Loaded {args.dataset} dataset with {len(dataset)} graphs, {num_features} features, {num_classes} classes")
        logging.info(f"Training graphs: {len(train_set)}, Testing graphs: {len(test_set)}")
    elif args.dataset == "Fluoride":
        dataset = get_dataset(args.dataset, args.root_path)
        train_set, test_set = dataset[500:], dataset[:500]
        num_features = train_set[0].x.shape[1]
        num_classes = (torch.unique(torch.tensor([data.y for data in test_set]))).numel()
        logging.info(f"Loaded {args.dataset} dataset with {len(dataset)} graphs, {num_features} features, {num_classes} classes")
        logging.info(f"Training graphs: {len(train_set)}, Testing graphs: {len(test_set)}")
    elif args.dataset == 'BA2Motif':
        dataset = get_dataset(args.dataset, args.root_path)
        train_set, test_set = dataset[:800], dataset[800:]
        num_features = train_set[0].x.shape[1]
        num_classes = (torch.unique(torch.tensor([data.y for data in test_set]))).numel()
        logging.info(f"Loaded {args.dataset} dataset with {len(dataset)} graphs, {num_features} features, {num_classes} classes")
        logging.info(f"Training graphs: {len(train_set)}, Testing graphs: {len(test_set)}")
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    if args.max_graphs is not None:
        if args.max_graphs < 2:
            raise ValueError("max_graphs must be at least 2")
        # Same truncation as explainer/eval_dxmule.py: the head of each split.
        train_set = list(train_set)[: args.max_graphs]
        test_keep = max(1, args.max_graphs // 4)
        test_set = list(test_set)[:test_keep]
        logging.info(
            "max_graphs=%s truncates the split (smoke, not a benchmark). train %s test %s",
            args.max_graphs,
            len(train_set),
            len(test_set),
        )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    explained_model = init_model(args.dataset, args.root_path, device)
    logging.info("Loaded pretrained GNN from %s", args.root_path + f"GNN/{args.dataset}.pth")

    model_config = {
        'mode': "multiclass_classification" if num_classes > 2 else "binary_classification",
        'task_level': "graph",
        'return_type': "raw"
    }
    __embedding = explained_model.embedding(train_set[0].x.to(device), train_set[0].edge_index.to(device))
    explainer = Explainer(
        model = explained_model,
        algorithm = Idea(
            epochs = args.epochs, lr = args.lr, 
            codebook_size = args.codebook_size, codebook_channels = __embedding.shape[1]
        ).to(device),
        explanation_type = "phenomenon",
        edge_mask_type = "object",
        model_config = model_config
    )

    codebook_root = args.root_path + 'codebook/'
    if args.use_ckpt and os.path.isfile(codebook_root + f'Codebook_A_{args.dataset}.pt'):
        logging.info(f'Loading pretrained codebooks from {codebook_root}')
        pretrained_weight = torch.load(codebook_root + f'Codebook_A_{args.dataset}.pt')
        explainer.algorithm.rv_emd_model.base_codebook.embedding.weight.data = pretrained_weight.to(device)
        pretrained_weight = torch.load(codebook_root + f'Codebook_B_{args.dataset}.pt')
        explainer.algorithm.rv_emd_model.exp_codebook.embedding.weight.data = pretrained_weight.to(device)
    else:
        from graph_tokenizer import GraphTokenizer
        graph_tokenizer = GraphTokenizer(
            codebook_size = args.codebook_size,
            codebook_dims = __embedding.shape[1],
            decoder_dims = num_features,
        ).to(device)
        
        __optimizer = torch.optim.Adam(
            graph_tokenizer.parameters(),
            lr = args.gt_learning_rate if hasattr(args, 'gt_learning_rate') else 1e-3,
            weight_decay = 1e-5
        )
        __criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(args.gt_epochs):
            graph_tokenizer.train()
            total_loss = 0
            token_losses, emb_rec_losses, node_rec_losses, edge_rec_losses, rq_losses = 0, 0, 0, 0, 0
            a_losses, b_losses = 0, 0
            correct = 0
            bst_correct = 0
            for data in tqdm(train_loader):
                data = data.to(device)
                __optimizer.zero_grad()
                
                embed = explained_model.embedding(data.x, data.edge_index)
                
                x_res_a, x_res_b, token_loss_all, emb_rec_loss, node_rec_loss, edge_rec_loss, rq_loss = graph_tokenizer(
                    embed, data.x, data.edge_index, use_sk = True)
                
                prediction_a = explained_model.prediction(x_res_a, batch = data.batch)
                prediction_b = explained_model.prediction(x_res_b, batch = data.batch)
                
                uniform_target = torch.ones_like(prediction_a, dtype=torch.float) / prediction_a.size(1)
                
                loss_a = F.kl_div(F.log_softmax(prediction_a, dim=1), uniform_target, reduction='batchmean')
                loss_b = __criterion(prediction_b, data.y)
                
                final_loss = token_loss_all + loss_a + loss_b
                
                final_loss.backward()
                __optimizer.step()
                
                total_loss += final_loss.item()
                token_losses += token_loss_all.item()
                emb_rec_losses += emb_rec_loss.item()
                node_rec_losses += node_rec_loss.item()
                edge_rec_losses += edge_rec_loss.item()
                rq_losses += rq_loss.item()
                a_losses += loss_a.item()
                b_losses += loss_b.item()
                correct += (prediction_b.argmax(dim = 1) == data.y).sum().item()
                
                if correct > bst_correct:
                    bst_correct = correct
                    codebook_a_weight = graph_tokenizer.codebook_a.embedding.weight.data.cpu()
                    codebook_b_weight = graph_tokenizer.codebook_b.embedding.weight.data.cpu()
            
            logging.info(f'Epoch {epoch + 1}/{args.gt_epochs}, Loss: {total_loss / len(train_loader)}')
            logging.info(f'Token Loss: {token_losses / len(train_loader)}, Embed Rec Loss: {emb_rec_losses / len(train_loader)}, Node Rec Loss: {node_rec_losses / len(train_loader)}, Edge Rec Loss: {edge_rec_losses / len(train_loader)}, RQ Loss: {rq_losses / len(train_loader)}')
            logging.info(f'A Loss: {a_losses / len(train_loader)}, B Loss: {b_losses / len(train_loader)}')
            logging.info(f'Accuracy: {correct / len(train_set)}')
            
            torch.save(codebook_a_weight, 
                        codebook_root + f'Codebook_A_{args.dataset}.pt')
            torch.save(codebook_b_weight, 
                        codebook_root + f'Codebook_B_{args.dataset}.pt')
        
        explainer.algorithm.rv_emd_model.base_codebook.embedding.weight.data = codebook_a_weight.to(device)
        explainer.algorithm.rv_emd_model.exp_codebook.embedding.weight.data = codebook_b_weight.to(device)

    _ = train_explainer(explainer,
                        train_loader, 
                        args.epochs,
                        device = device,
                        smoke = args.max_graphs is not None,
    )
    eval_started = time()
    _ = evaluate_explainer(explainer,
                                    test_loader,
                                    explained_model,
                                    device = device,
                                    gt_mask = True,
                                    dataset = args.dataset,
                                    fidelity_ratios = fidelity_ratios,
                                    auc_record_path = args.auc_record or None,
                                    auc_meta = {
                                        "dataset": args.dataset,
                                        "seed": args.seed,
                                        "protocol": {
                                            "epochs": args.epochs,
                                            "lr": args.lr,
                                            "batch_size": args.batch_size,
                                            "codebook_size": args.codebook_size,
                                            "use_ckpt": args.use_ckpt,
                                            "max_graphs": args.max_graphs,
                                            "device": device,
                                            "embedding_target": "current_batch",
                                        },
                                    })
    logging.info("Evaluate seconds: %.3f", time() - eval_started)

if __name__ == '__main__':
    main()