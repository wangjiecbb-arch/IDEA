# CPU smoke: single-scale IDEA vs multi-scale DXMulE (Benzene)

**Status: HYPOTHESIS / smoke / provisional.**

This file records one CPU session that ran both existing entry points to completion on the same machine. It is a plumbing check on a tiny slice of Benzene. It is not a paper result, not a tuned comparison, and not evidence that either method is better. A larger number in the table below is a logged scalar from this run. It is not a finding.

Raw logs from the session:

- `dxmule/smoke_logs/idea_cpu_smoke.log`
- `dxmule/smoke_logs/dxmule_cpu_smoke.log`

Regenerate with `bash dxmule/run_cpu_smoke.sh`. A rerun can move the process wall time. The loss, AUC, and retention figures below were stable across repeats in this session; the wall times are the ones in the committed logs.

## Session

| item | value |
| --- | --- |
| when (UTC) | IDEA 2026-11-02T03:36:07Z, DXMulE 2026-11-02T03:38:15Z |
| host | Linux x86_64, 4 CPUs, no CUDA (`torch.cuda.is_available()` is false) |
| torch | 2.6.0+cpu |
| torch-geometric | 2.6.1 |
| geomloss | 0.2.6 (PyTorch backend; PyKeOps was not installed) |
| numpy / scikit-learn | 1.26.4 / 1.6.1 |
| threads | `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4` |
| exit codes | both 0 |

`requirements.txt` pins a CUDA torch wheel. This session did not install that wheel. CPU install:

```bash
python3 -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install torch-geometric==2.6.1 geomloss==0.2.6 \
  scikit-learn==1.6.1 numpy==1.26.4 scipy==1.15.2 tqdm==4.67.1
```

Benzene was unzipped from `data/benzene_with_gt.pkl.zip` to `data/benzene_with_gt.pkl` (57,357,301 bytes). DXMulE unzips that archive itself if the pickle is missing. IDEA reads the pickle only. The pickle is local input and is not part of this change.

Before the smoke, `python3 -m dxmule.test_core` passed (9 tests). Those tests check retention masks and the cross-scale map on synthetic tensors. They do not score explanations.

## Commands

Same seed, learning rate, batch size, epoch count, codebook size, and pretrained checkpoints. DXMulE scales and loss weights are the script defaults, written out so the log is explicit.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python3 -u explainer/eval_idea.py \
  --dataset Benzene --root_path ./ --device -1 \
  --epochs 2 --lr 0.001 --batch_size 4 --max_graphs 8 \
  --seed 2025 --use_ckpt 1 --codebook_size 32

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python3 -u explainer/eval_dxmule.py \
  --dataset Benzene --root_path ./ --device -1 \
  --epochs 2 --lr 0.001 --batch_size 4 --max_graphs 8 \
  --seed 2025 --use_ckpt 1 --codebook_size 32 \
  --scales 0.1,0.3,0.5,0.7,0.9 \
  --cross_scale_weight 0.1 --complement_weight 0.1 \
  --shell_weight 0.1 --retention_temperature 1.0
```

`--max_graphs` was added to `explainer/eval_idea.py` so the single-scale path can take the same head-of-split truncation that `explainer/eval_dxmule.py` already had. With the flag unset, IDEA's full-split behavior is unchanged. Smoke-only logs (batch fingerprints, last-step gradient check, DXMulE train-loop seconds, per-graph AUC values) are also gated on `--max_graphs` or are extra lines in the existing evaluation print.

README launch snippet uses `--gt_epoch` and `--lr 0.0005`. The scripts accept `--gt_epochs` (plural). This run passed `--use_ckpt 1`, so the graph-tokenizer block did not run and `--gt_epochs` was unused. The learning rate is the shared argparse default, `0.001`, passed explicitly.

## What was aligned

Both runs:

- device `cpu` via `--device -1`
- dataset Benzene, 12,000 graphs, 14 node features, 2 classes on the full test split
- split from the existing scripts: `test_num = int(0.3 * 12000) = 3600`, train `dataset[3600:]` (8,400), test `dataset[:3600]`
- smoke slice after that split: train head 8 graphs, test head 2 graphs (`max_graphs // 4`)
- seed 2025 for `random`, NumPy, and `torch.manual_seed`
- frozen-in-practice GNN weights from `GNN/Benzene.pth` (see mismatches: only DXMulE sets `requires_grad_(False)`)
- pretrained codebooks `codebook/Codebook_A_Benzene.pt` and `codebook/Codebook_B_Benzene.pt`, shape `(32, 32)`, matching embedding width 32
- batch size 4 (two train batches), 2 epochs, MLP learning rate 0.001, codebook learning rate 0.1 inside `Idea`

The two test graphs are `dataset[0]` and `dataset[1]`, in that order (`shuffle=False`, batch size 1). Each has 52 stored edges and 12 positive `edge_mask` entries. Both have mixed edge labels, so both enter ROC-AUC.

## Side-by-side

Logged training losses are finite (ordinary decimals, no NaN/Inf). DXMulE also raises if its averaged total is non-finite; it did not raise.

IDEA's printed objective is the single-scale `RV_model` total (`rqvae_loss`, coefficient `emd = 1`). The line labeled "Codebook Loss" is the third value returned by `Idea.train`, which is `RV_model`'s reconstruction term, not a separate codebook penalty. Distribution is the Sinkhorn term. Those two are pieces of the RQVAE total; the rest is the vector-quantization commitment terms, which the IDEA log does not print on their own.

DXMulE's `loss` is `rv + 0.1 * cross + 0.1 * complement`, with `rv` the mean of the same `RV_model` total over the five retention ratios. Epoch 1: `4.3163 + 0.1 * 0.2557 + 0.1 * 0.2576 = 4.3676`. Epoch 2: `3.1231 + 0.1 * 0.1433 + 0.1 * 0.2443 = 3.1619`.

| metric | IDEA | DXMulE | notes |
| --- | --- | --- | --- |
| run | exit 0, device cpu | exit 0, device cpu | `--device -1` |
| graphs in the optimizer / AUC | 8 train, 2 test | 8 train, 2 test | head of the shared Benzene split |
| epoch 1 objective | RQVAE 4.3116 | total 4.3676 | different objectives; see above |
| epoch 1 RV total | 4.3116 | 4.3163 (mean over 5 scales) | IDEA figure is the single mask |
| epoch 1 distribution | 0.0017 | 0.0019 | Sinkhorn term |
| epoch 1 reconstruction | 1.3826 (logged as "Codebook Loss") | 1.3846 (`recon`) | same RV component, different label |
| epoch 1 cross-scale | none | 0.2557 (top-down KL 0.0011, reverse KL 0.0011, shell 2.5349, persistence 0.638) | weight 0.1 on the sum |
| epoch 1 complement | none | 0.2576 | weight 0.1 |
| epoch 2 objective | RQVAE 3.5278 | total 3.1619 | temperature also changed; see caveats |
| epoch 2 RV total | 3.5278 | 3.1231 | |
| epoch 2 distribution | 0.0015 | 0.0018 | |
| epoch 2 reconstruction | 1.1425 (logged as "Codebook Loss") | 1.0406 (`recon`) | |
| epoch 2 cross-scale | none | 0.1433 (top-down KL 0.0032, reverse KL 0.0031, shell 1.3702, persistence 0.638) | |
| epoch 2 complement | none | 0.2443 | |
| shared-score ROC-AUC | 0.8896 (n=2) values 1.0000, 0.7792 | 0.9042 (n=2) values 0.9708, 0.8375 | same two graphs, same order. Smoke only. No ranking. |
| hard masks nested | single mask, no `{k_m}` | True on both test graphs | finer hard mask ⊆ next coarser mask. A failure would have raised. |
| realized retention vs requested `k_m` | no retention axis | 0.115 / 0.10, 0.308 / 0.30, 0.500 / 0.50, 0.712 / 0.70, 0.904 / 0.90 | matches `ceil(k * 52)` on both test graphs |
| hard top-k precision (n=2) | not computed | 0.667, 0.594, 0.462, 0.324, 0.255 at k = 0.1, 0.3, 0.5, 0.7, 0.9 | overlap with `edge_mask`, not fidelity |
| hard top-k recall (n=2) | not computed | 0.333, 0.792, 1.000, 1.000, 1.000 | same order of k. 12 positive edges out of 52. |
| fidelity ACC@p | not computed | not computed | neither script implements it. Left out on purpose. |
| last-step gradients | finite, 6 parameters with nonzero grad, abs-sum 3.534387 | finite, 7 parameters with nonzero grad, abs-sum 3.010831 | after the last `backward` and optimizer step. The extra DXMulE parameter group is the cross-scale map. |
| train-loop time | 0.06 s (script timer at the end of epoch 2, includes the precompute pass) | 0.293 s (`SMOKE train loop seconds`) | DXMulE runs five masked embeddings, five complement embeddings, and five RV calls per step |
| process wall time | 5.729 s | 6.217 s | both spend about 3 s unpickling all 12,000 graphs before the slice. This gap is not a speed result. |

Soft retention masks are a monotone function of the same edge scores, so DXMulE does not report a separate ROC-AUC per scale. The shared-score AUC is the sigmoid of the MLP logit, the same edge interface IDEA evaluates.

## Mismatches left in place

These are properties of the current scripts. The smoke did not rewrite either objective to hide them.

1. **Embedding targets vs batches.** IDEA walks the loader once, stores pooled embeddings, then trains. `DataLoader(shuffle=True)` draws a new order on the next pass, and IDEA still indexes the stored targets by batch position. Fingerprints (node counts per graph in the batch):
   - IDEA precompute, and DXMulE epoch 1: `[[22, 25, 23, 21], [9, 25, 21, 24]]`
   - IDEA epoch 1: `[[9, 21, 21, 22], [24, 23, 25, 25]]`
   DXMulE recomputes the unmasked embedding on the batch it is training. On this run the two epoch-1 updates did not see the same pairing of graphs. With one batch (`batch_size >= max_graphs`) that particular mismatch would disappear; this smoke kept batch size 4, as requested.

2. **Objectives.** IDEA optimizes one concrete edge sample through `RV_model`. DXMulE averages that term over `k ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` and adds the cross-scale link and the complement alignment. The epoch totals are not two measurements of one loss.

3. **GNN freezing.** DXMulE sets `requires_grad_(False)` on the loaded GNN. IDEA does not. Neither optimizer steps GNN parameters. Forward weights still come from `GNN/Benzene.pth`.

4. **Class count.** IDEA counts classes on the full 3,600-graph test split before truncation and logs "Training graphs: 8400, Testing graphs: 3600", then logs the truncation. DXMulE counts classes on the 10 smoke graphs. Both counts were 2.

5. **Checkpoint log line.** IDEA prints `Loaded pretrained GNN from model ./models/Benzene.pt`. The path `init_model` actually opens is `./GNN/Benzene.pth`. The load succeeded from that `.pth` file. DXMulE logs the codebook directory it used (`/workspace/codebook`).

6. **CUDA seed call.** IDEA calls `torch.cuda.manual_seed_all(2025)`. DXMulE does not. On this CPU build the call returns without a device. The CPU, NumPy, and Python seeds are 2025 in both processes.

7. **Temperature schedule.** Concrete noise uses `temp = 5.0 * (2.0 / 5.0) ** (epoch / epochs)`. For 2 epochs that is 5.0 on the first epoch and about 3.162 on the second. The logged drop from epoch 1 to epoch 2 mixes that schedule with the parameter update and with a new `torch.rand` draw. It is not a learning curve.

8. **README flag name.** `--gt_epoch` in the README is not a flag on either script. The implemented name is `--gt_epochs`. Unused here because `--use_ckpt 1` found both codebook files.

## Caveats

- Loss weights are the code defaults (`emd = 1`, cross-scale 0.1, complement 0.1, shell 0.1). Nothing was tuned.
- The fit sees 8 graphs for 2 epochs. Several of those training graphs have an all-zero ground-truth edge mask. The training loss does not read that mask; the AUC does, and the AUC uses the other split (2 graphs).
- ROC-AUC, precision, and recall are means over those 2 graphs. One graph can move the mean by a large amount (IDEA's two values are 1.00 and 0.78).
- Recall of 1.0 at `k ≥ 0.5` means the 12 positive edges were inside the kept half of a 52-edge graph on this draw. It is a description of these two masks.
- This is not G2 evidence, not a leaderboard entry, and not a reason to prefer DXMulE or IDEA. Fidelity (ACC@p, deletion, sufficiency, necessity) was not run.
- Process wall time is mostly the pickle load. Compare the train-loop lines if the question is whether the step finished.
