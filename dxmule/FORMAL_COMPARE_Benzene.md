# Formal compare: single-scale IDEA vs multi-scale DXMulE (Benzene)

**Evidence grade: HYPOTHESIS.** Status: **PENDING_RESULT** until the evidence check. This file records one full-split run. It is not a gate approval, not a method-facts promotion, and not a claim that either method is better. A larger cell in a table below is a logged scalar from this run. It is not a finding.

Task `P001-EXP-G2-001`. Dataset Benzene only. No `--max_graphs`.

Raw logs:

- `dxmule/formal_logs/idea_benzene_full.log`
- `dxmule/formal_logs/dxmule_benzene_full.log`

Regenerate with `bash dxmule/run_formal_benzene.sh`. A rerun can move wall time. The metric lines below are copied from these logs. Per-graph ROC-AUC lists stay in the logs (rounded to 4 decimals there). Quote the `ROC-AUC summary` lines, which use the unrounded mean.

The training and metric code that produced the logs is commit `67d983bfe79a5064ccae5c78830d3b62611757e3`. This markdown file and the logs are added after that commit and do not change `explainer/eval_idea.py`, `explainer/eval_dxmule.py`, or `dxmule/fidelity.py`.

## Protocol

| item | value used |
| --- | --- |
| dataset | Benzene |
| split | repo rule `test_num = int(0.3 * len(dataset))`, then train `dataset[test_num:]`, test `dataset[:test_num]` |
| graphs | 12,000 total, train 8,400, test 3,600 (both logs) |
| seed | 2025 (`random`, NumPy, `torch.manual_seed`; IDEA also calls `torch.cuda.manual_seed_all`) |
| codebook | `--use_ckpt 1`. Both runs loaded the existing files. The tokenizer block did not run. |
| codebook files | `codebook/Codebook_A_Benzene.pt`, `codebook/Codebook_B_Benzene.pt`, each shape `(32, 32)` |
| GNN | `GNN/Benzene.pth`, class `GCN_3layer` (14 → 32, three `GCNConv` layers, 2-way linear head) |
| epochs | 20 for both. That is the argparse default of `explainer/eval_idea.py` and of `explainer/eval_dxmule.py`. The README example that passes `--epochs 10` was not used. |
| batch size | 32 for both (argparse defaults). No OOM, so it was not reduced. |
| learning rate | `0.001` (CLI). Codebook parameters inside `Idea` still use learning rate `0.1` (code, not a CLI flag). |
| codebook size | 32 |
| DXMulE scales | `0.1,0.3,0.5,0.7,0.9` |
| DXMulE weights | cross-scale 0.1, complement 0.1, shell 0.1, retention temperature 1.0 |
| IDEA fidelity grid | the same five ratios, via `--fidelity_scales` |
| truncation | `max_graphs=None` in both `PROTOCOL` lines. The runner refuses `--max_graphs`. |

## Environment (from the log headers)

| item | value |
| --- | --- |
| when (UTC) | IDEA header `2026-11-02T06:48:29Z`; DXMulE header `2026-11-02T06:50:39Z` |
| host | Linux x86_64, 4 CPUs, `nvidia-smi` not found |
| device | CPU. `torch.cuda.is_available()` is false. The runner passed `--device -1`. |
| torch | `2.6.0+cpu` |
| torch-geometric | 2.6.1 |
| geomloss | 0.2.6 |
| numpy / scikit-learn | 1.26.4 / 1.6.1 |
| threads | `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4` |
| memory in the header | about 15 GiB, no swap |
| progress bar | `TQDM_DISABLE=1` |
| exit codes | both 0 |

`requirements.txt` pins a CUDA torch wheel. This machine has no GPU, so the run used the CPU wheel above. Working tree at launch contained the untracked unzipped pickle `data/benzene_with_gt.pkl` (from `data/benzene_with_gt.pkl.zip`). That pickle is local input and is not part of the commit. The DXMulE header also lists `dxmule/formal_logs/` because the IDEA log had just been created.

Before the run, `python3 -m dxmule.test_core` passed (14 tests). Those tests check retention masks, the cross-scale map, and the fidelity arithmetic on synthetic tensors. They do not score Benzene.

## Commands

Recorded in the logs (`printf %q`, so commas are escaped). Readable form:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TQDM_DISABLE=1 \
python3 -u explainer/eval_idea.py \
  --dataset Benzene --root_path ./ --device -1 \
  --epochs 20 --lr 0.001 --batch_size 32 \
  --seed 2025 --use_ckpt 1 --codebook_size 32 \
  --fidelity_scales 0.1,0.3,0.5,0.7,0.9

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TQDM_DISABLE=1 \
python3 -u explainer/eval_dxmule.py \
  --dataset Benzene --root_path ./ --device -1 \
  --epochs 20 --lr 0.001 --batch_size 32 \
  --seed 2025 --use_ckpt 1 --codebook_size 32 \
  --scales 0.1,0.3,0.5,0.7,0.9 \
  --cross_scale_weight 0.1 --complement_weight 0.1 \
  --shell_weight 0.1 --retention_temperature 1.0
```

## Verified process facts

These are properties of this run, not quality claims.

- Both processes exited 0. `PROTOCOL` shows `max_graphs=None`, epochs 20, batch size 32, seed 2025, `use_ckpt=1`.
- IDEA logged `Loaded pretrained GNN from ./GNN/Benzene.pth` and `Loading pretrained codebooks from ./codebook/`.
- DXMulE logged `Loaded pretrained codebooks from /workspace/codebook`. Its shape check raises if A/B do not match the explainer. It did not raise.
- DXMulE logged `Hard retention masks nested: True (n=3600 graphs)`. A violation would have raised before that line.
- Both logs contain `FIDELITY mask_sensitivity=True graph_index=0` and `FIDELITY status=OK`. An all-zero edge mask changed the GNN logits relative to an all-one mask on the first test graph, so ACC@p and Fidelity@p were computed. `skipped_no_edges=0` and `skipped_nonfinite=0` for both.
- No batch-size reduction. No crash.

## Shared-score ROC-AUC (HYPOTHESIS)

The score is the sigmoid of the shared edge logit. DXMulE soft masks are monotone in that logit, so they are not given a separate AUC.

The two means are **not on the same graph set**. Both entry points skip a test graph whose ground-truth `edge_mask` is constant. DXMulE also skips a graph whose explanation score is constant. IDEA does not. Counts:

| rule | IDEA | DXMulE |
| --- | --- | --- |
| test graphs | 3600 | 3600 |
| skipped, constant ground-truth edge label | 1762 | 1762 |
| skipped, constant explanation score | not applied | 88 |
| scored `n` | 1838 | 1750 |

1762 + 1838 = 3600 and 1762 + 88 + 1750 = 3600. The 88 constant-score graphs sit inside IDEA's scored set. Do not subtract the means or treat the gap as a method effect.

`n_recall` for DXMulE hard masks is 1838, the same count as the mixed-label graphs. The recall average drops graphs with zero positive ground-truth edges. Read with the 1762 constant-label skips, those 1762 test graphs are all-zero ground-truth edge masks. That is a reading of the two counters and the code rule, not a second pass over the pickle.

| | IDEA | DXMulE |
| --- | --- | --- |
| mean | 0.183885 | 0.719092 |
| std (NumPy `ddof=0`) | 0.180245 | 0.110779 |
| min | 0.000000 | 0.494048 |
| max | 0.921053 | 1.000000 |
| n | 1838 | 1750 |
| log line, 4 d.p. mean | `ROC AUC: 0.1839 (n=1838)` | `Shared-score ROC-AUC 0.7191 (n=1750)` |

## DXMulE hard top-k vs ground-truth edges (HYPOTHESIS)

Nested hard masks, one shared ranking. Precision and recall are overlap with `edge_mask`, not fidelity. Realized retention is `ceil(k * n_edges) / n_edges`, averaged over graphs, so it sits slightly above the requested `k`. Recall is the mean over the 1838 graphs with at least one positive ground-truth edge. Precision and retention use all 3600 test graphs.

| k | realized retention | precision | recall | n_retention | n_precision | n_recall |
| --- | --- | --- | --- | --- | --- | --- |
| 0.10 | 0.110137 | 0.389861 | 0.320366 | 3600 | 3600 | 1838 |
| 0.30 | 0.309893 | 0.346331 | 0.797969 | 3600 | 3600 | 1838 |
| 0.50 | 0.500000 | 0.256067 | 0.952031 | 3600 | 3600 | 1838 |
| 0.70 | 0.709659 | 0.188928 | 0.995466 | 3600 | 3600 | 1838 |
| 0.90 | 0.909414 | 0.148107 | 0.999683 | 3600 | 3600 | 1838 |

IDEA does not log per-k precision, recall, or retention. Those columns are DXMulE only.

## ACC@p and Fidelity@p (HYPOTHESIS)

Computed in this run. Definition, copied from both logs:

> ACC@p is the fraction of graphs whose frozen-GNN argmax on the hard top-p subgraph equals the unmasked argmax. Fidelity+@p (necessity) is the mean drop in softmax probability of that unmasked class when the hard top-p edges are removed. Fidelity-@p (sufficiency) is the mean drop when only the hard top-p edges are kept. Hard top-p keeps ceil(p * n_edges) edges, at least 1 if the graph has edges, ranked by the shared explanation score with a stable argsort, per graph. This is `dxmule.scales.edges_to_keep`. GCNConv self-loops are not part of the stored `edge_index`; the explain hook leaves those loop messages at weight 1. Ground-truth edge labels are not used. IDEA and DXMulE training objectives are different and are not fidelity.

How to read a cell, without turning it into a ranking: higher ACC@p means the kept subgraph more often preserves the unmasked argmax. Higher Fidelity+@p means removing the top-p edges drops the original class probability more. Lower Fidelity-@p means the kept subgraph preserves more of that probability. `n=3600` is every test graph, including the constant-label graphs left out of ROC-AUC.

DXMulE uses the same hard masks that were checked for nesting. IDEA builds the hard mask from its sigmoid edge score with the same keep rule. Sigmoid is strictly increasing, so the ranking matches the logit ranking, ties included, under a stable argsort.

| p | IDEA ACC@p | IDEA Fidelity+ | IDEA Fidelity- | DXMulE ACC@p | DXMulE Fidelity+ | DXMulE Fidelity- |
| --- | --- | --- | --- | --- | --- | --- |
| 0.10 | 0.550556 | 0.051095 | 0.387317 | 0.552778 | 0.342024 | 0.400408 |
| 0.30 | 0.503333 | 0.134610 | 0.408198 | 0.614167 | 0.428658 | 0.321464 |
| 0.50 | 0.457500 | 0.257847 | 0.436350 | 0.757222 | 0.412226 | 0.190124 |
| 0.70 | 0.495556 | 0.453185 | 0.408434 | 0.853056 | 0.379671 | 0.098220 |
| 0.90 | 0.643889 | 0.392996 | 0.280439 | 0.924444 | 0.385782 | 0.040272 |

Not measured, and not claimed: any other fidelity formula, a deletion curve at ratios other than this grid, sufficiency/necessity under a different probability, and a paired test. Those are out of scope. They are not UNKNOWN inside this protocol: the protocol above did run. A different formula would be a different measurement.

## Times

| | IDEA | DXMulE |
| --- | --- | --- |
| train-loop seconds | 78.723, includes the unmasked-embedding precompute | 421.742, epoch loop only |
| evaluate seconds | 43.301 | 43.855 |
| process wall (`ELAPSED_SEC`) | 127.796 | 471.641 |

IDEA's last epoch line says `Time: 78.72s`, the same clock as the train-loop line. Process wall includes unpickling and evaluation. These are CPU times on 4 threads. They are not a speed result against a GPU run, because no GPU was present.

## Training losses (not comparable)

IDEA's logged objective is the single-scale `RV_model` total (`rqvae_loss`). The line labeled "Codebook Loss" is the reconstruction term returned by `Idea.train`, not a separate codebook penalty. Distribution is the Sinkhorn term.

DXMulE's `loss` is `rv + 0.1 * cross + 0.1 * complement`, with `rv` the mean of that `RV_model` total over the five retention ratios. Epoch 1: `0.7458 + 0.1 * 0.1641 + 0.1 * 0.4787 = 0.8101`. Epoch 20: `0.4326 + 0.1 * 0.2910 + 0.1 * 0.6104 = 0.5227` (log rounding to 4 decimals).

The two objectives are different. A smaller IDEA number and a smaller DXMulE number are not two measurements of one loss, and neither is evidence that a method is better. Concrete temperature is `5.0 * (2.0 / 5.0) ** (epoch / epochs)`, so the path from epoch 1 to epoch 20 mixes that schedule with the parameter update.

IDEA, from `idea_benzene_full.log`:

| epoch | RQVAE total | reconstruction (logged as Codebook Loss) | distribution | cumulative seconds |
| --- | --- | --- | --- | --- |
| 1 | 0.6919 | 0.2819 | 0.0014 | 4.39 |
| 2 | 0.4297 | 0.1680 | 0.0009 | 8.26 |
| 3 | 0.4080 | 0.1582 | 0.0009 | 12.23 |
| 4 | 0.3887 | 0.1495 | 0.0011 | 16.13 |
| 5 | 0.3680 | 0.1400 | 0.0013 | 20.03 |
| 6 | 0.3580 | 0.1359 | 0.0013 | 23.93 |
| 7 | 0.3562 | 0.1350 | 0.0012 | 27.94 |
| 8 | 0.3497 | 0.1322 | 0.0012 | 31.81 |
| 9 | 0.3507 | 0.1325 | 0.0012 | 35.69 |
| 10 | 0.3426 | 0.1288 | 0.0011 | 39.64 |
| 11 | 0.3351 | 0.1255 | 0.0010 | 43.52 |
| 12 | 0.3192 | 0.1187 | 0.0008 | 47.37 |
| 13 | 0.2440 | 0.0854 | 0.0005 | 51.23 |
| 14 | 0.2313 | 0.0798 | 0.0005 | 55.17 |
| 15 | 0.2294 | 0.0789 | 0.0005 | 59.16 |
| 16 | 0.2295 | 0.0791 | 0.0005 | 63.08 |
| 17 | 0.2292 | 0.0787 | 0.0005 | 67.01 |
| 18 | 0.2295 | 0.0789 | 0.0005 | 70.91 |
| 19 | 0.2268 | 0.0778 | 0.0005 | 74.82 |
| 20 | 0.2253 | 0.0772 | 0.0005 | 78.72 |

DXMulE, from `dxmule_benzene_full.log`. All of these losses were finite. The script raises if the averaged total is non-finite. It did not raise.

| epoch | total | rv | dist | recon | cross | top-down KL | reverse KL | shell | persistence | complement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8101 | 0.7458 | 0.0013 | 0.3060 | 0.1641 | 0.0256 | 0.0282 | 1.1028 | 0.650 | 0.4787 |
| 2 | 0.6301 | 0.5539 | 0.0009 | 0.2232 | 0.2344 | 0.0668 | 0.0763 | 0.9127 | 0.695 | 0.5281 |
| 3 | 0.5797 | 0.4965 | 0.0007 | 0.1973 | 0.2606 | 0.0822 | 0.0949 | 0.8338 | 0.727 | 0.5714 |
| 4 | 0.5526 | 0.4655 | 0.0006 | 0.1840 | 0.2725 | 0.0883 | 0.1021 | 0.8212 | 0.747 | 0.5986 |
| 5 | 0.5439 | 0.4547 | 0.0006 | 0.1788 | 0.2785 | 0.0913 | 0.1057 | 0.8148 | 0.761 | 0.6132 |
| 6 | 0.5402 | 0.4508 | 0.0006 | 0.1770 | 0.2842 | 0.0940 | 0.1074 | 0.8277 | 0.773 | 0.6104 |
| 7 | 0.5329 | 0.4427 | 0.0006 | 0.1736 | 0.2887 | 0.0963 | 0.1102 | 0.8218 | 0.781 | 0.6125 |
| 8 | 0.5276 | 0.4372 | 0.0006 | 0.1718 | 0.2853 | 0.0947 | 0.1094 | 0.8113 | 0.785 | 0.6184 |
| 9 | 0.5274 | 0.4376 | 0.0006 | 0.1712 | 0.2876 | 0.0958 | 0.1101 | 0.8170 | 0.789 | 0.6104 |
| 10 | 0.5336 | 0.4430 | 0.0005 | 0.1728 | 0.2890 | 0.0960 | 0.1105 | 0.8245 | 0.794 | 0.6172 |
| 11 | 0.5244 | 0.4341 | 0.0006 | 0.1695 | 0.2900 | 0.0969 | 0.1106 | 0.8251 | 0.797 | 0.6124 |
| 12 | 0.5260 | 0.4360 | 0.0006 | 0.1702 | 0.2882 | 0.0963 | 0.1106 | 0.8129 | 0.800 | 0.6123 |
| 13 | 0.5275 | 0.4378 | 0.0005 | 0.1709 | 0.2883 | 0.0965 | 0.1103 | 0.8146 | 0.802 | 0.6085 |
| 14 | 0.5203 | 0.4297 | 0.0006 | 0.1682 | 0.2915 | 0.0978 | 0.1108 | 0.8291 | 0.805 | 0.6139 |
| 15 | 0.5221 | 0.4318 | 0.0006 | 0.1689 | 0.2902 | 0.0974 | 0.1112 | 0.8158 | 0.806 | 0.6125 |
| 16 | 0.5244 | 0.4340 | 0.0006 | 0.1698 | 0.2888 | 0.0969 | 0.1106 | 0.8142 | 0.807 | 0.6146 |
| 17 | 0.5239 | 0.4338 | 0.0006 | 0.1692 | 0.2916 | 0.0978 | 0.1118 | 0.8212 | 0.807 | 0.6088 |
| 18 | 0.5227 | 0.4317 | 0.0006 | 0.1687 | 0.2926 | 0.0983 | 0.1123 | 0.8203 | 0.808 | 0.6172 |
| 19 | 0.5195 | 0.4296 | 0.0006 | 0.1680 | 0.2890 | 0.0969 | 0.1114 | 0.8072 | 0.808 | 0.6101 |
| 20 | 0.5227 | 0.4326 | 0.0005 | 0.1684 | 0.2910 | 0.0978 | 0.1127 | 0.8048 | 0.806 | 0.6104 |

## Limitations left in the scripts

1. **IDEA embedding targets vs batches.** IDEA pools unmasked embeddings once, then trains with `DataLoader(shuffle=True)` and indexes those stored targets by batch position. DXMulE recomputes the unmasked embedding on the batch it is training. This run used batch size 32 and shuffle, so IDEA's pairing can attach a graph to another graph's target. That behavior was not rewritten for this run.
2. **AUC inclusion** differs, as counted above. No matched-subset AUC was computed after the fact.
3. **GNN freezing.** DXMulE sets `requires_grad_(False)` on the loaded GNN. IDEA does not. Neither optimizer steps GNN parameters. Forward weights come from `GNN/Benzene.pth`.
4. **CUDA seed.** IDEA calls `torch.cuda.manual_seed_all(2025)`. DXMulE does not. There was no CUDA device. Python, NumPy, and CPU torch seeds are 2025 in both processes.
5. **Self-loops.** `GCNConv` adds self-loops that are not rows of `edge_index`. Fidelity does not mask them. `mask_sensitivity=True` only shows that zeroing the stored edges changes logits.
6. **One seed, one CPU machine, loss weights at the code defaults.** Nothing was tuned. No interval and no hypothesis test was run.
7. **README example** uses `--epochs 10`, `--lr 0.0005`, and `--batch_size 64`. This run used the argparse defaults (20, 0.001, 32), passed explicitly.

## Out of scope

Other datasets. MSExplainer, GNNExplainer, PGExplainer, and a C2 ablation matrix. Ranking the two methods. Promoting these numbers to method facts. Merging this branch on the strength of the tables.
