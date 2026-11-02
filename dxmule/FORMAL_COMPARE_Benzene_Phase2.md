# Phase-2 fair compare: IDEA vs DXMulE (Benzene)

**Evidence grade: HYPOTHESIS.** Status: **PENDING_RESULT** until an evidence check. This file is not a gate approval, not a method-facts promotion, and not a claim that either method is better. A larger cell below is a logged scalar from this run. It is not a finding. Task `P001-EXP-G2-002` is not G2 approved.

Loss values are **not comparable** across methods. IDEA logs the single-scale `RV_model` total. DXMulE logs `rv + 0.1 * cross + 0.1 * complement`, with `rv` the mean of that total over five retention ratios. A smaller number on one side is not a measurement of the other objective.

Phase-1 is `dxmule/FORMAL_COMPARE_Benzene.md` (one seed, unequal ROC-AUC graph sets, IDEA embedding targets cached by batch index). This file does not replace that record.

The training and metric code that produced these logs is commit `01e9252e4ceea507eb93828fed825445fd96aac3`. This markdown file and `dxmule/formal_logs_phase2/` are added after that commit and do not change the entry points.

## What changed relative to Phase-1

1. **Matched-subset ROC-AUC.** One shared skip rule, applied in `dxmule/matched_auc.py` after both per-graph records exist. A test graph is dropped for **both** methods when the ground-truth `edge_mask` is constant, or IDEA's explanation scores are constant, or DXMulE's explanation scores are constant, or either score vector is non-finite or empty. Constant means `numpy.allclose` of every score to the first score. Scores are the sigmoid edge masks (`Explanation.edge_mask`). The kept indices are the intersection, so the two means in a seed have the same `n`. ACC@p and Fidelity±@p are not filtered by this rule.
2. **IDEA embedding target.** `explainer/eval_idea.py` calls `dxmule.embedding_target.unmasked_embedding_target` on the batch just drawn. It does not cache a list and index it by DataLoader position. `DataLoader(shuffle=True)` changes batch membership every iteration, so the Phase-1 cache could pair a graph with another graph's pooled embedding. DXMulE uses the same helper. Both logs contain the line that the target is recomputed on the current batch.

`python3 -m dxmule.test_core` passed before the runs: 27 tests, `OK` (`dxmule/formal_logs_phase2/test_core.log`). Those tests do not score Benzene.

## Protocol

| item | value used |
| --- | --- |
| dataset | Benzene |
| split | repo rule `test_num = int(0.3 * len(dataset))`, then train `dataset[test_num:]`, test `dataset[:test_num]` |
| graphs | 12,000 total, train 8,400, test 3,600 (every log) |
| seeds | 2025, 2026, 2027 (`random`, NumPy, `torch.manual_seed`; both entry points call `torch.cuda.manual_seed_all`) |
| codebook | `--use_ckpt 1`. Both runs loaded the existing files. The tokenizer block did not run. |
| codebook files | `codebook/Codebook_A_Benzene.pt`, `codebook/Codebook_B_Benzene.pt` |
| GNN | `GNN/Benzene.pth` |
| epochs | 20 |
| batch size | 32 |
| learning rate | `0.001` (CLI). Codebook parameters inside `Idea` / `DXMulE` still use learning rate `0.1` (code, not a CLI flag). |
| codebook size | 32 |
| scales | `0.1,0.3,0.5,0.7,0.9` (DXMulE retention and IDEA fidelity grid) |
| DXMulE weights | cross-scale 0.1, complement 0.1, shell 0.1, retention temperature 1.0 |
| truncation | `max_graphs=None` in every `PROTOCOL` line. The runner refuses `--max_graphs`. The matched step was called with `--require-full-split --expect-n-test 3600`. |
| embedding target | `current_batch` in every `PROTOCOL` line |

## Environment (from the log headers)

| item | value |
| --- | --- |
| when (UTC) | IDEA seed 2025 header `2026-11-02T07:31:22Z`. Later seeds follow in the same process list. DXMulE seed 2027 header `2026-11-02T07:57:20Z`. |
| host | Linux x86_64, `nproc` 4, `nvidia-smi` not found |
| device | CPU. `torch.cuda.is_available()` is false. The runner passed `--device -1`. |
| python | `/home/ubuntu/.venv/bin/python3` |
| torch | `2.6.0+cpu` |
| torch-geometric | 2.6.1 |
| geomloss | 0.2.6 |
| numpy / scikit-learn | 1.26.4 / 1.6.1 |
| threads | `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4` |
| memory in the header | about 15 GiB, no swap |
| progress bar | `TQDM_DISABLE=1` |
| exit codes | all six training processes 0. Runner `DRIVER_EXIT:0`. |

Working tree at launch contained the untracked unzipped pickle `data/benzene_with_gt.pkl` (57,357,301 bytes, from `data/benzene_with_gt.pkl.zip`). That pickle is local input and is not part of the commit. Later logs also list `dxmule/formal_logs_phase2/` as untracked because earlier logs already existed.

## Commands

Regenerate with `bash dxmule/run_formal_benzene_phase2.sh`. A rerun can move wall time. Metric lines below are copied from the logs. Quote `MATCHED` lines for ROC-AUC (six decimals). Per-graph flags are the JSON records, not a second hand calculation.

Readable form of one seed (`2025`; `2026` and `2027` are the same with `--seed` changed). Commas are escaped in the log's `printf %q` copy.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TQDM_DISABLE=1 \
python3 -u explainer/eval_idea.py \
  --dataset Benzene --root_path ./ --device -1 \
  --epochs 20 --lr 0.001 --batch_size 32 \
  --seed 2025 --use_ckpt 1 --codebook_size 32 \
  --fidelity_scales 0.1,0.3,0.5,0.7,0.9 \
  --auc_record dxmule/formal_logs_phase2/idea_s2025_auc.json

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TQDM_DISABLE=1 \
python3 -u explainer/eval_dxmule.py \
  --dataset Benzene --root_path ./ --device -1 \
  --epochs 20 --lr 0.001 --batch_size 32 \
  --seed 2025 --use_ckpt 1 --codebook_size 32 \
  --scales 0.1,0.3,0.5,0.7,0.9 \
  --cross_scale_weight 0.1 --complement_weight 0.1 \
  --shell_weight 0.1 --retention_temperature 1.0 \
  --auc_record dxmule/formal_logs_phase2/dxmule_s2025_auc.json

python3 -m dxmule.matched_auc \
  --idea dxmule/formal_logs_phase2/idea_s2025_auc.json \
  --dxmule dxmule/formal_logs_phase2/dxmule_s2025_auc.json \
  --out dxmule/formal_logs_phase2/matched_s2025.json \
  --require-full-split --expect-n-test 3600
```

The runner then aggregates `matched_s2025.json`, `matched_s2026.json`, and `matched_s2027.json` into `dxmule/formal_logs_phase2/matched_summary.json`.

## Verified process facts

These are properties of this run, not quality claims.

- All six processes exited 0. Every `PROTOCOL` line shows `max_graphs=None`, epochs 20, batch size 32, `use_ckpt=1`, `embedding_target=current_batch`.
- IDEA logged `Loaded pretrained GNN from ./GNN/Benzene.pth` and `Loading pretrained codebooks from ./codebook/`.
- DXMulE logged `Loaded pretrained codebooks from /workspace/codebook` on each seed. The shape check did not raise.
- Each DXMulE log contains `Hard retention masks nested: True (n=3600 graphs)`.
- Every log contains `FIDELITY mask_sensitivity=True graph_index=0` and `FIDELITY status=OK`. `skipped_no_edges=0` and `skipped_nonfinite=0`.
- Matched `n_test=3600` on every seed. `MATCHED_EQUAL_N` has the same integer on both methods in every seed.
- Skip buckets plus matched `n` sum to 3600 on every seed (1762 + 5 + 88 + 0 + 1745; 1762 + 8 + 94 + 0 + 1736; 1762 + 4 + 104 + 1 + 1729).

## Matched ROC-AUC (HYPOTHESIS)

This is the fair ROC-AUC for this task. Within a seed, `std_ddof0` is the population standard deviation of the per-graph AUCs on the matched subset (`numpy` `ddof=0`). Across seeds, the summary is the unweighted mean of the three per-seed means. `std_ddof1` is the sample standard deviation of those three means. Seeds are not pooled into one multiset. Matched `n` is not the same integer on every seed, because which graphs have constant scores depends on the trained scores. The across-seed line does not weight seeds by `n`. It is not a significance test.

| seed | n | IDEA mean | IDEA std (ddof=0) | IDEA min | IDEA max | DXMulE mean | DXMulE std (ddof=0) | DXMulE min | DXMulE max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | 1745 | 0.165881 | 0.165700 | 0.000000 | 0.916667 | 0.719087 | 0.110858 | 0.494048 | 1.000000 |
| 2026 | 1736 | 0.142173 | 0.162031 | 0.000000 | 0.916667 | 0.715131 | 0.110057 | 0.494048 | 1.000000 |
| 2027 | 1729 | 0.195770 | 0.174452 | 0.000000 | 0.944444 | 0.707085 | 0.108692 | 0.511905 | 1.000000 |

Across seeds, from `matched_summary.log`:

| | mean of per-seed means | std of per-seed means (ddof=1) | per-seed n |
| --- | --- | --- | --- |
| IDEA | 0.167941 | 0.026858 | 1745, 1736, 1729 |
| DXMulE | 0.713768 | 0.006116 | 1745, 1736, 1729 |

### Skip counts (same rule, both methods)

| seed | n_test | constant ground-truth label | IDEA score constant only | DXMulE score constant only | both scores constant | non-finite or empty | matched n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | 3600 | 1762 | 5 | 88 | 0 | 0 | 1745 |
| 2026 | 3600 | 1762 | 8 | 94 | 0 | 0 | 1736 |
| 2027 | 3600 | 1762 | 4 | 104 | 1 | 0 | 1729 |

### Own-rule counts are not the fair comparison

Phase-1 IDEA scored every mixed-label graph, including constant explanation scores. Phase-1 DXMulE also dropped constant explanation scores. Those two counts still differ here. `idea_own_rule_n` is 1838 on every seed (3600 − 1762). `dxmule_own_rule_n` is 1750, 1744, and 1733. The own-rule means are in the per-process logs (`OWN-RULE ROC-AUC summary`). Do not compare those means. The fair means are the matched table above.

DXMulE seed 2025 own-rule summary is `mean=0.719092 std_ddof0=0.110779 min=0.494048 max=1.000000 n=1750`, the same printed summary as Phase-1. That is a reproduction of the same seed under the old inclusion rule, not a second finding, and it is not the matched number (`n=1745`, mean `0.719087`).

## ACC@p and Fidelity±@p (HYPOTHESIS)

Full test set. `n=3600` on every cell, including constant-label graphs left out of ROC-AUC. Definition, same text as the logs:

> ACC@p is the fraction of graphs whose frozen-GNN argmax on the hard top-p subgraph equals the unmasked argmax. Fidelity+@p (necessity) is the mean drop in softmax probability of that unmasked class when the hard top-p edges are removed. Fidelity-@p (sufficiency) is the mean drop when only the hard top-p edges are kept. Hard top-p keeps ceil(p * n_edges) edges, at least 1 if the graph has edges, ranked by the shared explanation score with a stable argsort, per graph. This is `dxmule.scales.edges_to_keep`. GCNConv self-loops are not part of the stored `edge_index`; the explain hook leaves those loop messages at weight 1. Ground-truth edge labels are not used. IDEA and DXMulE training objectives are different and are not fidelity.

How to read a cell, without turning it into a ranking: higher ACC@p means the kept subgraph more often preserves the unmasked argmax. Higher Fidelity+@p means removing the top-p edges drops the original class probability more. Lower Fidelity-@p means the kept subgraph preserves more of that probability.

### IDEA, logged cells

| p | seed 2025 ACC | Fid+ | Fid- | seed 2026 ACC | Fid+ | Fid- | seed 2027 ACC | Fid+ | Fid- |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.10 | 0.547222 | 0.050139 | 0.387350 | 0.551944 | 0.046937 | 0.385002 | 0.548889 | 0.059677 | 0.384369 |
| 0.30 | 0.488611 | 0.131257 | 0.419325 | 0.483056 | 0.115912 | 0.418266 | 0.499444 | 0.146727 | 0.411053 |
| 0.50 | 0.462222 | 0.256689 | 0.427649 | 0.462500 | 0.238846 | 0.430674 | 0.463889 | 0.278771 | 0.428842 |
| 0.70 | 0.489167 | 0.453117 | 0.407518 | 0.491667 | 0.441659 | 0.403975 | 0.487778 | 0.462937 | 0.415565 |
| 0.90 | 0.638889 | 0.389446 | 0.281549 | 0.609167 | 0.393929 | 0.306770 | 0.678333 | 0.386742 | 0.256164 |

### DXMulE, logged cells

| p | seed 2025 ACC | Fid+ | Fid- | seed 2026 ACC | Fid+ | Fid- | seed 2027 ACC | Fid+ | Fid- |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.10 | 0.552778 | 0.342024 | 0.400408 | 0.551667 | 0.341709 | 0.400016 | 0.550556 | 0.342769 | 0.399283 |
| 0.30 | 0.614167 | 0.428658 | 0.321464 | 0.617500 | 0.428432 | 0.320022 | 0.618611 | 0.428822 | 0.320923 |
| 0.50 | 0.757222 | 0.412226 | 0.190124 | 0.760278 | 0.411645 | 0.188019 | 0.755278 | 0.411022 | 0.191530 |
| 0.70 | 0.853056 | 0.379671 | 0.098220 | 0.855000 | 0.380189 | 0.096917 | 0.855000 | 0.380368 | 0.097053 |
| 0.90 | 0.924444 | 0.385782 | 0.040272 | 0.925833 | 0.386145 | 0.039237 | 0.927500 | 0.384975 | 0.038559 |

Seed 2025 DXMulE fidelity matches the Phase-1 printed cells. That is the same seed reproduced, not a new measurement protocol.

### Derived across-seed summary of those cells

Unweighted mean and sample standard deviation (`ddof=1`) of the three logged six-decimal cells. Computed from those cells when this file was written. Not a new forward pass and not a significance test.

| p | IDEA ACC | IDEA Fid+ | IDEA Fid- | DXMulE ACC | DXMulE Fid+ | DXMulE Fid- |
| --- | --- | --- | --- | --- | --- | --- |
| 0.10 | 0.549352 ± 0.002395 | 0.052251 ± 0.006627 | 0.385574 ± 0.001571 | 0.551667 ± 0.001111 | 0.342167 ± 0.000544 | 0.399902 ± 0.000571 |
| 0.30 | 0.490370 ± 0.008334 | 0.131299 ± 0.015408 | 0.416215 ± 0.004501 | 0.616759 ± 0.002313 | 0.428637 ± 0.000196 | 0.320803 ± 0.000728 |
| 0.50 | 0.462870 ± 0.000893 | 0.258102 ± 0.020000 | 0.429055 ± 0.001524 | 0.757593 ± 0.002521 | 0.411631 ± 0.000602 | 0.189891 ± 0.001767 |
| 0.70 | 0.489537 ± 0.001971 | 0.452571 ± 0.010650 | 0.409019 ± 0.005939 | 0.854352 ± 0.001122 | 0.380076 ± 0.000362 | 0.097397 ± 0.000716 |
| 0.90 | 0.642130 ± 0.034697 | 0.390039 ± 0.003630 | 0.281494 ± 0.025303 | 0.925926 ± 0.001530 | 0.385634 ± 0.000599 | 0.039356 ± 0.000863 |

## DXMulE hard top-k vs ground-truth edges (HYPOTHESIS)

Nested hard masks, one shared ranking. Precision and recall are overlap with `edge_mask`, not fidelity. Realized retention is `ceil(k * n_edges) / n_edges`, averaged over graphs. It does not depend on the score values, and it is the same on every seed. Recall is the mean over the 1838 graphs with at least one positive ground-truth edge. Precision and retention use all 3600 test graphs. IDEA does not log this table.

| k | realized retention | precision 2025 | recall 2025 | precision 2026 | recall 2026 | precision 2027 | recall 2027 | n_retention | n_precision | n_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.10 | 0.110137 | 0.389861 | 0.320366 | 0.389468 | 0.320049 | 0.389481 | 0.320094 | 3600 | 3600 | 1838 |
| 0.30 | 0.309893 | 0.346331 | 0.797969 | 0.346646 | 0.798830 | 0.346889 | 0.799510 | 3600 | 3600 | 1838 |
| 0.50 | 0.500000 | 0.256067 | 0.952031 | 0.256061 | 0.952077 | 0.256090 | 0.952167 | 3600 | 3600 | 1838 |
| 0.70 | 0.709659 | 0.188928 | 0.995466 | 0.188938 | 0.995511 | 0.188936 | 0.995511 | 3600 | 3600 | 1838 |
| 0.90 | 0.909414 | 0.148107 | 0.999683 | 0.148107 | 0.999683 | 0.148107 | 0.999683 | 3600 | 3600 | 1838 |

Seed 2025 matches the Phase-1 hard-retention table at the printed precision.

## Times

`Train loop seconds` is `time.perf_counter` around the epoch loop (IDEA's figure includes the per-batch unmasked embedding). `Evaluate seconds` is the test pass. `ELAPSED_SEC` is wall clock around the process, including unpickling.

| seed | IDEA train | IDEA eval | IDEA wall | DXMulE train | DXMulE eval | DXMulE wall |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | 92.117 | 49.976 | 148.196 | 481.547 | 50.818 | 538.972 |
| 2026 | 92.979 | 48.855 | 148.224 | 495.082 | 51.137 | 552.456 |
| 2027 | 94.237 | 50.385 | 151.350 | 453.047 | 46.776 | 936.608 |

DXMulE seed 2027: the logging clock jumps from `07:57:28` (target-recomputation line) to `08:05:02` (epoch 1). Epochs 2–20 are about 22 seconds apart on that clock. `Train loop seconds` is 453.047. `ELAPSED_SEC` is 936.608 and includes the gap. Use 453.047 as the training-loop figure for that seed. The gap lines up with a guest `virtual machine fork` note in `dmesg` at `08:04:53`. It is not a property of the explainer.

These are CPU times on 4 threads. They are not a speed result against a GPU run. No GPU was present. Phase-1 DXMulE seed 2025 reported train-loop seconds 421.742 on another CPU session; this session recorded 481.547 for the same seed. The metric lines reproduced. The seconds did not.

## Training losses (not comparable)

IDEA's logged objective is the single-scale `RV_model` total (`rqvae_loss`). The line labeled "Codebook Loss" is the reconstruction term returned by `Idea.train`, not a separate codebook penalty. Distribution is the Sinkhorn term.

DXMulE's `loss` is `rv + 0.1 * cross + 0.1 * complement`. Seed 2025 epoch 1: `0.7458 + 0.1 * 0.1641 + 0.1 * 0.4787 = 0.8101` at the log's four-decimal rounding. Seed 2025 epoch 20: `0.4326 + 0.1 * 0.2910 + 0.1 * 0.6104 = 0.5227`. The same arithmetic is why a DXMulE total is not an IDEA total.

Concrete temperature is `5.0 * (2.0 / 5.0) ** (epoch / epochs)` on both paths, so an epoch trace mixes that schedule with the parameter update.

IDEA totals below are copied from the three IDEA logs. Columns are RQVAE total, reconstruction (logged as Codebook Loss), and distribution.

| epoch | 2025 total | 2025 recon | 2025 dist | 2026 total | 2026 recon | 2026 dist | 2027 total | 2027 recon | 2027 dist |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.6811 | 0.2771 | 0.0013 | 0.7174 | 0.2933 | 0.0014 | 0.7181 | 0.2932 | 0.0015 |
| 2 | 0.4298 | 0.1681 | 0.0009 | 0.4370 | 0.1705 | 0.0009 | 0.4332 | 0.1697 | 0.0009 |
| 3 | 0.4146 | 0.1608 | 0.0010 | 0.4136 | 0.1610 | 0.0009 | 0.4126 | 0.1602 | 0.0009 |
| 4 | 0.3874 | 0.1490 | 0.0011 | 0.3934 | 0.1516 | 0.0011 | 0.3898 | 0.1500 | 0.0011 |
| 5 | 0.3669 | 0.1394 | 0.0014 | 0.3788 | 0.1439 | 0.0012 | 0.3673 | 0.1398 | 0.0013 |
| 6 | 0.3625 | 0.1374 | 0.0013 | 0.3638 | 0.1380 | 0.0014 | 0.3580 | 0.1360 | 0.0014 |
| 7 | 0.3570 | 0.1352 | 0.0013 | 0.3595 | 0.1360 | 0.0013 | 0.3581 | 0.1355 | 0.0012 |
| 8 | 0.3491 | 0.1323 | 0.0013 | 0.3558 | 0.1346 | 0.0012 | 0.3566 | 0.1344 | 0.0012 |
| 9 | 0.3480 | 0.1311 | 0.0012 | 0.3485 | 0.1316 | 0.0012 | 0.3487 | 0.1317 | 0.0012 |
| 10 | 0.3457 | 0.1294 | 0.0010 | 0.3416 | 0.1286 | 0.0011 | 0.3388 | 0.1274 | 0.0010 |
| 11 | 0.3298 | 0.1230 | 0.0009 | 0.3300 | 0.1236 | 0.0009 | 0.3305 | 0.1235 | 0.0008 |
| 12 | 0.2892 | 0.1050 | 0.0006 | 0.3054 | 0.1126 | 0.0006 | 0.2739 | 0.0985 | 0.0005 |
| 13 | 0.2410 | 0.0836 | 0.0004 | 0.2370 | 0.0822 | 0.0004 | 0.2342 | 0.0809 | 0.0004 |
| 14 | 0.2324 | 0.0806 | 0.0004 | 0.2340 | 0.0806 | 0.0004 | 0.2371 | 0.0820 | 0.0004 |
| 15 | 0.2320 | 0.0801 | 0.0004 | 0.2294 | 0.0793 | 0.0004 | 0.2364 | 0.0817 | 0.0004 |
| 16 | 0.2357 | 0.0817 | 0.0004 | 0.2360 | 0.0815 | 0.0004 | 0.2343 | 0.0808 | 0.0004 |
| 17 | 0.2341 | 0.0805 | 0.0004 | 0.2297 | 0.0796 | 0.0004 | 0.2267 | 0.0783 | 0.0004 |
| 18 | 0.2340 | 0.0808 | 0.0004 | 0.2391 | 0.0828 | 0.0004 | 0.2400 | 0.0833 | 0.0004 |
| 19 | 0.2304 | 0.0795 | 0.0004 | 0.2314 | 0.0797 | 0.0004 | 0.2317 | 0.0800 | 0.0004 |
| 20 | 0.2359 | 0.0810 | 0.0004 | 0.2340 | 0.0809 | 0.0004 | 0.2339 | 0.0809 | 0.0003 |

DXMulE totals (`loss`) and the `rv` term. Full component lines (dist, recon, cross, top-down KL, reverse KL, shell, persistence, complement) stay in `dxmule_benzene_s*.log`. Seed 2025's component trace matches the Phase-1 table at four decimals. All logged totals were finite. The script raises if the averaged total is non-finite. It did not raise.

| epoch | 2025 total | 2025 rv | 2026 total | 2026 rv | 2027 total | 2027 rv |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8101 | 0.7458 | 0.8260 | 0.7623 | 0.8273 | 0.7634 |
| 2 | 0.6301 | 0.5539 | 0.6378 | 0.5615 | 0.6371 | 0.5620 |
| 3 | 0.5797 | 0.4965 | 0.5760 | 0.4915 | 0.5808 | 0.4989 |
| 4 | 0.5526 | 0.4655 | 0.5490 | 0.4622 | 0.5532 | 0.4663 |
| 5 | 0.5439 | 0.4547 | 0.5460 | 0.4573 | 0.5417 | 0.4519 |
| 6 | 0.5402 | 0.4508 | 0.5373 | 0.4474 | 0.5344 | 0.4441 |
| 7 | 0.5329 | 0.4427 | 0.5344 | 0.4443 | 0.5347 | 0.4443 |
| 8 | 0.5276 | 0.4372 | 0.5337 | 0.4432 | 0.5320 | 0.4417 |
| 9 | 0.5274 | 0.4376 | 0.5265 | 0.4367 | 0.5305 | 0.4399 |
| 10 | 0.5336 | 0.4430 | 0.5278 | 0.4371 | 0.5258 | 0.4348 |
| 11 | 0.5244 | 0.4341 | 0.5249 | 0.4350 | 0.5290 | 0.4381 |
| 12 | 0.5260 | 0.4360 | 0.5265 | 0.4360 | 0.5249 | 0.4352 |
| 13 | 0.5275 | 0.4378 | 0.5247 | 0.4341 | 0.5216 | 0.4311 |
| 14 | 0.5203 | 0.4297 | 0.5245 | 0.4335 | 0.5265 | 0.4363 |
| 15 | 0.5221 | 0.4318 | 0.5192 | 0.4283 | 0.5239 | 0.4338 |
| 16 | 0.5244 | 0.4340 | 0.5231 | 0.4336 | 0.5265 | 0.4361 |
| 17 | 0.5239 | 0.4338 | 0.5184 | 0.4281 | 0.5158 | 0.4253 |
| 18 | 0.5227 | 0.4317 | 0.5291 | 0.4387 | 0.5277 | 0.4374 |
| 19 | 0.5195 | 0.4296 | 0.5224 | 0.4320 | 0.5208 | 0.4310 |
| 20 | 0.5227 | 0.4326 | 0.5243 | 0.4342 | 0.5216 | 0.4318 |

## Limitations still in the scripts

1. **GNN freezing.** DXMulE sets `requires_grad_(False)` on the loaded GNN. IDEA does not. Neither optimizer steps GNN parameters. Forward weights come from `GNN/Benzene.pth`. Both calls that build the embedding target run the GNN in eval mode.
2. **Self-loops.** `GCNConv` adds self-loops that are not rows of `edge_index`. Fidelity does not mask them. `mask_sensitivity=True` only shows that zeroing the stored edges changes logits.
3. **CPU, three seeds, default loss weights.** Nothing was tuned. No interval and no hypothesis test was run. There is no GPU result in this folder.
4. **Own-rule AUC** is still logged, and its `n` still differs between methods. The fair number is the matched table.
5. **Matched `n` differs by seed** (1745, 1736, 1729). The across-seed mean does not weight by `n` and does not pool graphs.
6. **Seed 2027 wall clock** includes the gap described under Times. Training-loop seconds for that seed are 453.047.
7. **README example** uses `--epochs 10`, `--lr 0.0005`, and `--batch_size 64`. This run used 20, 0.001, and 32, passed explicitly.

## Out of scope

Other datasets. MSExplainer, GNNExplainer, PGExplainer, and a C2 ablation matrix. Ranking the two methods. Promoting these numbers to method facts. Merging this branch on the strength of the tables.
