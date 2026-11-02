Paradigm Shift of GNN Explainer from Label Space to Prototypical Representation Space. ICLR 2026, Poster.
<div align="center">
  <img width="678" height="574" alt="image" src="https://github.com/user-attachments/assets/86bad68a-ca6e-47fc-a5c5-dba729d7bf8b" />
</div>

- --

### Environment
The environment has been exported to `IDEA/requirements.txt`.

### Dataset
Regrading the four real-wolrd datasets, i.e., Mutag, Alkane, Fluoride, Benzene, the original resource is available at [Harvard Dataverse](https://doi.org/10.7910/DVN/KULOS8), which is released by [_Evaluating explainability for graph neural networks_](https://www.nature.com/articles/s41597-023-01974-x).

Furthermore, the processed datasets alongside the ground explanantions are uploaled in `IDEA/data/`.

### Model
- The architecture of the target GNN to be explained in defined in `IDEA/gnn_model.py`. In `IDEA/GNN`, we upload the weights of trained GNN models.
- The RQ-VAE module is implemented in `IDEA/RQ-VAE.py`. Afterwards, the hierarchical graph tokenizer (HGTokenizer) is implemented in `IDEA/graph_tokenizer.py`.
- The IDEA explainer with PGExplainer as backbone is implemented in `IDEA/explainer/idea.py`. In addition, we upload the weights of pre-trained codebooks, including both shallow and deep branches, in `IDEA/codebook`.

### Layout of this fork
The project root is this repository. There is no nested `IDEA/` directory. Launch commands from the repo root with `--root_path ./`. Upstream instructions that prefix paths with `IDEA/` describe the original layout. `requirements.txt` pins the upstream CUDA wheels (`torch==2.6.0+cu126`). A CPU machine can install the matching CPU wheel instead (`pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu`, then `torch-geometric==2.6.1` and `geomloss`) and pass `--device -1`.

Benzene and Fluoride ship as zip archives. `explainer/eval_dxmule.py` unzips `data/benzene_with_gt.pkl.zip` (and the Fluoride archive) into `data/` when the pickle is absent. The original `explainer/eval_idea.py` expects the pickle to exist already.

### Evaluation
`explainer/eval_idea.py` is the original single-scale IDEA path. From this fork's root:

```bash
python explainer/eval_idea.py --dataset Benzene --root_path ./ --epochs 10 --lr 0.0005 --device -1 --batch_size 64 --codebook_size 32 --gt_learning_rate 0.01 --gt_epochs 10
```

The tokenizer epoch flag on that script is `--gt_epochs`.

### DXMulE multi-scale
DXMulE keeps IDEA's prototype codebooks, the HGTokenizer / RQ-VAE training that produces them, and the Wasserstein alignment in `RV_model`. It adds one scale axis: MSExplainer-style edge-retention ratios `{k_m}`.

`{k_m}` is applied to a single shared edge-score vector (one MLP, concrete samples during training, raw logits at explanation time). Ratio `k` keeps the top fraction `k` of edges inside each graph. Hard masks are nested because every ratio reads the same stable ranking. Default ratios are `0.1,0.3,0.5,0.7,0.9`.

Each masked subgraph is pooled and passed through the same `RV_model` call IDEA uses, then those losses are averaged. The cross-scale term (C2) is separate: a global row-stochastic map `W` on the explanatory codebook sends the fine-scale prototype assignment toward the next coarser scale (`p_fine @ W ≈ p_coarse`), with a top-down KL, a reverse KL, and a shell term on the embedding gap between the two scales. An auxiliary term matches complement subgraphs to the base codebook. Weights default to `0.1` and are not a tuned result.

```bash
python explainer/eval_dxmule.py --dataset Benzene --root_path ./ --device -1 \
    --epochs 1 --batch_size 4 --max_graphs 8 \
    --scales 0.1,0.3,0.5,0.7,0.9
```

`--max_graphs` truncates the split so the path can be smoke-tested. The script logs ROC-AUC of the shared edge score, and precision/recall of each hard top-k mask against the ground-truth edges. Soft masks are monotone in that shared score, so they do not get a separate AUC. Those logs are a sanity check. Both entry points also log ACC@p and Fidelity+/-@p from `dxmule/fidelity.py` on a hard top-p grid. A logged number is a run record. It is not a gate approval.

Core checks that do not load the GNN:

```bash
python -m dxmule.test_core
```

**Implemented**

- Retention ratios `{k_m}` on one shared score, per graph, hard masks nested.
- Per-scale Wasserstein alignment with the upstream `RV_model` and the released codebooks (or the upstream tokenizer if `--use_ckpt 0`).
- C2 global prototype map across adjacent scales, including the shell residual term.
- Auxiliary complement-to-base-codebook alignment.
- Original `explainer/eval_idea.py` left runnable as the single-scale path.

**Hypothesis / TODO**

- Whether C2, the shell, or the complement term improves ROC-AUC or fidelity over single-scale IDEA. A full-split Benzene run, when present, is recorded in `dxmule/FORMAL_COMPARE_Benzene.md` as HYPOTHESIS. It does not establish that one method is better. Phase-2 (`dxmule/run_formal_benzene_phase2.sh`, report `dxmule/FORMAL_COMPARE_Benzene_Phase2.md`) recomputes IDEA's unmasked embedding target on the current batch and scores ROC-AUC on the intersection of graphs whose ground-truth edge mask and both explanation scores are non-constant. Those numbers are also HYPOTHESIS. The two training losses are different objectives and are not comparable.
- Fidelity formulas other than the ACC@p / Fidelity+/-@p protocol in `dxmule/fidelity.py`. Complement-deletion curves beyond Fidelity+@p are not a separate sweep.
- Multi-scale quantization inside HGTokenizer. Codebooks are still learned on the full graph.
- Replacing the transport KL with a Sinkhorn loss on the same assignments.
- Choosing `{k_m}` per graph, node-level retention, and directed or heterogeneous graphs.
- MSExplainer's label-space contrastive objective. DXMulE stays in IDEA's prototype space; that label-space loss is a separate experiment.

- --
```bibtex
@inproceedings{idea2026,
  title     = {{Paradigm Shift of GNN Explainer from Label Space to Prototypical Representation Space}},
  author    = {Yin, Jun and Wang, Senzhang and Luo, Ziluowen and Huo, Peng and Yan, Hao and Miao, Hao and Li, Chaozhuo and Pan, Shirui and Zhang, Chengqi},
  booktitle = {International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=X7eYISNf01}
}
```
