"""Matched-subset ROC-AUC for IDEA and DXMulE.

Each explainer writes one record per test graph. This module applies a
single skip rule to both records so the two means use the same graph
indices. It does not rank the methods and it does not test a hypothesis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

MATCHED_RULE = (
    "Matched-subset ROC-AUC uses one shared skip rule for both methods. "
    "A test graph is excluded for BOTH methods when any of the following holds: "
    "the ground-truth edge_mask is constant on that graph; "
    "IDEA explanation scores are constant on that graph; "
    "DXMulE explanation scores are constant on that graph; "
    "or either score vector is non-finite or empty. "
    "Explanation scores are the sigmoid edge masks from each explainer "
    "(strictly increasing in that method's edge logit). "
    "Constant means numpy.allclose of every score to the first score. "
    "The kept indices are the intersection, so n is the same for both means. "
    "ACC@p and Fidelity+/-@p stay on the full test loader and are not filtered here."
)


def classify_graph_auc(labels, scores) -> dict[str, Any]:
    """Flags and ROC-AUC for one graph.

    ``auc`` is set only when labels vary, scores vary, and scores are finite.
    A constant-score graph therefore has ``auc is None`` even if labels vary.
    """
    label_arr = np.asarray(labels).reshape(-1)
    score_arr = np.asarray(scores, dtype=float).reshape(-1)
    if label_arr.size != score_arr.size:
        raise ValueError(
            f"label length {label_arr.size} != score length {score_arr.size}"
        )
    n_edges = int(label_arr.size)
    if n_edges == 0:
        return {
            "constant_label": True,
            "constant_score": True,
            "score_finite": False,
            "auc": None,
            "n_edges": 0,
        }
    label_int = label_arr.astype(int)
    score_finite = bool(np.isfinite(score_arr).all())
    constant_label = bool(label_int.min() == label_int.max())
    constant_score = bool(score_finite and np.allclose(score_arr, score_arr[0]))
    auc = None
    if score_finite and not constant_label and not constant_score:
        auc = float(roc_auc_score(label_int, score_arr))
    return {
        "constant_label": constant_label,
        "constant_score": constant_score,
        "score_finite": score_finite,
        "auc": auc,
        "n_edges": n_edges,
    }


def write_auc_record(path: str, payload: dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def _auc_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "std_ddof0": None, "min": None, "max": None, "n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std_ddof0": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "nan"
    number = float(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.6f}"


def build_matched(
    idea: dict[str, Any],
    dxmule: dict[str, Any],
    *,
    expect_n_test: int | None = None,
    require_full_split: bool = False,
) -> dict[str, Any]:
    """Intersection AUC. Raises if the two records are not the same split."""
    if idea.get("method") != "idea":
        raise ValueError(f"expected method idea, got {idea.get('method')}")
    if dxmule.get("method") != "dxmule":
        raise ValueError(f"expected method dxmule, got {dxmule.get('method')}")
    if idea.get("dataset") != dxmule.get("dataset"):
        raise ValueError(
            f"dataset mismatch {idea.get('dataset')} vs {dxmule.get('dataset')}"
        )
    if int(idea["seed"]) != int(dxmule["seed"]):
        raise ValueError(f"seed mismatch {idea['seed']} vs {dxmule['seed']}")
    if require_full_split:
        for name, payload in (("idea", idea), ("dxmule", dxmule)):
            max_graphs = payload.get("protocol", {}).get("max_graphs", None)
            if max_graphs is not None:
                raise ValueError(f"{name} record used max_graphs={max_graphs}")

    idea_graphs = list(idea["graphs"])
    dx_graphs = list(dxmule["graphs"])
    if len(idea_graphs) != len(dx_graphs):
        raise ValueError(
            f"graph count {len(idea_graphs)} != {len(dx_graphs)}"
        )
    if expect_n_test is not None and len(idea_graphs) != int(expect_n_test):
        raise ValueError(
            f"expected {expect_n_test} test graphs, found {len(idea_graphs)}"
        )

    skips = {
        "constant_label": 0,
        "idea_only_constant_score": 0,
        "dxmule_only_constant_score": 0,
        "both_constant_score": 0,
        "nonfinite_or_empty": 0,
    }
    idea_aucs: list[float] = []
    dx_aucs: list[float] = []
    for position, (left, right) in enumerate(zip(idea_graphs, dx_graphs)):
        if int(left["index"]) != position or int(right["index"]) != position:
            raise ValueError(f"AUC records are not aligned at position {position}")
        if bool(left["constant_label"]) != bool(right["constant_label"]):
            raise ValueError(
                f"ground-truth constant-label flag disagrees at graph {position}"
            )
        if int(left["n_edges"]) != int(right["n_edges"]):
            raise ValueError(f"edge count disagrees at graph {position}")
        if left["constant_label"]:
            skips["constant_label"] += 1
            continue
        left_bad = (not bool(left["score_finite"])) or int(left["n_edges"]) == 0
        right_bad = (not bool(right["score_finite"])) or int(right["n_edges"]) == 0
        if left_bad or right_bad:
            skips["nonfinite_or_empty"] += 1
            continue
        left_constant = bool(left["constant_score"])
        right_constant = bool(right["constant_score"])
        if left_constant or right_constant:
            if left_constant and right_constant:
                skips["both_constant_score"] += 1
            elif left_constant:
                skips["idea_only_constant_score"] += 1
            else:
                skips["dxmule_only_constant_score"] += 1
            continue
        if left["auc"] is None or right["auc"] is None:
            raise ValueError(
                f"graph {position} passed the skip rule but auc is missing"
            )
        left_auc = float(left["auc"])
        right_auc = float(right["auc"])
        if not (math.isfinite(left_auc) and math.isfinite(right_auc)):
            skips["nonfinite_or_empty"] += 1
            continue
        idea_aucs.append(left_auc)
        dx_aucs.append(right_auc)

    idea_summary = _auc_summary(idea_aucs)
    dx_summary = _auc_summary(dx_aucs)
    if idea_summary["n"] != dx_summary["n"]:
        raise RuntimeError("matched n diverged between methods")
    accounted = idea_summary["n"] + sum(skips.values())
    if accounted != len(idea_graphs):
        raise RuntimeError(
            f"skip buckets account for {accounted} graphs, not {len(idea_graphs)}"
        )
    # Own-rule counts describe the Phase-1 inclusion gap. They are not the fair n.
    idea_own_n = (
        idea_summary["n"]
        + skips["idea_only_constant_score"]
        + skips["dxmule_only_constant_score"]
        + skips["both_constant_score"]
    )
    dxmule_own_n = idea_summary["n"] + skips["idea_only_constant_score"]
    return {
        "rule": MATCHED_RULE,
        "dataset": idea["dataset"],
        "seed": int(idea["seed"]),
        "n_test": len(idea_graphs),
        "n": idea_summary["n"],
        "equal_n": True,
        "idea": idea_summary,
        "dxmule": dx_summary,
        "skips": skips,
        "inclusion": {
            "idea_own_rule_n": idea_own_n,
            "dxmule_own_rule_n": dxmule_own_n,
            "matched_n": idea_summary["n"],
            "note": (
                "idea_own_rule_n scores every mixed-label graph with finite scores, "
                "including graphs whose IDEA scores are constant. "
                "dxmule_own_rule_n also drops graphs whose DXMulE scores are constant. "
                "Those two counts can differ. The fair means use matched_n only."
            ),
        },
    }


def summarize_matched_seeds(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Unweighted mean of per-seed matched means. Not a significance test."""
    if not payloads:
        raise ValueError("no matched records to summarize")
    seeds = [int(payload["seed"]) for payload in payloads]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds in summary: {seeds}")

    def _across(means: list[Any]) -> dict[str, Any]:
        if any(mean is None or not math.isfinite(float(mean)) for mean in means):
            return {"mean": None, "std_ddof1": None, "n_seeds": len(means)}
        arr = np.asarray([float(mean) for mean in means], dtype=float)
        std = None if arr.size < 2 else float(arr.std(ddof=1))
        return {"mean": float(arr.mean()), "std_ddof1": std, "n_seeds": int(arr.size)}

    idea_means = [payload["idea"]["mean"] for payload in payloads]
    dx_means = [payload["dxmule"]["mean"] for payload in payloads]
    return {
        "aggregation": "unweighted_mean_of_per_seed_means",
        "n_seeds": len(payloads),
        "seeds": seeds,
        "per_seed_n": [int(payload["n"]) for payload in payloads],
        "per_seed_idea_mean": idea_means,
        "per_seed_dxmule_mean": dx_means,
        "idea": _across(idea_means),
        "dxmule": _across(dx_means),
        "note": (
            "Across-seed std is the sample standard deviation (ddof=1) of the "
            "per-seed means. It is omitted when only one seed is present. "
            "Seeds are not pooled into one graph multiset. "
            "This is not a significance test and it is not a claim that one method is better."
        ),
    }


def format_matched_lines(result: dict[str, Any]) -> list[str]:
    idea = result["idea"]
    dxmule = result["dxmule"]
    skips = result["skips"]
    inclusion = result["inclusion"]
    return [
        f"MATCHED_RULE {result['rule']}",
        (
            "MATCHED seed={seed} n={n} "
            "idea_mean={idea_mean} idea_std_ddof0={idea_std} "
            "idea_min={idea_min} idea_max={idea_max} "
            "dxmule_mean={dx_mean} dxmule_std_ddof0={dx_std} "
            "dxmule_min={dx_min} dxmule_max={dx_max}"
        ).format(
            seed=result["seed"],
            n=result["n"],
            idea_mean=_fmt(idea["mean"]),
            idea_std=_fmt(idea["std_ddof0"]),
            idea_min=_fmt(idea["min"]),
            idea_max=_fmt(idea["max"]),
            dx_mean=_fmt(dxmule["mean"]),
            dx_std=_fmt(dxmule["std_ddof0"]),
            dx_min=_fmt(dxmule["min"]),
            dx_max=_fmt(dxmule["max"]),
        ),
        (
            "MATCHED_SKIP n_test={n_test} constant_label={constant_label} "
            "idea_only_constant_score={idea_only} "
            "dxmule_only_constant_score={dx_only} "
            "both_constant_score={both} nonfinite_or_empty={nonfinite} matched={n}"
        ).format(
            n_test=result["n_test"],
            constant_label=skips["constant_label"],
            idea_only=skips["idea_only_constant_score"],
            dx_only=skips["dxmule_only_constant_score"],
            both=skips["both_constant_score"],
            nonfinite=skips["nonfinite_or_empty"],
            n=result["n"],
        ),
        (
            "MATCHED_EQUAL_N idea={idea_n} dxmule={dx_n}"
        ).format(idea_n=idea["n"], dx_n=dxmule["n"]),
        (
            "MATCHED_INCLUSION idea_own_rule_n={idea_own} "
            "dxmule_own_rule_n={dx_own} matched_n={matched}"
        ).format(
            idea_own=inclusion["idea_own_rule_n"],
            dx_own=inclusion["dxmule_own_rule_n"],
            matched=inclusion["matched_n"],
        ),
    ]


def format_summary_lines(summary: dict[str, Any]) -> list[str]:
    seeds = ",".join(str(seed) for seed in summary["seeds"])
    per_n = ",".join(
        f"{seed}:{n}" for seed, n in zip(summary["seeds"], summary["per_seed_n"])
    )
    per_idea = ",".join(
        f"{seed}:{_fmt(mean)}"
        for seed, mean in zip(summary["seeds"], summary["per_seed_idea_mean"])
    )
    per_dx = ",".join(
        f"{seed}:{_fmt(mean)}"
        for seed, mean in zip(summary["seeds"], summary["per_seed_dxmule_mean"])
    )
    std_note = "NA" if summary["n_seeds"] < 2 else "ddof1"
    return [
        (
            "MATCHED_ACROSS n_seeds={n_seeds} seeds={seeds} "
            "aggregation={aggregation} std={std_note}"
        ).format(
            n_seeds=summary["n_seeds"],
            seeds=seeds,
            aggregation=summary["aggregation"],
            std_note=std_note,
        ),
        (
            "MATCHED_ACROSS idea_mean={idea_mean} idea_std_ddof1={idea_std} "
            "dxmule_mean={dx_mean} dxmule_std_ddof1={dx_std}"
        ).format(
            idea_mean=_fmt(summary["idea"]["mean"]),
            idea_std=_fmt(summary["idea"]["std_ddof1"]),
            dx_mean=_fmt(summary["dxmule"]["mean"]),
            dx_std=_fmt(summary["dxmule"]["std_ddof1"]),
        ),
        f"MATCHED_ACROSS per_seed_n={per_n}",
        f"MATCHED_ACROSS per_seed_idea_mean={per_idea}",
        f"MATCHED_ACROSS per_seed_dxmule_mean={per_dx}",
        f"MATCHED_ACROSS note={summary['note']}",
    ]


def _write_stdout(lines: list[str]) -> None:
    for line in lines:
        print(line)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Matched-subset ROC-AUC")
    parser.add_argument("--idea", type=str, default="", help="IDEA per-graph JSON")
    parser.add_argument("--dxmule", type=str, default="", help="DXMulE per-graph JSON")
    parser.add_argument("--out", type=str, default="", help="Write the matched JSON here")
    parser.add_argument(
        "--require-full-split",
        action="store_true",
        help="Refuse records produced with --max_graphs",
    )
    parser.add_argument(
        "--expect-n-test",
        type=int,
        default=None,
        help="Require this many test graphs in both records",
    )
    parser.add_argument(
        "--summary",
        nargs="*",
        default=None,
        help="Matched JSON files to aggregate across seeds",
    )
    parser.add_argument("--summary-out", type=str, default="")
    args = parser.parse_args(argv)

    if args.idea or args.dxmule:
        if not args.idea or not args.dxmule:
            raise SystemExit("--idea and --dxmule must be passed together")
        result = build_matched(
            load_json(args.idea),
            load_json(args.dxmule),
            expect_n_test=args.expect_n_test,
            require_full_split=args.require_full_split,
        )
        if args.out:
            write_auc_record(args.out, result)
        _write_stdout(format_matched_lines(result))

    if args.summary is not None:
        if not args.summary:
            raise SystemExit("--summary needs at least one matched JSON")
        summary = summarize_matched_seeds([load_json(path) for path in args.summary])
        if args.summary_out:
            write_auc_record(args.summary_out, summary)
        _write_stdout(format_summary_lines(summary))


if __name__ == "__main__":
    main()
