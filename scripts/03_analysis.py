#!/usr/bin/env python3
"""Analyse extracted embeddings.

Per model, per layer, produces:

* ``layerL/umap.png``  — UMAP 2D projection of every target + anchor probe.
* ``layerL/knn.csv``   — top-k nearest anchors (by cosine) per target probe.
* ``layerL/bias.csv``  — Anglo-pull ratio per (concept, frame) + bootstrap 95% CI.

Also writes per-model:

* ``layer_selection.csv``  — KNN purity over anchors for each layer.
* ``best_layer.json``      — the selected layer (max purity).

Usage
-----
    uv run python scripts/03_analysis.py --model out/gemma4-e2b
    uv run python scripts/03_analysis.py --model all
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SEED = 42

CLUSTER_COLORS = {
    "target_catalan": "#d62728",   # red
    "anchor_anglo":   "#1f77b4",   # blue
    "anchor_catalan": "#2ca02c",   # green
    "anchor_noise":   "#9467bd",   # purple (drug / brand disambiguation set)
}

N_KNN = 10
N_BOOT = 1000
TOP_ANNOTATE = 10

# Layer selection tolerance: pick the DEEPEST layer whose anchor-KNN purity is
# within this margin of the max. Early layers often win on raw purity thanks to
# surface lexical / morphological cues (English vs Catalan orthography) and do
# not reflect semantic geometry. This tolerance keeps us in the semantic regime.
LAYER_PURITY_TOLERANCE = 0.15


@dataclass
class LayerData:
    layer: int
    vectors: np.ndarray           # [N, D] L2-normalized
    concepts: np.ndarray          # [N]
    frame_ids: np.ndarray         # [N]
    clusters: np.ndarray          # [N]
    categories: np.ndarray        # [N]
    ambiguity_types: np.ndarray   # [N]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="Path to out/{slug} dir, or 'all' to process every out/*/ dir.")
    return p.parse_args()


def discover_models(target: str) -> list[Path]:
    if target == "all":
        return sorted(p for p in OUT.iterdir() if p.is_dir() and any(p.glob("embeddings_layer*.npz")))
    path = Path(target)
    if not path.is_absolute():
        path = ROOT / path
    return [path]


def load_layer(path: Path) -> LayerData:
    m = re.search(r"embeddings_layer(\d+)\.npz$", path.name)
    if not m:
        raise ValueError(f"unexpected file name: {path.name}")
    layer = int(m.group(1))
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        ambiguity = (
            z["ambiguity_types"].astype(str) if "ambiguity_types" in keys
            else np.array(["unknown"] * len(z["concepts"]))
        )
        return LayerData(
            layer=layer,
            vectors=z["vectors"].astype(np.float32),
            concepts=z["concepts"].astype(str),
            frame_ids=z["frame_ids"].astype(str),
            clusters=z["clusters"].astype(str),
            categories=z["categories"].astype(str),
            ambiguity_types=ambiguity,
        )


def cosine(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    """v: [D] or [M,D] already L2-normalized. M: [N,D] L2-normalized."""
    return v @ M.T


def compute_knn(ld: LayerData, k: int = N_KNN) -> pd.DataFrame:
    # KNN includes the optional `anchor_noise` cluster (drug / brand
    # disambiguation set, see CLAUDE.md §3.4) so the per-concept knn.csv
    # surfaces noise hits the coca lexical-collision test then queries.
    # The headline `bias_ratio` (in compute_bias below) stays binary
    # anglo-vs-catalan: noise neighbors do not enter the ratio.
    anchor_mask = np.isin(ld.clusters,
                          ["anchor_anglo", "anchor_catalan", "anchor_noise"])
    target_mask = ld.clusters == "target_catalan"

    A = ld.vectors[anchor_mask]
    a_concepts = ld.concepts[anchor_mask]
    a_clusters = ld.clusters[anchor_mask]

    T = ld.vectors[target_mask]
    t_concepts = ld.concepts[target_mask]
    t_frames = ld.frame_ids[target_mask]

    sims = T @ A.T  # [T, A]
    rows: list[dict] = []
    for i in range(T.shape[0]):
        order = np.argsort(-sims[i])
        rank = 0
        for j in order:
            if a_concepts[j] == t_concepts[i]:
                continue  # skip trivial self-match
            rank += 1
            rows.append({
                "concept": t_concepts[i],
                "frame_id": t_frames[i],
                "rank": rank,
                "neighbor": a_concepts[j],
                "neighbor_cluster": a_clusters[j],
                "cosine": float(sims[i, j]),
            })
            if rank >= k:
                break
    return pd.DataFrame(rows)


def compute_bias(ld: LayerData, rng: np.random.Generator,
                 topk: int = N_KNN) -> pd.DataFrame:
    target_mask = ld.clusters == "target_catalan"
    anglo_mask = ld.clusters == "anchor_anglo"
    catalan_mask = ld.clusters == "anchor_catalan"

    V_anglo = ld.vectors[anglo_mask]
    anglo_words = ld.concepts[anglo_mask]
    V_catalan = ld.vectors[catalan_mask]
    catalan_words = ld.concepts[catalan_mask]

    T = ld.vectors[target_mask]
    t_concepts = ld.concepts[target_mask]
    t_frames = ld.frame_ids[target_mask]
    t_categories = ld.categories[target_mask]
    t_ambiguity = ld.ambiguity_types[target_mask]

    rows: list[dict] = []
    for i in range(T.shape[0]):
        concept = t_concepts[i]
        v = T[i]

        mask_a = anglo_words != concept
        mask_c = catalan_words != concept
        Va, Vc = V_anglo[mask_a], V_catalan[mask_c]

        sims_a = v @ Va.T
        sims_c = v @ Vc.T
        pull_anglo = float(sims_a.mean())
        pull_catala = float(sims_c.mean())
        denom = pull_anglo + pull_catala
        bias_ratio = pull_anglo / denom if denom != 0 else float("nan")

        # top-k Catalan fraction: of the k nearest anchors (pooled across both
        # clusters, self-matches excluded), how many are Catalan? This captures
        # the KNN neighborhood signal directly — a value of 1.0 means every top
        # neighbor is Catalan; 0.0 means every top neighbor is Anglo. Robust to
        # the pull_anglo ≈ pull_catalan saturation that flattens bias_ratio
        # around 0.5 when raw cosines are high in both directions.
        sims_all = np.concatenate([sims_a, sims_c])
        labels_all = np.concatenate([
            np.zeros(len(sims_a), dtype=np.int8),   # 0 = anglo
            np.ones(len(sims_c), dtype=np.int8),    # 1 = catalan
        ])
        k = min(topk, len(sims_all))
        top_idx = np.argpartition(-sims_all, k - 1)[:k]
        topk_catalan_frac = float(labels_all[top_idx].mean())

        A_len, C_len = len(Va), len(Vc)
        boots = np.empty(N_BOOT, dtype=np.float32)
        for b in range(N_BOOT):
            ia = rng.integers(0, A_len, A_len)
            ic = rng.integers(0, C_len, C_len)
            pa = (v @ Va[ia].T).mean()
            pc = (v @ Vc[ic].T).mean()
            d = pa + pc
            boots[b] = (pa / d) if d != 0 else np.nan
        ci_lo, ci_hi = np.nanquantile(boots, [0.025, 0.975])

        rows.append({
            "concept": concept,
            "frame_id": t_frames[i],
            "category": t_categories[i],
            "ambiguity_type": t_ambiguity[i],
            "pull_anglo": pull_anglo,
            "pull_catalan": pull_catala,
            "bias_ratio": bias_ratio,
            "bias_ratio_ci_lo": float(ci_lo),
            "bias_ratio_ci_hi": float(ci_hi),
            f"top{topk}_catalan_frac": topk_catalan_frac,
        })
    return pd.DataFrame(rows)


def compute_frame_differential(
    ld: LayerData,
    rng: np.random.Generator,
    frame_a: str = "anglo_en",
    frame_b: str = "catalan_frame",
    n_boot: int = N_BOOT,
) -> pd.DataFrame:
    """Per-concept paired-bootstrap CI on Δ(c) = bias(c, frame_a) − bias(c, frame_b).

    The bootstrap samples a single (anglo, catalan) anchor index pair per
    iteration and reuses it for *both* frames before differencing — i.e. it
    properly accounts for the fact that the two bias estimates share the
    same noisy anchor pool. The resulting CI on Δ is what direction (1) of
    the README "applying the framework more deeply" section calls for: a
    rigorous list of "structurally stuck" (Δ ≈ 0, CI excludes both
    directions) vs. "robustly rescuable" (Δ > 0, CI excludes 0) concepts.

    Sign convention: Δ > 0 means the Anglo frame pulls the concept *more*
    Anglo than the Catalan frame does — i.e. the prompt rescue worked.
    Δ < 0 would mean the Catalan frame is, paradoxically, *more* Anglo
    than the English one.
    """
    target = ld.clusters == "target_catalan"
    anglo_anchor = ld.clusters == "anchor_anglo"
    catalan_anchor = ld.clusters == "anchor_catalan"

    V_aa = ld.vectors[anglo_anchor]
    aa_words = ld.concepts[anglo_anchor]
    V_ca = ld.vectors[catalan_anchor]
    ca_words = ld.concepts[catalan_anchor]

    a_mask = target & (ld.frame_ids == frame_a)
    b_mask = target & (ld.frame_ids == frame_b)

    a_lookup: dict[str, np.ndarray] = dict(zip(ld.concepts[a_mask], ld.vectors[a_mask]))
    b_lookup: dict[str, np.ndarray] = dict(zip(ld.concepts[b_mask], ld.vectors[b_mask]))
    amb_lookup: dict[str, str] = dict(zip(ld.concepts[a_mask], ld.ambiguity_types[a_mask]))
    cat_lookup: dict[str, str] = dict(zip(ld.concepts[a_mask], ld.categories[a_mask]))

    rows: list[dict] = []
    for concept, v_a in a_lookup.items():
        if concept not in b_lookup:
            continue
        v_b = b_lookup[concept]
        keep_aa = aa_words != concept
        keep_ca = ca_words != concept
        Vaa, Vca = V_aa[keep_aa], V_ca[keep_ca]
        n_aa, n_ca = len(Vaa), len(Vca)
        if n_aa == 0 or n_ca == 0:
            continue

        sims_a_anglo = v_a @ Vaa.T
        sims_a_catalan = v_a @ Vca.T
        sims_b_anglo = v_b @ Vaa.T
        sims_b_catalan = v_b @ Vca.T

        def _bias(pa: float, pc: float) -> float:
            d = pa + pc
            return float(pa / d) if d != 0 else float("nan")

        bias_a = _bias(float(sims_a_anglo.mean()), float(sims_a_catalan.mean()))
        bias_b = _bias(float(sims_b_anglo.mean()), float(sims_b_catalan.mean()))
        delta = bias_a - bias_b

        boots = np.empty(n_boot, dtype=np.float64)
        for k in range(n_boot):
            ia = rng.integers(0, n_aa, n_aa)
            ic = rng.integers(0, n_ca, n_ca)
            pa_a = (v_a @ Vaa[ia].T).mean()
            pa_c = (v_a @ Vca[ic].T).mean()
            pb_a = (v_b @ Vaa[ia].T).mean()
            pb_c = (v_b @ Vca[ic].T).mean()
            ba = _bias(float(pa_a), float(pa_c))
            bb = _bias(float(pb_a), float(pb_c))
            boots[k] = ba - bb
        ci_lo, ci_hi = np.nanquantile(boots, [0.025, 0.975])
        rows.append({
            "concept": concept,
            "ambiguity_type": amb_lookup.get(concept, "unknown"),
            "category": cat_lookup.get(concept, "unknown"),
            "frame_a": frame_a,
            "frame_b": frame_b,
            "bias_frame_a": bias_a,
            "bias_frame_b": bias_b,
            "delta": delta,
            "delta_ci_lo": float(ci_lo),
            "delta_ci_hi": float(ci_hi),
            "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        })
    return pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)


def _bare_subset(ld: LayerData) -> LayerData:
    mask = ld.frame_ids == "bare"
    return LayerData(
        layer=ld.layer,
        vectors=ld.vectors[mask],
        concepts=ld.concepts[mask],
        frame_ids=ld.frame_ids[mask],
        clusters=ld.clusters[mask],
        categories=ld.categories[mask],
        ambiguity_types=ld.ambiguity_types[mask],
    )


def ambiguity_stratified_bias(bias_df: pd.DataFrame) -> pd.DataFrame:
    """Median bias_ratio and median top10_catalan_frac per (ambiguity_type, frame)."""
    if "ambiguity_type" not in bias_df.columns:
        return pd.DataFrame()
    agg_cols = {"bias_ratio": "median"}
    topk_col = next((c for c in bias_df.columns if c.startswith("top") and c.endswith("_catalan_frac")),
                    None)
    if topk_col is not None:
        agg_cols[topk_col] = "median"
    agg = (bias_df
           .groupby(["ambiguity_type", "frame_id"])
           .agg({**agg_cols, "concept": "count"})
           .rename(columns={
               "bias_ratio": "median_bias_ratio",
               **({topk_col: f"median_{topk_col}"} if topk_col else {}),
               "concept": "count",
           })
           .reset_index())
    return agg


def knn_anchor_purity(ld: LayerData, k: int = 5) -> float:
    """Leave-one-out KNN cluster purity over anchor-only probes (bare frame)."""
    bare = _bare_subset(ld)
    anchor_mask = np.isin(bare.clusters, ["anchor_anglo", "anchor_catalan"])
    V = bare.vectors[anchor_mask]
    C = bare.clusters[anchor_mask]
    if len(V) <= k:
        return float("nan")
    sims = V @ V.T
    np.fill_diagonal(sims, -np.inf)
    nn_idx = np.argsort(-sims, axis=1)[:, :k]
    hits = (C[nn_idx] == C[:, None]).sum(axis=1)
    return float(hits.mean() / k)


def plot_umap(ld: LayerData, out_path: Path) -> None:
    try:
        import umap  # type: ignore
    except ImportError:
        log.warning("umap-learn not installed — skipping UMAP figure")
        return
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED,
    )
    xy = reducer.fit_transform(ld.vectors)

    fig, ax = plt.subplots(figsize=(9, 7))
    for cluster, color in CLUSTER_COLORS.items():
        mask = ld.clusters == cluster
        ax.scatter(xy[mask, 0], xy[mask, 1], c=color, s=18, alpha=0.55, label=cluster,
                   edgecolors="none")

    # Annotate a handful of target concepts (bare frame) + all anchors.
    bare_target = (ld.clusters == "target_catalan") & (ld.frame_ids == "bare")
    anchor = np.isin(ld.clusters, ["anchor_anglo", "anchor_catalan"])
    for i in np.where(bare_target | anchor)[0]:
        ax.annotate(
            ld.concepts[i], (xy[i, 0], xy[i, 1]),
            fontsize=6.5, alpha=0.8,
            xytext=(2, 2), textcoords="offset points",
        )
    ax.set_title(f"UMAP — layer {ld.layer}")
    ax.legend(loc="best", frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def process_model(model_dir: Path) -> None:
    files = sorted(model_dir.glob("embeddings_layer*.npz"))
    if not files:
        log.warning("no embeddings_layer*.npz in %s", model_dir)
        return
    log.info("analysing %s (%d layers)", model_dir, len(files))

    rng = np.random.default_rng(SEED)
    selection_rows: list[dict] = []

    for f in files:
        ld = load_layer(f)
        layer_dir = model_dir / f"layer{ld.layer}"
        layer_dir.mkdir(exist_ok=True)

        knn_df = compute_knn(ld)
        knn_df.to_csv(layer_dir / "knn.csv", index=False)

        bias_df = compute_bias(ld, rng)
        bias_df.to_csv(layer_dir / "bias.csv", index=False)

        ambig_df = ambiguity_stratified_bias(bias_df)
        if not ambig_df.empty:
            ambig_df.to_csv(layer_dir / "bias_by_ambiguity.csv", index=False)

        delta_df = compute_frame_differential(ld, rng)
        if not delta_df.empty:
            delta_df.to_csv(layer_dir / "delta.csv", index=False)

        plot_umap(ld, layer_dir / "umap.png")

        purity = knn_anchor_purity(ld, k=5)
        median_bias_by_frame = bias_df.groupby("frame_id")["bias_ratio"].median().to_dict()
        median_by_ambig = bias_df.groupby("ambiguity_type")["bias_ratio"].median().to_dict()
        topk_col = next((c for c in bias_df.columns if c.startswith("top") and c.endswith("_catalan_frac")),
                        None)
        median_topk_by_frame = (
            bias_df.groupby("frame_id")[topk_col].median().to_dict() if topk_col else {}
        )
        if not delta_df.empty:
            n_robust = int(delta_df["ci_excludes_zero"].sum())
            log.info("  layer %d  purity=%.3f  Δ median=%.3f  robust|Δ|>0: %d/%d  "
                     "bias by frame=%s  top10ca by frame=%s",
                     ld.layer, purity,
                     float(delta_df["delta"].median()), n_robust, len(delta_df),
                     {k: round(v, 3) for k, v in median_bias_by_frame.items()},
                     {k: round(v, 3) for k, v in median_topk_by_frame.items()})
        else:
            log.info("  layer %d  purity=%.3f  bias by frame=%s  top10ca by frame=%s",
                     ld.layer, purity,
                     {k: round(v, 3) for k, v in median_bias_by_frame.items()},
                     {k: round(v, 3) for k, v in median_topk_by_frame.items()})
        selection_rows.append({
            "layer": ld.layer,
            "knn_anchor_purity": purity,
            **{f"median_bias_{k}": v for k, v in median_bias_by_frame.items()},
            **{f"median_bias_amb_{k}": v for k, v in median_by_ambig.items()},
            **{f"median_top10ca_{k}": v for k, v in median_topk_by_frame.items()},
        })

    sel_df = pd.DataFrame(selection_rows).sort_values("layer")
    sel_df.to_csv(model_dir / "layer_selection.csv", index=False)

    if sel_df["knn_anchor_purity"].notna().any():
        max_purity = sel_df["knn_anchor_purity"].max()
        eligible = sel_df[sel_df["knn_anchor_purity"] >= max_purity - LAYER_PURITY_TOLERANCE]
        best = int(eligible.sort_values("layer", ascending=False).iloc[0]["layer"])
        criterion = f"deepest_layer_within_{LAYER_PURITY_TOLERANCE}_of_max_purity"
    else:
        best = int(sel_df["layer"].iloc[len(sel_df) // 2])
        criterion = "median_layer_fallback"
    with (model_dir / "best_layer.json").open("w") as f:
        json.dump({
            "best_layer": best,
            "criterion": criterion,
            "tolerance": LAYER_PURITY_TOLERANCE,
            "per_layer_purity": sel_df.set_index("layer")["knn_anchor_purity"].to_dict(),
        }, f, indent=2)
    log.info("best layer for %s: %d (criterion=%s)", model_dir.name, best, criterion)


def main() -> None:
    args = parse_args()
    for model_dir in discover_models(args.model):
        process_model(model_dir)


if __name__ == "__main__":
    main()
