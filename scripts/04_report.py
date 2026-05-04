#!/usr/bin/env python3
"""Compose ``out/report.md`` plus composite figures under ``out/figs/``.

Inputs: any ``out/{slug}/`` directory that contains ``best_layer.json``,
``layer{L}/bias.csv`` and ``embeddings_layer{L}.npz``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIGS = OUT / "figs"
SEED = 42

CLUSTER_COLORS = {
    "target_catalan": "#d62728",
    "anchor_anglo":   "#1f77b4",
    "anchor_catalan": "#2ca02c",
}


@dataclass
class ModelResults:
    slug: str
    best_layer: int
    bias: pd.DataFrame
    vectors: np.ndarray
    concepts: np.ndarray
    frame_ids: np.ndarray
    clusters: np.ndarray


def discover() -> list[ModelResults]:
    """Include models with bias.csv even when their NPZ is off-disk.

    Large-tier models (RunPod-extracted) commit only the analysis artifacts
    — `*.npz` is gitignored and stays on the pod. Such models have empty
    vector arrays, which downstream functions detect and skip the
    UMAP-based figures for (preserving any pre-rendered PNGs that were
    committed from the box that did have the NPZ).
    """
    results: list[ModelResults] = []
    empty = np.empty((0, 0), dtype=np.float32)
    empty_str = np.empty((0,), dtype=object)
    for d in sorted(OUT.iterdir()):
        if not d.is_dir():
            continue
        bp = d / "best_layer.json"
        if not bp.exists():
            continue
        best = int(json.loads(bp.read_text())["best_layer"])
        bias_csv = d / f"layer{best}" / "bias.csv"
        npz = d / f"embeddings_layer{best}.npz"
        if not bias_csv.exists():
            log.warning("skipping %s — missing layer %d bias.csv", d.name, best)
            continue
        bias = pd.read_csv(bias_csv)
        if npz.exists():
            with np.load(npz, allow_pickle=True) as z:
                results.append(ModelResults(
                    slug=d.name,
                    best_layer=best,
                    bias=bias,
                    vectors=z["vectors"].astype(np.float32),
                    concepts=z["concepts"].astype(str),
                    frame_ids=z["frame_ids"].astype(str),
                    clusters=z["clusters"].astype(str),
                ))
        else:
            log.info("%s — bias.csv present, NPZ off-disk; vector-free model row",
                     d.name)
            results.append(ModelResults(
                slug=d.name,
                best_layer=best,
                bias=bias,
                vectors=empty,
                concepts=empty_str,
                frame_ids=empty_str,
                clusters=empty_str,
            ))
    return results


def _has_vectors(mr: "ModelResults") -> bool:
    return mr.vectors.size > 0


def _family_version_sort_key(slug: str) -> tuple:
    """Sort by family then full numeric tuple, so within a generation the
    smaller variant precedes the larger (gemma-4-e2b < gemma-4-26b-a4b).

    Family priority: gemma < qwen < salamandra (AINA family) < alia.
    Putting the Catalan-aware tier last so its panels show up at the end
    of every figure — easier visual comparison vs. the general-purpose
    six.
    """
    s = slug.lower()
    if "gemma" in s:
        fam = "1_gemma"
    elif "qwen" in s:
        fam = "2_qwen"
    elif "mistral" in s:
        fam = "3_mistral"
    elif "deepseek" in s:
        fam = "4_deepseek"
    elif "salamandra" in s:
        fam = "5_salamandra"
    elif "alia" in s:
        fam = "6_alia"
    else:
        fam = "9_other"
    digits = tuple(float(d) for d in re.findall(r"[\d.]+", slug))
    return (fam, digits or (0.0,))


def _grid_shape(n: int) -> tuple[int, int]:
    """Single row up to 4 models; 3×2 for 5–6 models; 3×3 for 7–9.

    The 3×3 branch (added when the Catalan-aware control tier brought the panel
    count to 9) keeps the rendered figure short enough that LaTeX captions fit
    on the same page at width=\\textwidth. Tall 5×2 layouts overflow the
    page-text height once the caption is appended.
    """
    if n <= 4:
        return 1, n
    if n <= 6:
        return (n + 1) // 2, 2
    if n <= 9:
        return 3, 3
    return 3, 4


def headline_figure(results: list[ModelResults], out_path: Path) -> None:
    """Bare-frame only — clearer geometry without the four-frame red cloud.

    If some models have NPZ off-disk (typical when the large tier was
    extracted on RunPod and the NPZ stayed there), we preserve the
    previously committed full-panel PNG and instead render a
    `*_local.png` sibling with only the models we can re-render. This
    means the local box can still surface the AINA tier alongside the
    small general-purpose tier without clobbering the canonical PNG.
    """
    try:
        import umap  # type: ignore
    except ImportError:
        log.warning("umap-learn not installed — skipping headline figure")
        return
    rendered = [mr for mr in results if _has_vectors(mr)]
    if not rendered:
        return
    if len(rendered) < len(results):
        missing = [mr.slug for mr in results if not _has_vectors(mr)]
        local_path = out_path.with_name(out_path.stem + "_local" + out_path.suffix)
        log.info("partial render — NPZ off-disk for: %s; canonical PNG preserved, "
                 "local subset rendered to %s", ", ".join(missing), local_path.name)
        out_path = local_path
        results = rendered
    ordered = sorted(results, key=lambda mr: _family_version_sort_key(mr.slug))
    n = len(ordered)
    nrows, ncols = _grid_shape(n)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(8.5 * ncols, 8.5 * nrows),
                             squeeze=False)
    flat_axes = axes.flatten()
    for ax, mr in zip(flat_axes[:n], ordered):
        bare = mr.frame_ids == "bare"
        V = mr.vectors[bare]
        clusters = mr.clusters[bare]
        concepts = mr.concepts[bare]
        reducer = umap.UMAP(
            n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED,
        )
        xy = reducer.fit_transform(V)
        for cluster, color in CLUSTER_COLORS.items():
            m = clusters == cluster
            size = 48 if cluster == "target_catalan" else 130
            ax.scatter(xy[m, 0], xy[m, 1], c=color, s=size, alpha=0.75,
                       label=cluster, edgecolors="white", linewidths=0.6)
        anchor_mask = np.isin(clusters, ["anchor_anglo", "anchor_catalan"])
        for i in np.where(anchor_mask)[0]:
            ax.annotate(concepts[i], (xy[i, 0], xy[i, 1]),
                        fontsize=11, alpha=0.95,
                        xytext=(4, 4), textcoords="offset points",
                        weight="bold")
        # also annotate "coca" if present (it's the title's hero)
        coca_idx = np.where(concepts == "coca")[0]
        for i in coca_idx:
            ax.annotate("coca", (xy[i, 0], xy[i, 1]),
                        fontsize=13, color="#000",
                        xytext=(5, -12), textcoords="offset points",
                        weight="bold",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  facecolor="#fff8b0", edgecolor="#000", alpha=0.9))
        ax.set_title(f"{mr.slug} — layer {mr.best_layer}", fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="best", frameon=False, fontsize=11)
    for ax in flat_axes[n:]:
        ax.axis("off")
    fig.suptitle("UMAP of bare-frame contextualized hidden states "
                 "(generations top → bottom, family left → right)",
                 y=1.003, fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _per_concept_anglo_counts(results: list[ModelResults], concept: str) -> pd.DataFrame:
    rows: list[dict] = []
    for mr in results:
        knn_path = OUT / mr.slug / f"layer{mr.best_layer}" / "knn.csv"
        if not knn_path.exists():
            continue
        knn = pd.read_csv(knn_path)
        c = knn[knn["concept"] == concept]
        if c.empty:
            continue
        for frame, sub in c.groupby("frame_id"):
            top10 = sub.sort_values("rank").head(10)
            rows.append({
                "model": mr.slug,
                "frame_id": frame,
                "n_anglo_top10": int((top10["neighbor_cluster"] == "anchor_anglo").sum()),
            })
    return pd.DataFrame(rows)


def concept_anglo_bar_chart(
    results: list[ModelResults],
    concept: str,
    out_path: Path,
    title: str | None = None,
) -> None:
    df = _per_concept_anglo_counts(results, concept)
    if df.empty:
        return
    pivot = (df
             .pivot(index="frame_id", columns="model", values="n_anglo_top10")
             .reindex(index=["bare", "anglo_en", "catalan_frame", "neutral_ca"])
             .reindex(columns=sorted(df["model"].unique(),
                                     key=_family_version_sort_key)))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    # Two-tone Gemma palette (light → dark), two-tone Qwen palette, then
    # two extra slots reserved for a third tier per family (e.g. RunPod
    # large models). Truncated to actual column count.
    palette = ["#a6cee3", "#1f78b4", "#08306b",   # Gemma  (light → dark)
               "#fb9a99", "#e31a1c", "#67000d",   # Qwen   (light → dark)
               "#b2df8a", "#33a02c", "#006d2c"]   # AINA   (light → dark)
    pivot.plot(kind="bar", ax=ax, width=0.82,
               color=palette[:len(pivot.columns)])
    ax.axhline(5, color="gray", linestyle="--", alpha=0.55, linewidth=1)
    ax.text(3.45, 5.05, "neutral midpoint", color="gray", fontsize=8,
            va="bottom", ha="right")
    ax.set_ylabel(f'Anglo anchors in top-10 KNN of "{concept}"')
    ax.set_xlabel("probe frame")
    ax.set_ylim(0, 10)
    ax.set_yticks(range(0, 11, 2))
    ax.set_title(title or
                 f"Anglo contamination of *{concept}* by frame and model",
                 fontsize=11)
    ax.legend(title="model", frameon=False, loc="upper left", fontsize=9)
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


FRAME_COLORS = {
    "bare":          "#666666",  # gray
    "anglo_en":      "#1f77b4",  # blue (English context)
    "catalan_frame": "#d62728",  # red (Catalan context)
    "neutral_ca":    "#ff7f0e",  # orange (neutral Catalan)
}
FRAME_ORDER = ["bare", "anglo_en", "neutral_ca", "catalan_frame"]  # adversarial → native


def concept_trajectory_panel(
    results: list[ModelResults],
    concept: str,
    out_path: Path,
) -> None:
    """Per-model UMAP zoomed on a single concept's four-frame trajectory.

    Each panel keeps that model's 40 anchors as scaffolding, then over-plots
    the four frame embeddings of `concept` as large stars colored by frame.
    Reading left → right (chronological model order) shows how each
    successive generation moves the concept around its anchor cloud.
    """
    try:
        import umap  # type: ignore
    except ImportError:
        log.warning("umap-learn not installed — skipping trajectory plot for %s", concept)
        return
    rendered = [mr for mr in results if _has_vectors(mr)]
    if not rendered:
        return
    if len(rendered) < len(results):
        missing = [mr.slug for mr in results if not _has_vectors(mr)]
        local_path = out_path.with_name(out_path.stem + "_local" + out_path.suffix)
        log.info("partial trajectory render for %s — NPZ off-disk for: %s; "
                 "canonical PNG preserved, local subset → %s",
                 concept, ", ".join(missing), local_path.name)
        out_path = local_path
        results = rendered

    ordered = sorted(results, key=lambda mr: _family_version_sort_key(mr.slug))
    n = len(ordered)
    nrows, ncols = _grid_shape(n)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7.5 * ncols, 7.5 * nrows),
                             squeeze=False)
    flat_axes = axes.flatten()

    for ax, mr in zip(flat_axes[:n], ordered):
        target_mask = (mr.concepts == concept) & (mr.clusters == "target_catalan")
        if not target_mask.any():
            ax.set_title(f"{mr.slug} — {concept} not found")
            ax.axis("off")
            continue
        anchor_mask = np.isin(mr.clusters, ["anchor_anglo", "anchor_catalan"])
        keep = target_mask | anchor_mask
        V = mr.vectors[keep]
        labels = mr.concepts[keep]
        clusters = mr.clusters[keep]
        frames = mr.frame_ids[keep]
        is_target = (clusters == "target_catalan")

        reducer = umap.UMAP(
            n_neighbors=min(10, len(V) - 1),
            min_dist=0.1, metric="cosine", random_state=SEED,
        )
        xy = reducer.fit_transform(V)

        anglo_pts = xy[clusters == "anchor_anglo"]
        catalan_pts = xy[clusters == "anchor_catalan"]
        anglo_lbls = labels[clusters == "anchor_anglo"]
        catalan_lbls = labels[clusters == "anchor_catalan"]

        ax.scatter(anglo_pts[:, 0], anglo_pts[:, 1], c="#1f77b4", s=100,
                   alpha=0.55, edgecolors="white", linewidths=0.6,
                   label="Anglo anchor")
        ax.scatter(catalan_pts[:, 0], catalan_pts[:, 1], c="#2ca02c", s=100,
                   alpha=0.55, edgecolors="white", linewidths=0.6,
                   label="Catalan anchor")
        for pt, lbl in zip(anglo_pts, anglo_lbls):
            ax.annotate(lbl, (pt[0], pt[1]), fontsize=10, alpha=0.9,
                        xytext=(4, 4), textcoords="offset points")
        for pt, lbl in zip(catalan_pts, catalan_lbls):
            ax.annotate(lbl, (pt[0], pt[1]), fontsize=10, alpha=0.9,
                        xytext=(4, 4), textcoords="offset points")

        # Frame-coloured stars for the concept itself.
        target_pts = xy[is_target]
        target_frames = frames[is_target]
        order_lookup = {f: i for i, f in enumerate(FRAME_ORDER)}
        sort_idx = sorted(range(len(target_frames)),
                          key=lambda i: order_lookup.get(target_frames[i], 99))
        for i in sort_idx:
            pt = target_pts[i]
            f = target_frames[i]
            ax.scatter(pt[0], pt[1], c=FRAME_COLORS.get(f, "#000"),
                       s=480, marker="*", edgecolors="black",
                       linewidths=1.8, zorder=10,
                       label=f"{concept} · {f}")
            ax.annotate(f, (pt[0], pt[1]),
                        fontsize=10, weight="bold", color="#000",
                        xytext=(10, 8), textcoords="offset points", zorder=11,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor=FRAME_COLORS.get(f, "#fff"),
                                  edgecolor="black", alpha=0.85,
                                  linewidth=0.7))

        ax.set_title(f"{mr.slug} — layer {mr.best_layer}", fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])

    for ax in flat_axes[n:]:
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
                   markersize=12, label="Anglo anchor"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
                   markersize=12, label="Catalan anchor"),
    ]
    for f in FRAME_ORDER:
        handles.append(plt.Line2D([0], [0], marker="*", color="w",
                                  markerfacecolor=FRAME_COLORS[f],
                                  markeredgecolor="black", markersize=16,
                                  label=f"{concept} · {f}"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.015))

    fig.suptitle(
        f"Trajectory of *{concept}* across four frames — chronological model order",
        y=1.005, fontsize=15,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def cross_gen_heatmap(gen_df: pd.DataFrame, out_path: Path) -> None:
    if gen_df.empty:
        return
    pivot = (gen_df
             .pivot_table(index=["family", "ambiguity_type"],
                          columns="frame_id",
                          values="delta_newer_minus_older")
             .reindex(columns=["bare", "anglo_en", "catalan_frame", "neutral_ca"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vmax = max(0.4, float(np.nanmax(np.abs(pivot.values))))
    im = ax.imshow(pivot.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{fam} · {amb}" for (fam, amb) in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isnan(v):
                continue
            text_color = "white" if abs(v) > vmax * 0.5 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color=text_color, fontsize=10, weight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Δ top10_catalan_frac\n(newer − older)", fontsize=9)
    ax.set_title("Cross-generation evolution — sign tells the family's strategy",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def frame_differential_per_concept(
    results: list[ModelResults],
    frame_a: str = "anglo_en",
    frame_b: str = "catalan_frame",
) -> pd.DataFrame:
    """Per-(model, concept) Δ = bias(c, frame_a) − bias(c, frame_b) at best layer.

    Prefer the proper paired-bootstrap CI from ``layer{L}/delta.csv`` when
    present (small tier, where 03 had access to NPZ). Fall back to a
    *conservative* CI derived from marginal bootstrap CIs of each frame —
    `Δ_lo = bias_a_lo − bias_b_hi`, `Δ_hi = bias_a_hi − bias_b_lo` — when
    only ``layer{L}/bias.csv`` is on disk (large tier, RunPod-extracted,
    NPZ kept off-repo). The conservative CI assumes anchor independence
    across the two frames, which is wrong (they share anchors), so the
    fallback overstates uncertainty — that is the safer direction for
    direction-1 ranking.
    """
    rows: list[dict] = []
    for mr in results:
        layer_dir = OUT / mr.slug / f"layer{mr.best_layer}"
        delta_path = layer_dir / "delta.csv"
        bias_path = layer_dir / "bias.csv"
        if delta_path.exists():
            df = pd.read_csv(delta_path).copy()
            df["model"] = mr.slug
            df["ci_kind"] = "paired_bootstrap"
            rows.append(df)
            continue
        if not bias_path.exists():
            continue
        bias = pd.read_csv(bias_path)
        a = bias[bias["frame_id"] == frame_a].set_index("concept")
        b = bias[bias["frame_id"] == frame_b].set_index("concept")
        common = a.index.intersection(b.index)
        if len(common) == 0:
            continue
        out = pd.DataFrame({
            "concept": common,
            "ambiguity_type": a.loc[common, "ambiguity_type"].values,
            "category": a.loc[common, "category"].values,
            "frame_a": frame_a,
            "frame_b": frame_b,
            "bias_frame_a": a.loc[common, "bias_ratio"].values,
            "bias_frame_b": b.loc[common, "bias_ratio"].values,
        })
        out["delta"] = out["bias_frame_a"] - out["bias_frame_b"]
        if {"bias_ratio_ci_lo", "bias_ratio_ci_hi"}.issubset(bias.columns):
            out["delta_ci_lo"] = (a.loc[common, "bias_ratio_ci_lo"].values
                                  - b.loc[common, "bias_ratio_ci_hi"].values)
            out["delta_ci_hi"] = (a.loc[common, "bias_ratio_ci_hi"].values
                                  - b.loc[common, "bias_ratio_ci_lo"].values)
            out["ci_excludes_zero"] = (out["delta_ci_lo"] > 0) | (out["delta_ci_hi"] < 0)
        else:
            out["delta_ci_lo"] = np.nan
            out["delta_ci_hi"] = np.nan
            out["ci_excludes_zero"] = False
        out["model"] = mr.slug
        out["ci_kind"] = "conservative_marginal"
        rows.append(out.reset_index(drop=True))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


_AMBIGUITY_COLORS = {
    "cross_lingual_bias": "#1b9e77",
    "domain_specific":    "#7570b3",
    "internal_polysemy":  "#d95f02",
}


def frame_differential_forest(
    delta_df: pd.DataFrame,
    out_path: Path,
    top_k: int = 12,
) -> None:
    """One forest-plot panel per model, ranking concepts by paired Δ.

    Top-k concepts with the largest Δ at each model's best layer, with the
    bootstrap CI as a horizontal error bar; concepts whose CI excludes 0
    get a ★ marker. Color encodes ambiguity_type (the headline stratum
    structurally responsible for whatever the row is showing).
    """
    if delta_df.empty:
        return
    ordered_models = sorted(delta_df["model"].unique(),
                            key=_family_version_sort_key)
    n = len(ordered_models)
    nrows, ncols = _grid_shape(n)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 0.36 * top_k * nrows + 1.5),
                             squeeze=False)
    flat_axes = axes.flatten()

    for ax, slug in zip(flat_axes[:n], ordered_models):
        sub = (delta_df[delta_df["model"] == slug]
               .sort_values("delta", ascending=False)
               .head(top_k)
               .reset_index(drop=True))
        if sub.empty:
            ax.set_title(f"{slug} — no delta data")
            ax.axis("off")
            continue
        ys = np.arange(len(sub))[::-1]  # top of plot = largest Δ
        for amb, color in _AMBIGUITY_COLORS.items():
            mask = sub["ambiguity_type"].values == amb
            if not mask.any():
                continue
            ax.scatter(sub.loc[mask, "delta"], ys[mask],
                       c=color, s=70, zorder=3, label=amb,
                       edgecolors="black", linewidths=0.5)
        # Error bars (only where CI is finite)
        finite = sub["delta_ci_lo"].notna() & sub["delta_ci_hi"].notna()
        if finite.any():
            xerr = np.array([
                (sub["delta"] - sub["delta_ci_lo"]).where(finite, 0).values,
                (sub["delta_ci_hi"] - sub["delta"]).where(finite, 0).values,
            ])
            ax.errorbar(sub["delta"], ys, xerr=xerr, fmt="none",
                        ecolor="#444", elinewidth=1.0, capsize=3, zorder=2)
        # Star next to robust rows
        for i, robust in enumerate(sub["ci_excludes_zero"].fillna(False).values):
            if robust:
                ax.text(sub["delta"].iloc[i], ys[i], "  ★",
                        va="center", ha="left", fontsize=12, color="#000")
        ax.axvline(0, color="#bbb", linewidth=0.8, zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels(sub["concept"].tolist(), fontsize=12)
        ax.set_xlabel("Δ = bias(anglo_en) − bias(catalan_frame)", fontsize=13)
        ax.tick_params(axis="x", labelsize=11)
        ci_kind = sub["ci_kind"].iloc[0] if "ci_kind" in sub.columns else "?"
        ax.set_title(f"{slug}   ({ci_kind})", fontsize=14)
        ax.legend(loc="lower right", frameon=False, fontsize=10)
    for ax in flat_axes[n:]:
        ax.axis("off")
    fig.suptitle("Top concepts by paired Δ — ★ = 95% CI excludes 0",
                 y=1.005, fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Direction (2) — layer-depth × frame interaction
# ---------------------------------------------------------------------------

def layer_depth_delta(results: list[ModelResults]) -> pd.DataFrame:
    """Per (model, layer, ambiguity_type) median Δ across all committed layers.

    Direction (2) of the README: pick a *single* best layer hides whether the
    `anglo_en − catalan_frame` differential peaks at a different depth than
    the KNN-purity does. Reading bias.csv at all three layers per model lets
    us draw the depth-vs-Δ trajectory.
    """
    rows: list[dict] = []
    for mr in results:
        model_dir = OUT / mr.slug
        for bias_path in sorted(model_dir.glob("layer*/bias.csv")):
            layer_match = re.search(r"layer(\d+)", bias_path.parent.name)
            if not layer_match:
                continue
            L = int(layer_match.group(1))
            bias = pd.read_csv(bias_path)
            if "ambiguity_type" not in bias.columns:
                continue
            a = bias[bias["frame_id"] == "anglo_en"].set_index("concept")
            b = bias[bias["frame_id"] == "catalan_frame"].set_index("concept")
            common = a.index.intersection(b.index)
            if len(common) == 0:
                continue
            delta = (a.loc[common, "bias_ratio"].values
                     - b.loc[common, "bias_ratio"].values)
            amb_lookup = a.loc[common, "ambiguity_type"]
            tmp = pd.DataFrame({"ambiguity_type": amb_lookup.values, "delta": delta})
            for amb, sub in tmp.groupby("ambiguity_type"):
                rows.append({
                    "model": mr.slug,
                    "layer": L,
                    "ambiguity_type": amb,
                    "median_delta": float(sub["delta"].median()),
                    "n": len(sub),
                })
            rows.append({
                "model": mr.slug,
                "layer": L,
                "ambiguity_type": "_all",
                "median_delta": float(np.median(delta)),
                "n": len(delta),
            })
    return pd.DataFrame(rows)


def layer_depth_delta_plot(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    ordered_models = sorted(df["model"].unique(), key=_family_version_sort_key)
    n = len(ordered_models)
    nrows, ncols = _grid_shape(n)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4 * nrows),
                             squeeze=False)
    flat_axes = axes.flatten()
    palette = {
        "cross_lingual_bias": "#1b9e77",
        "domain_specific":    "#7570b3",
        "internal_polysemy":  "#d95f02",
        "_all":               "#000000",
    }
    for ax, slug in zip(flat_axes[:n], ordered_models):
        sub = df[df["model"] == slug].sort_values(["ambiguity_type", "layer"])
        for amb, color in palette.items():
            s = sub[sub["ambiguity_type"] == amb]
            if s.empty:
                continue
            ls = "--" if amb == "_all" else "-"
            ax.plot(s["layer"], s["median_delta"], marker="o", color=color,
                    linewidth=2, linestyle=ls,
                    label=("pooled median" if amb == "_all" else amb))
        ax.axhline(0, color="#bbb", linewidth=0.6)
        ax.set_xlabel("layer index", fontsize=13)
        ax.set_ylabel("median Δ (anglo_en − catalan_frame)", fontsize=13)
        ax.set_title(slug, fontsize=14)
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(frameon=False, fontsize=11)
    for ax in flat_axes[n:]:
        ax.axis("off")
    fig.suptitle("Layer-depth × frame interaction — Δ across the committed layer sweep",
                 y=1.005, fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Title test — does *coca* drift into the noise (drug / brand) cluster?
# ---------------------------------------------------------------------------

def coca_lexical_collision_test(
    results: list[ModelResults],
    concept: str = "coca",
    k: int = 10,
) -> pd.DataFrame:
    """Per (model, frame) count of top-k KNN hits in the noise (drug/brand)
    cluster vs. anglo vs. catalan, for the title concept.

    The reviewer's issue (1): the paper title promises a literal
    *coca↔cocaine* lexical collision (F1) but the headline figures only
    show *coca → fairy-tale neighborhood* (F2 — cultural-frame
    displacement). To validate or falsify the title we add a noise cluster
    of {cocaine, coke, Coca-Cola, pepsi, drug, powder, addiction, soda,
    beverage, brand, caffeine, sugar, logo} and report whether any of
    those words appear in the top-k of *coca* under any frame. If yes,
    F1 is empirically grounded. If not, the title must be reframed.
    """
    rows: list[dict] = []
    for mr in results:
        knn_path = OUT / mr.slug / f"layer{mr.best_layer}" / "knn.csv"
        if not knn_path.exists():
            continue
        knn = pd.read_csv(knn_path)
        c = knn[knn["concept"] == concept]
        if c.empty:
            continue
        for frame, sub in c.groupby("frame_id"):
            top = sub.sort_values("rank").head(k)
            n_anglo = int((top["neighbor_cluster"] == "anchor_anglo").sum())
            n_catalan = int((top["neighbor_cluster"] == "anchor_catalan").sum())
            n_noise = int((top["neighbor_cluster"] == "anchor_noise").sum())
            noise_hits = top.loc[top["neighbor_cluster"] == "anchor_noise",
                                 "neighbor"].tolist()
            rows.append({
                "model": mr.slug,
                "frame_id": frame,
                "n_anglo_top10": n_anglo,
                "n_catalan_top10": n_catalan,
                "n_noise_top10": n_noise,
                "noise_hits": ", ".join(noise_hits) if noise_hits else "",
            })
    return pd.DataFrame(rows)


def coca_lexical_chart(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    ordered = sorted(df["model"].unique(), key=_family_version_sort_key)
    df = df.copy()
    df["model"] = pd.Categorical(df["model"], ordered)
    df["frame_id"] = pd.Categorical(
        df["frame_id"], ["bare", "anglo_en", "catalan_frame", "neutral_ca"]
    )
    df = df.sort_values(["frame_id", "model"]).reset_index(drop=True)

    cols = ["n_noise_top10", "n_anglo_top10", "n_catalan_top10"]
    colors = ["#9467bd", "#fb9a99", "#2ca02c"]
    labels = ["noise (drug/brand)", "anglo (fairy-tale)", "catalan (folkloric)"]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    for col, color, label in zip(cols, colors, labels):
        ax.bar(x, df[col].values, bottom=bottom, color=color, label=label,
               edgecolor="white", linewidth=0.4)
        bottom += df[col].values

    ax.set_xticks(x)
    short = {
        "gemma2-2b": "G2-2B", "gemma4-e2b": "G4-E2B",
        "gemma4-26b-a4b": "G4-26B",
        "qwen2-1.5b": "Q2-1.5B", "qwen3.5-4b-base": "Q3.5-4B",
        "qwen3.6-35b-a3b": "Q3.6-35B",
        "mistral-small-24b": "M-24B",
        "deepseek-llm-67b": "DS-67B",
        "salamandra-2b": "Sal-2B", "salamandra-7b": "Sal-7B",
        "ALIA-40b": "ALIA-40B",
    }
    ax.set_xticklabels([short.get(m, m) for m in df["model"]],
                       rotation=35, ha="right", fontsize=8)
    n_models = len(ordered)
    for k in range(1, 4):
        ax.axvline(k * n_models - 0.5, color="#aaa", linewidth=0.5)
    for k, frame in enumerate(["bare", "anglo_en", "catalan_frame", "neutral_ca"]):
        ax.text(k * n_models + n_models / 2 - 0.5, 10.4, frame,
                ha="center", va="bottom", fontsize=10, weight="bold")
    ax.set_ylim(0, 11)
    ax.set_ylabel("top-10 KNN composition")
    ax.set_title("Coca lexical-collision test — does *coca* land near drug/brand?",
                 fontsize=12)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Direction (3) — sub-classify the Anglo anchor cluster
# ---------------------------------------------------------------------------

def anglo_subcluster_pull(results: list[ModelResults], focus: list[str]) -> pd.DataFrame:
    """Per (model, focus_concept, frame, anglo_subcluster) count of top-10 KNN hits.

    Reads `data/anchors_anglo.csv`'s `subcluster` column (added in direction 3
    of the README — `fairy_tale`, `medieval`, `character`) and joins it onto
    `out/{slug}/layer{L}/knn.csv` to resolve which Anglo neighborhood is
    actually catching each focus concept's drift, broken out by frame.
    """
    anglo_path = ROOT / "data" / "anchors_anglo.csv"
    if not anglo_path.exists():
        return pd.DataFrame()
    anglo = pd.read_csv(anglo_path)
    if "subcluster" not in anglo.columns:
        return pd.DataFrame()
    sub_lookup = dict(zip(anglo["anchor"], anglo["subcluster"]))

    rows: list[dict] = []
    for mr in results:
        knn_path = OUT / mr.slug / f"layer{mr.best_layer}" / "knn.csv"
        if not knn_path.exists():
            continue
        knn = pd.read_csv(knn_path)
        for c in focus:
            cdf = knn[knn["concept"] == c]
            if cdf.empty:
                continue
            for frame, sub in cdf.groupby("frame_id"):
                top10 = sub.sort_values("rank").head(10)
                anglo_rows = top10[top10["neighbor_cluster"] == "anchor_anglo"]
                counts = {sc: 0 for sc in {"fairy_tale", "medieval", "character"}}
                for n in anglo_rows["neighbor"]:
                    sc = sub_lookup.get(n)
                    if sc in counts:
                        counts[sc] += 1
                rows.append({
                    "model": mr.slug,
                    "concept": c,
                    "frame_id": frame,
                    "n_fairy_tale": counts["fairy_tale"],
                    "n_medieval": counts["medieval"],
                    "n_character": counts["character"],
                    "n_catalan": int((top10["neighbor_cluster"] == "anchor_catalan").sum()),
                })
    return pd.DataFrame(rows)


def anglo_subcluster_chart(df: pd.DataFrame, concept: str, out_path: Path) -> None:
    sub = df[df["concept"] == concept].copy()
    if sub.empty:
        return
    ordered = sorted(sub["model"].unique(), key=_family_version_sort_key)
    sub["model"] = pd.Categorical(sub["model"], ordered)
    sub["frame_id"] = pd.Categorical(
        sub["frame_id"], ["bare", "anglo_en", "catalan_frame", "neutral_ca"]
    )
    sub = sub.sort_values(["frame_id", "model"]).reset_index(drop=True)

    cols = ["n_fairy_tale", "n_medieval", "n_character", "n_catalan"]
    colors = ["#e31a1c", "#fb9a99", "#fdbf6f", "#33a02c"]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(sub))
    bottom = np.zeros(len(sub))
    for col, color in zip(cols, colors):
        ax.bar(x, sub[col].values, bottom=bottom, color=color, label=col,
               edgecolor="white", linewidth=0.4)
        bottom += sub[col].values

    # Tick labels: model abbreviated under each bar; frame label spans the four
    # bars per frame group.
    ax.set_xticks(x)
    short = {
        "gemma2-2b": "G2-2B",
        "gemma4-e2b": "G4-E2B",
        "gemma4-26b-a4b": "G4-26B",
        "qwen2-1.5b": "Q2-1.5B",
        "qwen3.5-4b-base": "Q3.5-4B",
        "qwen3.6-35b-a3b": "Q3.6-35B",
        "mistral-small-24b": "M-24B",
        "deepseek-llm-67b": "DS-67B",
    }
    ax.set_xticklabels([short.get(m, m) for m in sub["model"]],
                       rotation=35, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    # Frame group dividers
    n_models = len(ordered)
    for k in range(1, 4):
        ax.axvline(k * n_models - 0.5, color="#aaa", linewidth=0.5)
    for k, frame in enumerate(["bare", "anglo_en", "catalan_frame", "neutral_ca"]):
        ax.text(k * n_models + n_models / 2 - 0.5, 10.4, frame,
                ha="center", va="bottom", fontsize=13, weight="bold")
    ax.set_ylim(0, 11)
    ax.set_ylabel("top-10 KNN composition", fontsize=13)
    ax.set_title(f"Anglo subcluster pull on *{concept}* — which English universe catches the drift",
                 fontsize=14)
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Direction (4) — stratified-bootstrap CI on ambiguity-type medians
# ---------------------------------------------------------------------------

def stratified_ambiguity_ci(
    results: list[ModelResults],
    n_boot: int = 1000,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Per (model, ambiguity_type, frame) median bias_ratio with bootstrap CI.

    The bootstrap resamples *concepts within each ambiguity_type* (not
    anchors), so the CI tells us how stable the cell-level median is to the
    particular concepts we happened to put in that stratum. Direction (4) of
    the README. Particularly important because `cross_lingual_bias` has only
    9 concepts — its medians can move a lot under resampling.
    """
    rng = np.random.default_rng(rng_seed)
    rows: list[dict] = []
    for mr in results:
        if "ambiguity_type" not in mr.bias.columns:
            continue
        for (amb, frame), sub in mr.bias.groupby(["ambiguity_type", "frame_id"]):
            vals = sub["bias_ratio"].dropna().to_numpy()
            if len(vals) == 0:
                continue
            med = float(np.median(vals))
            boots = np.empty(n_boot)
            n = len(vals)
            for k in range(n_boot):
                idx = rng.integers(0, n, n)
                boots[k] = np.median(vals[idx])
            ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975])
            rows.append({
                "model": mr.slug,
                "ambiguity_type": amb,
                "frame_id": frame,
                "n": n,
                "median": med,
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
            })
    return pd.DataFrame(rows)


def ambiguity_ci_forest(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    frames = ["bare", "anglo_en", "neutral_ca", "catalan_frame"]
    df = df[df["frame_id"].isin(frames)].copy()
    df["frame_id"] = pd.Categorical(df["frame_id"], frames)
    ordered_models = sorted(df["model"].unique(), key=_family_version_sort_key)
    n_models = len(ordered_models)
    n_frames = len(frames)
    nrows, ncols = _grid_shape(n_frames)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 0.32 * n_models * 3 * nrows + 1),
                             squeeze=False)
    flat_axes = axes.flatten()

    palette = _AMBIGUITY_COLORS
    for ax, frame in zip(flat_axes[:n_frames], frames):
        sub = df[df["frame_id"] == frame].copy()
        sub["model"] = pd.Categorical(sub["model"], ordered_models)
        sub = sub.sort_values(["model", "ambiguity_type"]).reset_index(drop=True)

        # Layout: per model, three rows (one per ambiguity_type)
        amb_order = ["cross_lingual_bias", "domain_specific", "internal_polysemy"]
        ys: list[float] = []
        labels: list[str] = []
        offset = 0.25
        for i, slug in enumerate(ordered_models):
            for j, amb in enumerate(amb_order):
                row = sub[(sub["model"] == slug) & (sub["ambiguity_type"] == amb)]
                if row.empty:
                    continue
                y = -(i * 4 + j * 0.9)
                ys.append(y)
                labels.append(f"{slug}  ·  {amb}")
                ax.errorbar(row["median"].values, y,
                            xerr=[(row["median"] - row["ci_lo"]).values,
                                  (row["ci_hi"] - row["median"]).values],
                            fmt="o", color=palette.get(amb, "#000"),
                            ecolor=palette.get(amb, "#000"), elinewidth=1.5,
                            capsize=3, markersize=8)
        ax.axvline(0.5, color="#bbb", linewidth=0.6, linestyle="--")
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("median bias_ratio")
        ax.set_title(f"frame = {frame}", fontsize=11)
        ax.set_xlim(0.30, 0.55)
    for ax in flat_axes[n_frames:]:
        ax.axis("off")
    fig.suptitle("Stratified-bootstrap 95% CI on median bias_ratio per (model, ambiguity_type)",
                 y=1.003, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Direction (5) — Procrustes cross-model alignment + combined UMAP
# ---------------------------------------------------------------------------

def procrustes_combined_umap(
    results: list[ModelResults],
    focus: list[str],
    out_path: Path,
    pca_dim: int = 30,
    anchor_subset: list[str] | None = None,
    title_suffix: str = "",
) -> None:
    """One combined UMAP of all small-tier models' anchors + focus targets.

    Each model is PCA-reduced to ``pca_dim`` and then orthogonally aligned to
    the first (chronologically earliest) model using its shared bare-frame
    anchor set. The combined matrix runs through one UMAP. For each focus
    concept we draw four polylines (one per frame) that visit the concept's
    aligned position across the small-tier models in chronological order —
    so the reader can literally trace where, e.g., `coca` migrates as we
    move from Gemma 2 → Gemma 4 → Qwen2 → Qwen 3.5.

    Big-tier models (no NPZ on disk) are excluded; this is documented in the
    figure caption and discussed in README direction 5.
    """
    try:
        import umap  # type: ignore
        from scipy.linalg import orthogonal_procrustes  # type: ignore
        from sklearn.decomposition import PCA  # type: ignore
    except ImportError as e:
        log.warning("procrustes plot deps missing (%s) — skipping", e)
        return
    small = [mr for mr in results if _has_vectors(mr)]
    if len(small) < 2:
        log.info("procrustes: need ≥2 small-tier models, got %d — skipping", len(small))
        return
    ordered = sorted(small, key=lambda mr: _family_version_sort_key(mr.slug))

    # Establish the canonical anchor order (intersection across all models).
    anchor_sets = []
    for mr in ordered:
        m = (np.isin(mr.clusters, ["anchor_anglo", "anchor_catalan"])
             & (mr.frame_ids == "bare"))
        anchor_sets.append(set(mr.concepts[m].tolist()))
    canonical = sorted(set.intersection(*anchor_sets))
    if len(canonical) < pca_dim + 1:
        log.warning("procrustes: only %d shared anchors — reducing pca_dim to %d",
                    len(canonical), max(2, len(canonical) - 1))
        pca_dim = max(2, len(canonical) - 1)

    aligned_anchor_blocks: list[np.ndarray] = []
    aligned_focus_blocks: list[np.ndarray] = []
    block_meta: list[dict] = []
    ref_anchor_pca: np.ndarray | None = None

    frames_of_interest = ["bare", "anglo_en", "catalan_frame", "neutral_ca"]

    for mr in ordered:
        # Anchors in canonical order
        m_bare = (np.isin(mr.clusters, ["anchor_anglo", "anchor_catalan"])
                  & (mr.frame_ids == "bare"))
        ord_idx = []
        for w in canonical:
            hits = np.where(m_bare & (mr.concepts == w))[0]
            if len(hits):
                ord_idx.append(int(hits[0]))
        anchor_vecs = mr.vectors[ord_idx]
        anchor_clusters = mr.clusters[ord_idx]

        # Focus concept vectors per (concept, frame)
        focus_vecs = []
        focus_keys = []
        for c in focus:
            for f in frames_of_interest:
                m = ((mr.concepts == c) & (mr.frame_ids == f)
                     & (mr.clusters == "target_catalan"))
                hits = np.where(m)[0]
                if len(hits) == 0:
                    continue
                focus_vecs.append(mr.vectors[int(hits[0])])
                focus_keys.append((c, f))
        focus_vecs = (np.stack(focus_vecs) if focus_vecs
                      else np.zeros((0, anchor_vecs.shape[1])))

        combined = np.vstack([anchor_vecs, focus_vecs]) if focus_vecs.size \
            else anchor_vecs
        k = min(pca_dim, combined.shape[0] - 1, combined.shape[1])
        pca = PCA(n_components=k, random_state=SEED)
        pca_combined = pca.fit_transform(combined)
        n_a = anchor_vecs.shape[0]
        a_pca = pca_combined[:n_a]
        f_pca = pca_combined[n_a:] if focus_vecs.size else np.zeros((0, k))

        if ref_anchor_pca is None:
            ref_anchor_pca = a_pca
            a_aligned = a_pca
            f_aligned = f_pca
        else:
            # Pad / truncate to common dim
            d = ref_anchor_pca.shape[1]
            if a_pca.shape[1] < d:
                a_pca = np.pad(a_pca, ((0, 0), (0, d - a_pca.shape[1])))
                f_pca = np.pad(f_pca, ((0, 0), (0, d - f_pca.shape[1])))
            elif a_pca.shape[1] > d:
                a_pca = a_pca[:, :d]
                f_pca = f_pca[:, :d]
            R, _scale = orthogonal_procrustes(a_pca, ref_anchor_pca)
            a_aligned = a_pca @ R
            f_aligned = f_pca @ R

        aligned_anchor_blocks.append(a_aligned)
        aligned_focus_blocks.append(f_aligned)
        block_meta.append({
            "slug": mr.slug,
            "anchor_words": canonical,
            "anchor_clusters": anchor_clusters,
            "focus_keys": focus_keys,
        })

    # Combined UMAP over (per-model anchors + per-model focus). To avoid
    # double-counting anchors, average them across models first into one
    # canonical anchor cloud — Procrustes alignment makes that average
    # meaningful. The focus concept points stay per-model so we can connect
    # them.
    avg_anchors = np.mean(np.stack(aligned_anchor_blocks), axis=0)
    avg_anchor_clusters = block_meta[0]["anchor_clusters"]

    focus_points: list[tuple[str, str, str, np.ndarray]] = []
    for blk, meta in zip(aligned_focus_blocks, block_meta):
        for vec, (c, f) in zip(blk, meta["focus_keys"]):
            focus_points.append((meta["slug"], c, f, vec))

    if not focus_points:
        log.info("procrustes: no focus points — skipping")
        return

    fp_arr = np.stack([p[3] for p in focus_points])
    combined_xy_input = np.vstack([avg_anchors, fp_arr])
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.15,
                        metric="cosine", random_state=SEED)
    xy = reducer.fit_transform(combined_xy_input)
    n_anc = avg_anchors.shape[0]
    anchor_xy = xy[:n_anc]
    focus_xy = xy[n_anc:]

    fig, ax = plt.subplots(figsize=(11, 9))
    # In clean mode we still draw every anchor's cluster marker (so the reader
    # sees the overall Anglo / Catalan scaffolding) but only label the anchors
    # in `anchor_subset` — otherwise the figure becomes illegible.
    label_mask = (np.ones(len(canonical), dtype=bool) if anchor_subset is None
                  else np.array([w in set(anchor_subset) for w in canonical]))
    for cluster, color in [("anchor_anglo", "#1f77b4"),
                           ("anchor_catalan", "#2ca02c")]:
        m = avg_anchor_clusters == cluster
        scatter_alpha = 0.55 if anchor_subset is None else 0.25
        ax.scatter(anchor_xy[m, 0], anchor_xy[m, 1],
                   c=color, s=110, alpha=scatter_alpha,
                   edgecolors="white", linewidths=0.6,
                   label=cluster, zorder=2)
    for w, xy_pt, keep in zip(canonical, anchor_xy, label_mask):
        if not keep:
            continue
        ax.annotate(w, xy_pt, fontsize=8, alpha=0.85,
                    xytext=(4, 4), textcoords="offset points")

    # Focus concept polylines per (concept, frame)
    concept_colors = {"coca": "#000000", "gegant": "#7b3294", "castell": "#008837"}
    frame_styles = {"bare": ":", "anglo_en": "-", "catalan_frame": "--",
                    "neutral_ca": "-."}
    by_concept_frame: dict[tuple[str, str], list[tuple[str, np.ndarray]]] = {}
    for (slug, c, f, _vec), pt in zip(focus_points, focus_xy):
        by_concept_frame.setdefault((c, f), []).append((slug, pt))
    for (c, f), entries in by_concept_frame.items():
        entries.sort(key=lambda e: _family_version_sort_key(e[0]))
        pts = np.stack([e[1] for e in entries])
        col = concept_colors.get(c, "#444")
        ls = frame_styles.get(f, "-")
        ax.plot(pts[:, 0], pts[:, 1], color=col, linestyle=ls,
                linewidth=1.5, alpha=0.7, zorder=4)
        ax.scatter(pts[:, 0], pts[:, 1], color=col, s=80,
                   edgecolors="white", linewidths=0.6, zorder=5,
                   marker="*" if f == "anglo_en" else
                          ("D" if f == "catalan_frame" else
                           ("s" if f == "neutral_ca" else "o")))
        # Label final point with model abbreviation
        for slug, pt in entries:
            short = slug.replace("base", "").replace("-", "")[:6]
            pass
        ax.annotate(c, pts[0], fontsize=9, color=col, weight="bold",
                    xytext=(6, 6), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor=col, alpha=0.9))

    ax.set_title("Procrustes-aligned combined UMAP — small tier "
                 "(anchors averaged; focus concepts traced across models)" + title_suffix,
                 fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", frameon=False, fontsize=9)

    handles_extra = [
        plt.Line2D([0], [0], color="black", linestyle=":", linewidth=2,
                   label="bare"),
        plt.Line2D([0], [0], color="black", linestyle="-", linewidth=2,
                   label="anglo_en (★)"),
        plt.Line2D([0], [0], color="black", linestyle="-.", linewidth=2,
                   label="neutral_ca (■)"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=2,
                   label="catalan_frame (◆)"),
    ]
    ax.legend(handles=handles_extra + [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#1f77b4", markersize=10, label="Anglo anchor"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#2ca02c", markersize=10, label="Catalan anchor"),
    ], loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def median_bias_table(results: list[ModelResults]) -> pd.DataFrame:
    rows: list[dict] = []
    for mr in results:
        for frame, sub in mr.bias.groupby("frame_id"):
            rows.append({
                "model": mr.slug,
                "frame_id": frame,
                "median_bias_ratio": sub["bias_ratio"].median(),
                "n": len(sub),
            })
    return pd.DataFrame(rows).pivot(index="model", columns="frame_id",
                                     values="median_bias_ratio").round(3)


def _topk_column(df: pd.DataFrame) -> str | None:
    return next((c for c in df.columns
                 if c.startswith("top") and c.endswith("_catalan_frac")), None)


def median_bias_by_ambiguity(results: list[ModelResults]) -> pd.DataFrame:
    """One row per (model, ambiguity_type); columns = frames. Headline stratification."""
    rows: list[dict] = []
    for mr in results:
        if "ambiguity_type" not in mr.bias.columns:
            continue
        for (amb, frame), sub in mr.bias.groupby(["ambiguity_type", "frame_id"]):
            rows.append({
                "model": mr.slug,
                "ambiguity_type": amb,
                "frame_id": frame,
                "median_bias_ratio": sub["bias_ratio"].median(),
                "n": len(sub),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return (df.pivot(index=["model", "ambiguity_type"],
                     columns="frame_id",
                     values="median_bias_ratio")
              .round(3))


def coca_case_study(results: list[ModelResults]) -> pd.DataFrame:
    """Per-model top-10 KNN for 'coca' under every frame, across all models.

    This is the paper's title-case: the Catalan flatbread *coca* colliding
    with the English 'coca' (cocaine, Coca-Cola). One row per (model, frame);
    columns split anchor counts by cluster plus the ordered top-10 list.
    """
    rows: list[dict] = []
    for mr in results:
        knn_path = OUT / mr.slug / f"layer{mr.best_layer}" / "knn.csv"
        if not knn_path.exists():
            continue
        knn = pd.read_csv(knn_path)
        coca = knn[knn["concept"] == "coca"]
        if coca.empty:
            continue
        for frame, sub in coca.groupby("frame_id"):
            sub = sub.sort_values("rank").head(10)
            anglo = int((sub["neighbor_cluster"] == "anchor_anglo").sum())
            cat = int((sub["neighbor_cluster"] == "anchor_catalan").sum())
            top3 = ", ".join(sub["neighbor"].head(3).tolist())
            rows.append({
                "model": mr.slug,
                "best_layer": mr.best_layer,
                "frame_id": frame,
                "n_anglo_top10": anglo,
                "n_catalan_top10": cat,
                "top3_neighbors": top3,
            })
    return pd.DataFrame(rows)


def cross_generation_pairs(results: list[ModelResults]) -> pd.DataFrame:
    """Within-family Δ(top10_catalan_frac) between predecessor and successor.

    Hard-coded family pairs (older → newer):
        gemma2-2b  → gemma4-e2b
        qwen2-1.5b → qwen3.5-4b-base

    Positive delta = newer model places targets closer to Catalan neighborhood
    than older model did. Computed per (ambiguity_type, frame_id).
    """
    pairs = [
        ("gemma2-2b", "gemma4-e2b", "Gemma"),
        ("qwen2-1.5b", "qwen3.5-4b-base", "Qwen"),
    ]
    by_slug = {mr.slug: mr for mr in results}
    rows: list[dict] = []
    for old, new, family in pairs:
        if old not in by_slug or new not in by_slug:
            continue
        mr_old, mr_new = by_slug[old], by_slug[new]
        col_old = _topk_column(mr_old.bias)
        col_new = _topk_column(mr_new.bias)
        if col_old is None or col_new is None:
            continue
        old_agg = (mr_old.bias.groupby(["ambiguity_type", "frame_id"])[col_old]
                   .median().rename("older"))
        new_agg = (mr_new.bias.groupby(["ambiguity_type", "frame_id"])[col_new]
                   .median().rename("newer"))
        joined = pd.concat([old_agg, new_agg], axis=1).reset_index()
        joined["delta_newer_minus_older"] = (joined["newer"] - joined["older"]).round(2)
        joined.insert(0, "family", family)
        joined.insert(1, "pair", f"{old} → {new}")
        rows.append(joined)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def median_topk_catalan_by_ambiguity(results: list[ModelResults]) -> pd.DataFrame:
    """Median top10_catalan_frac per (model, ambiguity_type) × frame.

    Higher = more Catalan anchors in the top-k neighborhood. 1.0 means every
    top neighbor is Catalan; 0.0 means every top neighbor is Anglo. Captures
    the KNN signal that bias_ratio saturation around 0.5 can hide.
    """
    rows: list[dict] = []
    for mr in results:
        col = _topk_column(mr.bias)
        if col is None or "ambiguity_type" not in mr.bias.columns:
            continue
        for (amb, frame), sub in mr.bias.groupby(["ambiguity_type", "frame_id"]):
            rows.append({
                "model": mr.slug,
                "ambiguity_type": amb,
                "frame_id": frame,
                "median_topk_catalan_frac": sub[col].median(),
                "n": len(sub),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return (df.pivot(index=["model", "ambiguity_type"],
                     columns="frame_id",
                     values="median_topk_catalan_frac")
              .round(3))


def per_concept_rollup(results: list[ModelResults]) -> pd.DataFrame:
    """Average bias_ratio per concept across frames and models."""
    frames_of_interest = ["bare", "anglo_en", "catalan_frame", "neutral_ca"]
    rows: list[dict] = []
    for mr in results:
        for (concept, category), sub in mr.bias.groupby(["concept", "category"]):
            row = {"model": mr.slug, "concept": concept, "category": category}
            for f in frames_of_interest:
                vals = sub.loc[sub["frame_id"] == f, "bias_ratio"]
                row[f"bias_{f}"] = float(vals.mean()) if len(vals) else float("nan")
            row["bias_mean"] = float(sub["bias_ratio"].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def category_rollup(results: list[ModelResults]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for mr in results:
        g = mr.bias.groupby("category")["bias_ratio"].agg(["median", "count"]).reset_index()
        g.insert(0, "model", mr.slug)
        parts.append(g)
    return pd.concat(parts, ignore_index=True).rename(columns={"median": "median_bias"})


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=True) if df.index.name or df.index.names[0] else df.to_markdown(index=False)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    results = discover()
    if not results:
        log.error("no model results found under out/ — run 02 + 03 first")
        return

    headline_path = FIGS / "headline_umap.png"
    headline_figure(results, headline_path)

    med = median_bias_table(results)
    med.to_csv(OUT / "median_bias_by_model_frame.csv")

    ambig = median_bias_by_ambiguity(results)
    if not ambig.empty:
        ambig.to_csv(OUT / "median_bias_by_ambiguity.csv")

    topk_ambig = median_topk_catalan_by_ambiguity(results)
    if not topk_ambig.empty:
        topk_ambig.to_csv(OUT / "median_topk_catalan_by_ambiguity.csv")

    coca_df = coca_case_study(results)
    if not coca_df.empty:
        coca_df.to_csv(OUT / "coca_case_study.csv", index=False)

    gen_df = cross_generation_pairs(results)
    if not gen_df.empty:
        gen_df.to_csv(OUT / "cross_generation_deltas.csv", index=False)
        cross_gen_heatmap(gen_df, FIGS / "cross_generation_heatmap.png")

    # Direction (1) — paired-bootstrap CI on Δ per concept.
    delta_df = frame_differential_per_concept(results)
    if not delta_df.empty:
        delta_df.to_csv(OUT / "delta_per_concept.csv", index=False)
        frame_differential_forest(delta_df, FIGS / "delta_forest.png")

    # Direction (2) — layer-depth × frame interaction.
    depth_df = layer_depth_delta(results)
    if not depth_df.empty:
        depth_df.to_csv(OUT / "layer_depth_delta.csv", index=False)
        layer_depth_delta_plot(depth_df, FIGS / "layer_depth_delta.png")

    # Direction (3) — sub-classify the Anglo anchor cluster.
    sub_df = anglo_subcluster_pull(results, focus=["coca", "gegant", "castell"])
    if not sub_df.empty:
        sub_df.to_csv(OUT / "anglo_subcluster_pull.csv", index=False)
        for c in ["coca", "gegant", "castell"]:
            anglo_subcluster_chart(sub_df, c, FIGS / f"subcluster_{c}.png")

    # Direction (4) — stratified-bootstrap CI on ambiguity-type medians.
    strat_df = stratified_ambiguity_ci(results)
    if not strat_df.empty:
        strat_df.to_csv(OUT / "ambiguity_stratified_ci.csv", index=False)
        ambiguity_ci_forest(strat_df, FIGS / "ambiguity_stratified_ci.png")

    # Direction (5) — Procrustes cross-model alignment + combined UMAP.
    # Two versions: "clean" (body figure, only focus concepts + curated core
    # anchors labeled) and "full" (supplementary, every anchor labeled).
    procrustes_core_anchors = [
        "giant", "castle", "dragon", "beanstalk", "knight", "jester",
        "enchanted forest", "witch",
        "grallers", "diables", "cercavila", "casteller", "festa major",
        "Patum", "folklore", "Barcelona",
    ]
    procrustes_combined_umap(
        results,
        focus=["coca", "gegant", "castell"],
        out_path=FIGS / "procrustes_combined_umap.png",
        anchor_subset=procrustes_core_anchors,
        title_suffix=" — clean",
    )
    procrustes_combined_umap(
        results,
        focus=["coca", "gegant", "castell"],
        out_path=FIGS / "procrustes_combined_umap_full.png",
        title_suffix=" — full (all anchors labeled)",
    )

    # Title test — coca lexical-collision (does the noise cluster appear?).
    coca_lex = coca_lexical_collision_test(results)
    if not coca_lex.empty:
        coca_lex.to_csv(OUT / "coca_lexical_collision.csv", index=False)
        coca_lexical_chart(coca_lex, FIGS / "coca_lexical_collision.png")

    # Per-concept focus: bar charts + UMAP trajectory across the four models.
    focus_concepts = [
        ("coca",
         '"Coca Is Not Cocaine" — Anglo contamination of *coca* by frame and model'),
        ("gegant",
         "Anglo pull on *gegant* (the cross_lingual_bias archetype: gegant↔giant)"),
        ("castell",
         "Anglo pull on *castell* (cross_lingual_bias: human-tower vs medieval castle)"),
    ]
    for concept, title in focus_concepts:
        concept_anglo_bar_chart(results, concept,
                                FIGS / f"focus_{concept}_bar.png", title=title)
        concept_trajectory_panel(results, concept,
                                 FIGS / f"focus_{concept}_trajectory.png")

    roll = per_concept_rollup(results)
    roll.to_csv(OUT / "bias_per_concept.csv", index=False)

    cat = category_rollup(results)
    cat.to_csv(OUT / "bias_by_category.csv", index=False)

    mean_by_concept = (
        roll.groupby(["concept", "category"])["bias_mean"].mean().reset_index()
        .sort_values("bias_mean", ascending=False)
    )
    top_anglo = mean_by_concept.head(15).round(3)
    top_catalan = mean_by_concept.tail(5).round(3)

    lines: list[str] = []
    lines.append("# Coca Is Not Cocaine: Measuring Lexical-Cultural Collisions in "
                 "Open-Weight LLMs for Catalan\n")
    lines.append("**Status:** Preprint — not peer-reviewed.\n")
    lines.append("## Abstract\n")
    lines.append(
        "Culturally-marked Catalan concepts collide with the Anglo-Saxon imaginary "
        "in at least three distinct ways, and naive cultural-bias probes conflate "
        "them. A *gegant* is not a fairy-tale giant (it is a festive effigy), a "
        "*coca* is not cocaine (it is a flatbread), and a *pinya* at the base of a "
        "castell is neither a pinecone nor a pineapple. We extend the CAMeL "
        f"framework (Naous et al., ACL 2024) to {len(results)} open-weight base "
        "LLM(s) via contextualized hidden-state geometry — mean-pooled, "
        "L2-normalized vectors at a best-layer chosen by KNN purity over anchors — "
        "and stratify the bias ratio `pull_anglo / (pull_anglo + pull_catalan)` "
        "along an `ambiguity_type` axis. This separates (i) cross-lingual cognate "
        "drag (gegant, castell, coca), (ii) purely domain-specific representation "
        "gaps (correfoc, casteller, romesco), and (iii) intra-Catalan polysemy "
        "(pinya, tió, folre, manilles). 95% bootstrap CIs (1000 resamples) are "
        "reported per (concept, frame). The interesting finding is not the pooled "
        "median — it is which stratum moves under which framing.\n")

    lines.append("## Method\n")
    lines.append(
        "Let `v(c, f)` be the contextualized hidden-state vector of concept `c` inside "
        "probe frame `f`, computed by (i) tokenizing the probe sentence with the model's "
        "fast tokenizer, (ii) mapping the character span of `c` back to its token span "
        "via `offset_mapping`, (iii) taking mean over those subword positions at a given "
        "layer `L`, and (iv) L2-normalizing. Anchors are embedded with a bare-lemma probe.\n\n"
        "For each target `c` under frame `f`:\n\n"
        "```\n"
        "pull_anglo(c,f)   = mean_a∈ANGLO   cos( v(c,f), v(a,bare) )\n"
        "pull_catalan(c,f) = mean_a∈CATALAN cos( v(c,f), v(a,bare) )\n"
        "bias_ratio(c,f)   = pull_anglo / (pull_anglo + pull_catalan)\n"
        "```\n\n"
        "95% CIs come from 1000 bootstrap resamples over the anchor sets. Self-matches "
        "(same surface string across target and anchor lists) are excluded. We sweep "
        "three layers per model (early/mid/late) and select the best layer by KNN-5 "
        "purity over the anchor-only subset.\n")

    lines.append("## Headline figure\n")
    lines.append(f"![Headline UMAP]({headline_path.relative_to(OUT)})\n")

    lines.append("## Bias ratio stratified by ambiguity type\n")
    lines.append(
        "The headline claim depends on which *type* of concept we look at. We split "
        "targets into three disjoint classes to isolate the cause of any Anglo pull:\n\n"
        "- **cross_lingual_bias** — the Catalan word has a direct English cognate "
        "(gegant↔giant, castell↔castle, drac↔dragon, diables↔devils). A high bias "
        "ratio is consistent with both anglification AND a shared cognate meaning; "
        "the two effects are confounded here by construction.\n"
        "- **domain_specific** — no English equivalent (correfoc, casteller, "
        "cercavila, grallers). A high bias ratio here is the *cleanest* evidence for "
        "anglification: there is no lexical shortcut, so any pull toward fairy-tale "
        "vocabulary is structural.\n"
        "- **internal_polysemy** — two distinct Catalan senses, one mundane and one "
        "festive (pinya=pinecone/castell-base, tió=log/Christmas-log, falla=fault/"
        "bonfire-tower). A high bias ratio could be *either* anglification *or* "
        "the model capturing the more frequent mundane sense. The frame differential "
        "`bias(anglo_en) − bias(catalan_frame)` separates the two: a large drop under "
        "the Catalan frame implies the festive sense was retrievable and the default "
        "was polysemy, not anglification.\n\n")
    if not ambig.empty:
        lines.append(ambig.to_markdown() + "\n")
    else:
        lines.append("*(no stratified table available — run 02 + 03 with updated schema)*\n")

    lines.append("## Top-10 Catalan fraction (complementary metric)\n")
    lines.append(
        "`bias_ratio` saturates near 0.5 when raw cosines are high toward both "
        "clusters, which happens for most concepts at late layers. "
        "`top10_catalan_frac` is more discriminative: of the 10 anchors nearest "
        "(by cosine) to the target vector, what fraction are from the Catalan "
        "cluster? A value of 1.0 means the KNN neighborhood is entirely Catalan "
        "folkloric; 0.0 means it is entirely Anglo fairy-tale; 0.5 is the "
        "neutral midpoint. This is what the reader intuits when they read the "
        "per-concept KNN tables in `layer{L}/knn.csv`.\n\n")
    if not topk_ambig.empty:
        lines.append(topk_ambig.to_markdown() + "\n")
    else:
        lines.append("*(top-k Catalan fraction not available)*\n")

    lines.append("## Median bias ratio by model × frame (pooled across ambiguity types)\n")
    lines.append(med.to_markdown() + "\n")

    lines.append("## Top-15 most Anglo-pulled concepts\n")
    lines.append(top_anglo.to_markdown(index=False) + "\n")

    lines.append("## Top-5 most Catalan-pulled concepts\n")
    lines.append(top_catalan.to_markdown(index=False) + "\n")

    lines.append("## Bias by category (median across models)\n")
    cat_piv = (cat.pivot(index="category", columns="model", values="median_bias")
                  .round(3).sort_index())
    lines.append(cat_piv.to_markdown() + "\n")

    if not coca_df.empty:
        lines.append("## Case study — *coca*\n")
        lines.append(
            "The title of the paper hangs on this one word. *Coca* in Catalan is "
            "a flatbread; in English, *coca* points to the plant / drug / brand. "
            "Under the Anglo fairy-tale frame each model places the Catalan "
            "lemma among its top-10 neighbors as follows:\n\n")
        lines.append("![coca bar chart](figs/focus_coca_bar.png)\n\n")
        lines.append("![coca trajectory](figs/focus_coca_trajectory.png)\n\n")
        lines.append(
            "The trajectory plot shows where each model places the four "
            "frame-versions of *coca* relative to that model's own anchor cloud. "
            "Reading left → right is reading time: Gemma 2 → Gemma 4 → Qwen2 → "
            "Qwen 3.5. The four stars per panel are the same Catalan word "
            "embedded in four different sentences; their spread tells you how "
            "much the surrounding language can move the concept.\n\n")
        piv = (coca_df
               .pivot(index="model", columns="frame_id", values="n_anglo_top10")
               .rename_axis(columns=None)
               .reindex(columns=["bare", "anglo_en", "catalan_frame", "neutral_ca"]))
        piv.columns = [f"{c}_n_anglo" for c in piv.columns]
        lines.append("Anglo-anchor count in top-10 KNN (lower = more Catalan):\n\n")
        lines.append(piv.to_markdown() + "\n\n")
        lines.append(
            "The eye-catcher is the `anglo_en` column — *\"Once upon a time, "
            "there was a coca who lived in the enchanted forest\"*. Qwen 3.5 4B "
            "Base places **8 of 10** nearest anchors in the Anglo cluster "
            "(beanstalk, peasant, enchanted forest, witch, castle, dwarf, giant, "
            "dragon). Its own predecessor Qwen2 1.5B placed only **2 of 10**. "
            "This is not a generational improvement — on this single concept, "
            "the newer and larger Qwen is *more* English-contaminated than the "
            "older, smaller one.\n\n"
            "Top-3 neighbors per (model, frame):\n\n")
        top3_table = (coca_df
                      .pivot(index="model", columns="frame_id", values="top3_neighbors")
                      .rename_axis(columns=None)
                      .reindex(columns=["bare", "anglo_en", "catalan_frame", "neutral_ca"]))
        lines.append(top3_table.to_markdown() + "\n")

    # Per-concept focus sections for two more cross_lingual_bias archetypes.
    for concept, blurb in [
        ("gegant",
         "*gegant* is the textbook cross_lingual_bias case: the Catalan "
         "festive effigy shares its lemma's English cognate (*giant*) with a "
         "different cultural universe (medieval / fairy-tale)."),
        ("castell",
         "*castell* in Catalan today refers primarily to the human tower "
         "(castellers); its English cognate *castle* points to the medieval "
         "fortification. Whether the model retrieves the tower or the "
         "fortification under each frame is the question."),
    ]:
        bar_path = FIGS / f"focus_{concept}_bar.png"
        traj_path = FIGS / f"focus_{concept}_trajectory.png"
        if not bar_path.exists() and not traj_path.exists():
            continue
        lines.append(f"## Focus — *{concept}*\n")
        lines.append(blurb + "\n\n")
        if bar_path.exists():
            lines.append(f"![{concept} bar chart](figs/{bar_path.name})\n\n")
        if traj_path.exists():
            lines.append(f"![{concept} trajectory](figs/{traj_path.name})\n\n")

    if not gen_df.empty:
        lines.append("## Cross-generation evolution\n")
        lines.append(
            "We probed both the current and the previous base model per family: "
            "Gemma 2 2B → Gemma 4 E2B, and Qwen2 1.5B → Qwen 3.5 4B Base. The "
            "delta below is `top10_catalan_frac(newer) − top10_catalan_frac(older)`; "
            "positive (blue) = the newer model's top-10 neighborhood for this "
            "stratum and frame is more Catalan than the older model's; "
            "negative (red) = the newer model regressed.\n\n")
        lines.append("![cross-generation heatmap](figs/cross_generation_heatmap.png)\n\n")
        pivot_delta = (gen_df
                       .pivot_table(index=["family", "ambiguity_type"],
                                    columns="frame_id",
                                    values="delta_newer_minus_older")
                       .round(2))
        lines.append(pivot_delta.to_markdown() + "\n\n")
        lines.append(
            "**The two families moved in opposite directions.** Within Gemma, the "
            "English-frame placement improved (less Anglo pull) but the "
            "Catalan-frame ceiling dropped — Google's next generation is *more "
            "context-invariant*: higher floor under adversarial framing, lower "
            "ceiling under native framing. Within Qwen, the pattern flipped — "
            "the Catalan-frame placement climbed to a perfect 1.0 across every "
            "stratum, but the English-frame placement collapsed by roughly "
            "0.3 in cross_lingual_bias and internal_polysemy: Alibaba's next "
            "generation is *more context-amplified*. Neither direction is "
            "uniformly better. The right model for Catalan depends on whether "
            "the deployment can guarantee Catalan framing: if yes, Qwen 3.5 "
            "wins; if the surrounding text is mixed or untrusted, Gemma 4 is "
            "the safer choice. This also invalidates the naïve reading that "
            "\"newer is better\" or \"one family dominates the other\" — the "
            "relationship is an axis, not a ranking.\n")

    # ----- Direction sections -----
    if not depth_df.empty:
        lines.append("## Direction (2) — layer-depth × frame interaction\n")
        lines.append(
            "Picking a single best layer by KNN purity collapses the depth "
            "dimension. Computing `Δ = bias(anglo_en) − bias(catalan_frame)` "
            "at every committed layer per model shows whether the rescue "
            "happens early, mid or late in the stack.\n\n")
        lines.append("![layer-depth Δ trajectory](figs/layer_depth_delta.png)\n\n")

    coca_lex_path = FIGS / "coca_lexical_collision.png"
    if coca_lex_path.exists():
        lines.append("## Title test — does *coca* land near *cocaine*?\n")
        lines.append(
            "The paper title (\"Coca Is Not Cocaine\") promises an F1-style "
            "*lexical* collision (the Catalan word *coca* drifting toward the "
            "English drug / brand neighborhood). The headline figures only "
            "demonstrated F2 (*coca* drifting toward fairy-tale vocabulary), "
            "which left the title vulnerable to over-claiming. To validate or "
            "falsify the F1 reading we add a noise anchor cluster of "
            "{cocaine, coke, Coca-Cola, pepsi, drug, powder, addiction, soda, "
            "beverage, brand, caffeine, sugar, logo} and check the top-10 KNN "
            "of *coca* under every (model, frame) for hits in this cluster. "
            "The chart's purple bars are the test:\n\n")
        lines.append(f"![coca lexical collision](figs/{coca_lex_path.name})\n\n")
        n_noise_total = int(coca_lex["n_noise_top10"].sum())
        lines.append(
            f"Total noise hits across all (model × frame) cells: "
            f"**{n_noise_total} of {len(coca_lex)*10}**. "
            "If this number is zero, the title is fairy-tale-displacement only "
            "and should be reframed; if non-zero, the lexical collision is "
            "empirically grounded for those cells.\n\n")

    if not sub_df.empty:
        lines.append("## Direction (3) — Anglo subcluster pull on focus concepts\n")
        lines.append(
            "Splitting `anchors_anglo` into three subclusters (`fairy_tale`, "
            "`medieval`, `character`) lets us see *which* English universe "
            "actually catches the focus concepts when they drift. For each "
            "concept, one bar per (frame, model); the stack composition is "
            "the top-10 KNN composition.\n\n")
        for c in ["coca", "gegant", "castell"]:
            p = FIGS / f"subcluster_{c}.png"
            if p.exists():
                lines.append(f"![{c} Anglo subcluster pull](figs/{p.name})\n\n")

    if not strat_df.empty:
        lines.append("## Direction (4) — stratified-bootstrap CIs on ambiguity-type medians\n")
        lines.append(
            "Resampling concepts within each `ambiguity_type` stratum gives a "
            "CI on each cell of the model × ambiguity_type × frame medians "
            "table. `cross_lingual_bias` has only 9 concepts: its medians "
            "carry the widest CIs by design and the widest CIs are exactly "
            "where one should be most cautious about reading the headline.\n\n")
        lines.append("![stratified-bootstrap CI forest](figs/ambiguity_stratified_ci.png)\n\n")

    procrustes_path = FIGS / "procrustes_combined_umap.png"
    if procrustes_path.exists():
        lines.append("## Direction (5) — Procrustes cross-model alignment\n")
        lines.append(
            "Each small-tier model is PCA-reduced to 30 dimensions and then "
            "orthogonally aligned to Gemma 2 2B using the shared bare-anchor "
            "set. After alignment the per-model anchors are averaged into a "
            "canonical cloud; the focus concepts (coca, gegant, castell) keep "
            "their per-model identity so we can connect them with a polyline "
            "across the four small-tier models in chronological order. "
            "Big-tier models are excluded — their NPZ is off-disk on RunPod.\n\n")
        lines.append(f"![Procrustes combined UMAP](figs/{procrustes_path.name})\n\n")

    if not delta_df.empty:
        lines.append("## Direction (1) — paired Δ per concept with bootstrap CI\n")
        lines.append(
            "For every model at its best layer we compute "
            "`Δ(c) = bias(c, anglo_en) − bias(c, catalan_frame)` and a 95% CI on "
            "Δ. On the small tier the CI is a proper paired bootstrap over the "
            "anchor pool (resamples are shared between the two frames before "
            "differencing), surfaced from `out/{slug}/layer{L}/delta.csv`. On the "
            "large tier the original NPZ embeddings are off-repo (gitignored, "
            "stayed on the RunPod box), so the CI falls back to a *conservative* "
            "interval derived from the marginal bootstrap CIs on each frame's "
            "`bias_ratio` (`Δ_lo = bias_a_lo − bias_b_hi`, "
            "`Δ_hi = bias_a_hi − bias_b_lo`) — that is, it pretends the two "
            "frames' anchor noise is independent, which it is not, so it "
            "overstates uncertainty. Concepts marked ★ have a CI that excludes "
            "zero even under that conservative bound: a robust sign that the "
            "Catalan frame really does change the geometry of that concept "
            "relative to the Anglo frame.\n\n")
        lines.append("![paired Δ forest plot](figs/delta_forest.png)\n\n")
        # Cross-model summary: how many robust concepts per model
        sum_rows: list[dict] = []
        for slug in sorted(delta_df["model"].unique(), key=_family_version_sort_key):
            sub = delta_df[delta_df["model"] == slug]
            sum_rows.append({
                "model": slug,
                "median_delta": round(float(sub["delta"].median()), 3),
                "n_robust": int(sub["ci_excludes_zero"].fillna(False).sum()),
                "n_total": len(sub),
                "ci_kind": sub["ci_kind"].iloc[0] if "ci_kind" in sub.columns else "?",
            })
        lines.append(pd.DataFrame(sum_rows).to_markdown(index=False) + "\n\n")

    lines.append("## Discussion\n")
    lines.append(
        "Read the stratified table first. The headline claim of *Jack and the Gegant* "
        "is that heavy English-language pretraining pulls culturally-marked Catalan "
        "concepts toward the Anglo fairy-tale/medieval neighborhood. The cleanest "
        "evidence lives in the `domain_specific` row: if cercavila, correfoc, "
        "panellet, casteller still sit closer to *giant / jester / enchanted forest* "
        "than to *festa major / grallers / diables* — even under a native Catalan "
        "frame — the pull is structural, not prompt-rescuable, and not explained by "
        "any lexical cognate.\n\n"
        "`cross_lingual_bias` items (gegant, castell, drac, diables) are expected to "
        "show the highest raw bias ratios, but that number is confounded with the "
        "cognate meaning of the Catalan word itself. Don't read them as the "
        "headline — read them as the *upper bound* of how strong the pull can get "
        "when the lexical shortcut is available.\n\n"
        "`internal_polysemy` items (pinya, tió, tronc, falla) are a *different* "
        "story: a high bias ratio on `bare` or `neutral_ca` frames most likely "
        "reflects the model capturing the dominant non-festive Catalan sense "
        "(pinecone, log, geological fault) — *not* an English-language arrow. The "
        "interesting datum is the frame differential: does `catalan_frame` pull the "
        "vector back into the festive neighborhood? If yes, the festive sense is "
        "retrievable and we were seeing polysemy; if no, even the explicit Catalan "
        "festive context can't override the default sense.\n\n"
        "The `anglo_en` ↔ `catalan_frame` differential is therefore the key "
        "diagnostic:\n\n"
        "- Large drop in `bias_ratio` under `catalan_frame` → prompt-rescuable; the "
        "representation *does* carry the festive sense, it just wasn't the default.\n"
        "- Small / no drop → the Catalan frame cannot override the English anchor; "
        "this is the structural anglification result.\n")

    lines.append("## Limitations\n")
    lines.append(
        "- Mean-pooling over subwords smears the representation of multi-token concepts.\n"
        "- Anchor lists are hand-curated — results are sensitive to that curation.\n"
        "- Cross-model cosine comparisons are not valid; only `bias_ratio` is normalized.\n"
        "- Only base (non-RLHF) models are probed; instruct variants likely mask the bias.\n"
        "- No cross-lingual rotation / Procrustes alignment: we're reading within-model "
        "geometry, not a shared space.\n")

    lines.append("## References\n")
    lines.append(
        "- Naous, Ryan, Ritter & Xu (2024). *Having Beer after Prayer? Measuring Cultural "
        "Bias in Large Language Models*. ACL 2024.\n"
        "- Tao, Viberg, Baker & Kizilcec (2024). *Cultural Bias and Cultural Alignment of "
        "Large Language Models*. PNAS Nexus.\n"
        "- Li et al. (2023). *GeoMLAMA: Geographical and Cultural Biases in Multilingual "
        "LMs*.\n"
        "- Rystrøm, Kirk & Hale (2025). *Multilingual ≠ multicultural*. arXiv:2502.16534.\n"
        "- Wang et al. (2023). *Not all countries celebrate Thanksgiving*.\n"
        "- Bolukbasi et al. (2016). *Man is to computer programmer as woman is to "
        "homemaker*.\n"
        "- Ada Lovelace Institute (2025). *Tokenising culture*.\n")

    report_path = OUT / "report.md"
    report_path.write_text("\n".join(lines))
    log.info("wrote %s", report_path)


if __name__ == "__main__":
    main()
