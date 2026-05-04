#!/usr/bin/env python3
"""Evaluate H6.a/b/c/d falsifiers on the Catalan-aware (Salamandra) tier.

Reads each Salamandra model's best layer's `bias.csv`, `delta.csv`, and
`knn.csv` (the latter for the *coca* H6.d test) and prints a
machine-readable verdict per sub-hypothesis.

Run:
    uv run python scripts/eval_h6.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"

SALAMANDRA = ["salamandra-2b", "salamandra-7b", "ALIA-40b"]
GENPURPOSE = ["gemma2-2b", "gemma4-e2b", "qwen2-1.5b", "qwen3.5-4b-base",
              "gemma4-26b-a4b", "qwen3.6-35b-a3b",
              "mistral-small-24b", "deepseek-llm-67b"]
FRAMES = ["bare", "anglo_en", "catalan_frame", "neutral_ca"]

H6A_THRESHOLD = 0.85
F_H6A_FLOOR = 0.70
H6B_SAT_FRAC = 0.80
H6C_BAND = 0.02
F_H6C_BAND = 0.05


def best_layer(slug: str) -> int:
    with (OUT / slug / "best_layer.json").open() as f:
        return int(json.load(f)["best_layer"])


def load_bias(slug: str) -> pd.DataFrame:
    L = best_layer(slug)
    df = pd.read_csv(OUT / slug / f"layer{L}" / "bias.csv")
    df["model"] = slug
    df["best_layer"] = L
    return df


def load_delta(slug: str) -> pd.DataFrame:
    L = best_layer(slug)
    p = OUT / slug / f"layer{L}" / "delta.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["model"] = slug
    df["best_layer"] = L
    return df


def load_knn(slug: str) -> pd.DataFrame:
    L = best_layer(slug)
    df = pd.read_csv(OUT / slug / f"layer{L}" / "knn.csv")
    df["model"] = slug
    df["best_layer"] = L
    return df


def h6a(bias: pd.DataFrame) -> dict:
    """All Salamandra models reach median top10_catalan_frac >= 0.85 on F2 across all 4 frames."""
    f2 = bias[bias["ambiguity_type"] == "domain_specific"]
    piv = (f2.groupby(["model", "frame_id"])["top10_catalan_frac"]
             .median().unstack().reindex(SALAMANDRA)[FRAMES])
    pass_h6a = (piv >= H6A_THRESHOLD).all().all()
    f_h6a_triggered = (piv < F_H6A_FLOOR).any().any()
    failing_cells = [
        (m, f, float(piv.loc[m, f]))
        for m in piv.index for f in piv.columns
        if piv.loc[m, f] < H6A_THRESHOLD
    ]
    floor_cells = [
        (m, f, float(piv.loc[m, f]))
        for m in piv.index for f in piv.columns
        if piv.loc[m, f] < F_H6A_FLOOR
    ]
    return {
        "pivot_F2_median_top10ca": piv.round(3).to_dict(),
        "H6.a (>=0.85 on all 12 cells)": "PASS" if pass_h6a else "FAIL",
        "cells_below_0.85": failing_cells,
        "F-H6.a (any cell <0.7)": "TRIGGERED" if f_h6a_triggered else "NOT TRIGGERED",
        "cells_below_0.7": floor_cells,
    }


def h6b(bias: pd.DataFrame) -> dict:
    """ALIA-40b reaches top10_catalan_frac = 1.0 on >=80% of F1 (cross_lingual_bias) under bare."""
    res = {}
    for m in SALAMANDRA:
        sub = bias[(bias["model"] == m) &
                   (bias["ambiguity_type"] == "cross_lingual_bias") &
                   (bias["frame_id"] == "bare")]
        n = len(sub)
        sat = float((sub["top10_catalan_frac"] == 1.0).mean()) if n else float("nan")
        res[m] = {"n_F1_bare": int(n), "frac_saturating_1.0": round(sat, 3)}
    alia = res.get("ALIA-40b", {})
    pass_h6b = alia.get("frac_saturating_1.0", 0.0) >= H6B_SAT_FRAC
    return {
        "per_model": res,
        "H6.b (ALIA-40b >=80% saturating on F1 bare)": "PASS" if pass_h6b else "FAIL",
        "F-H6.b (ALIA-40b does not saturate)": "TRIGGERED" if not pass_h6b else "NOT TRIGGERED",
    }


def h6c(deltas: pd.DataFrame, gp_deltas: pd.DataFrame) -> dict:
    """Median paired Δ (anglo_en − catalan_frame) on Salamandra is |median|<0.02 with 95% CI ⊃ 0,
    versus general-purpose tier in [+0.02, +0.10]."""
    if deltas.empty:
        return {"error": "no delta.csv on Salamandra tier"}
    sal = deltas.groupby("model")["delta"].agg(["median", "count"]).round(3)
    sal_med = float(deltas["delta"].median())
    sal_ci_lo = float(deltas["delta"].quantile(0.025))
    sal_ci_hi = float(deltas["delta"].quantile(0.975))
    gp_summary = (gp_deltas.groupby("model")["delta"].median().round(3).to_dict()
                  if not gp_deltas.empty else {})
    pass_h6c = abs(sal_med) < H6C_BAND and sal_ci_lo <= 0 <= sal_ci_hi
    f_h6c_triggered = abs(sal_med) > F_H6C_BAND
    return {
        "salamandra_per_model_delta_median": sal.to_dict(),
        "salamandra_pooled_delta_median": round(sal_med, 3),
        "salamandra_pooled_delta_95pct_quantile_band": [round(sal_ci_lo, 3), round(sal_ci_hi, 3)],
        "general_purpose_per_model_delta_median": gp_summary,
        "H6.c (|sal median Δ|<0.02 with band ⊃ 0)": "PASS" if pass_h6c else "FAIL",
        "F-H6.c (|sal median Δ|>0.05)": "TRIGGERED" if f_h6c_triggered else "NOT TRIGGERED",
    }


def h6d(knn: pd.DataFrame) -> dict:
    """No noise anchor (cocaine, coke, Coca-Cola, ...) in top-10 of coca on any Salamandra
    model under any frame."""
    coca = knn[knn["concept"] == "coca"].copy()
    if coca.empty:
        return {"error": "no coca rows in knn"}
    cells = (coca[coca["neighbor_cluster"] == "anchor_noise"]
             .groupby(["model", "frame_id"]).size().to_dict())
    n_cells_with_noise = len(cells)
    pass_h6d = n_cells_with_noise == 0
    return {
        "noise_hits_per_cell": {f"{m}/{f}": int(n) for (m, f), n in cells.items()},
        "n_cells_with_noise (out of 12)": n_cells_with_noise,
        "H6.d (max n_noise = 0 across 12 cells)": "PASS" if pass_h6d else "FAIL",
        "F-H6.d (>=1 noise hit on any Salamandra)": "TRIGGERED" if not pass_h6d else "NOT TRIGGERED",
    }


def main() -> None:
    avail_sal = [m for m in SALAMANDRA if (OUT / m / "best_layer.json").exists()]
    avail_gp = [m for m in GENPURPOSE if (OUT / m / "best_layer.json").exists()]
    print(f"Salamandra models with artifacts: {avail_sal}")
    print(f"General-purpose models with artifacts: {avail_gp}")
    print()

    sal_bias = pd.concat([load_bias(m) for m in avail_sal], ignore_index=True)
    sal_delta = pd.concat([load_delta(m) for m in avail_sal], ignore_index=True)
    sal_knn = pd.concat([load_knn(m) for m in avail_sal], ignore_index=True)
    gp_delta = pd.concat([load_delta(m) for m in avail_gp], ignore_index=True)

    verdict = {
        "best_layers": {m: best_layer(m) for m in avail_sal},
        "H6.a": h6a(sal_bias),
        "H6.b": h6b(sal_bias),
        "H6.c": h6c(sal_delta, gp_delta),
        "H6.d": h6d(sal_knn),
    }

    out_path = OUT / "h6_verdict.json"
    with out_path.open("w") as f:
        json.dump(verdict, f, indent=2, default=str)

    print(json.dumps(verdict, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
