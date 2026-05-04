#!/usr/bin/env python3
"""Assemble ``data/probes.csv`` from concepts + anchors.

For each target Catalan concept, four probe frames are emitted:

* ``bare``           — just the lemma (``"C"``)
* ``anglo_en``       — English fairy-tale frame with the **Catalan lemma
  embedded verbatim**. Even for en-cognate words (gegant, castell, drac), we do
  NOT substitute the English translation: that would measure the English word's
  geometry, not the cross-lingual pull of the Catalan subword mean-pool, and it
  would collide literally with the Anglo anchor set.
* ``catalan_frame``  — native Catalan festive sentence.
* ``neutral_ca``     — minimal Catalan sentence.

For each anchor (Anglo or Catalan cluster) a single ``bare`` probe is emitted so
anchors share the same embedding pipeline as targets.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ANGLO_FRAME_TMPL = (
    "Once upon a time, there was a {token} who lived in the enchanted forest."
)
CATALAN_FRAME_TMPL = (
    "A la festa major del poble, els {token} van sortir a la plaça "
    "acompanyats dels grallers."
)
NEUTRAL_CA_TMPL = "El {token} és un element important."


def build_target_frames(row: pd.Series) -> list[dict]:
    concept = str(row["concept"])

    base = {
        "cluster": "target_catalan",
        "category": row["category"],
        "ambiguity_type": row["ambiguity_type"],
    }
    return [
        {
            **base,
            "concept": concept,
            "frame_id": "bare",
            "frame_lang": "none",
            "full_text": concept,
        },
        {
            **base,
            "concept": concept,
            "frame_id": "anglo_en",
            "frame_lang": "en",
            "full_text": ANGLO_FRAME_TMPL.format(token=concept),
        },
        {
            **base,
            "concept": concept,
            "frame_id": "catalan_frame",
            "frame_lang": "ca",
            "full_text": CATALAN_FRAME_TMPL.format(token=concept),
        },
        {
            **base,
            "concept": concept,
            "frame_id": "neutral_ca",
            "frame_lang": "ca",
            "full_text": NEUTRAL_CA_TMPL.format(token=concept),
        },
    ]


def build_anchor_frames(df: pd.DataFrame, cluster: str) -> list[dict]:
    return [
        {
            "concept": str(r["anchor"]),
            "frame_id": "bare",
            "frame_lang": "none",
            "full_text": str(r["anchor"]),
            "cluster": cluster,
            "category": "anchor",
            "ambiguity_type": "anchor",
        }
        for _, r in df.iterrows()
    ]


def main() -> None:
    concepts = pd.read_csv(DATA / "concepts_ca.csv")
    anglo = pd.read_csv(DATA / "anchors_anglo.csv")
    catalan = pd.read_csv(DATA / "anchors_catalan.csv")
    noise_path = DATA / "anchors_noise.csv"
    noise = pd.read_csv(noise_path) if noise_path.exists() else pd.DataFrame()

    rows: list[dict] = []
    for _, r in concepts.iterrows():
        rows.extend(build_target_frames(r))
    rows.extend(build_anchor_frames(anglo, cluster="anchor_anglo"))
    rows.extend(build_anchor_frames(catalan, cluster="anchor_catalan"))
    if not noise.empty:
        rows.extend(build_anchor_frames(noise, cluster="anchor_noise"))

    probes = pd.DataFrame(rows, columns=[
        "concept", "frame_id", "frame_lang", "full_text",
        "cluster", "category", "ambiguity_type",
    ])

    dup = probes.duplicated(subset=["concept", "frame_id", "cluster"], keep=False)
    if dup.any():
        log.warning("duplicate probe keys detected:\n%s", probes[dup])

    missing = probes[~probes.apply(lambda r: r["concept"] in r["full_text"], axis=1)]
    if not missing.empty:
        log.warning("concept not found verbatim in frame for %d rows:\n%s",
                    len(missing), missing[["concept", "frame_id", "full_text"]])

    out_path = DATA / "probes.csv"
    probes.to_csv(out_path, index=False)
    log.info("wrote %d probes → %s", len(probes), out_path)
    log.info("cluster counts: %s", probes["cluster"].value_counts().to_dict())
    log.info("frame counts:   %s", probes["frame_id"].value_counts().to_dict())


if __name__ == "__main__":
    main()
