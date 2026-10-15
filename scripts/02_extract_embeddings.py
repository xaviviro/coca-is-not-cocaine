#!/usr/bin/env python3
"""Extract contextualized hidden-state embeddings for every probe row.

For each probe, a forward pass is run with ``output_hidden_states=True`` and the
token span corresponding to the target ``concept`` (mapped via the fast
tokenizer's ``offset_mapping``) is mean-pooled and L2-normalized at each
requested layer.

Outputs (per model):
  out/{slug}/embeddings_layer{L}.npz      vectors + aligned metadata arrays
  out/{slug}/metadata.parquet             probe-level metadata (shared sidecar)
  out/{slug}/run_meta.json                HF revision, dtype, versions, seed

Usage
-----
    uv run python scripts/02_extract_embeddings.py \\
        --model google/gemma-4-E2B --layers 6,15,22 --out out/gemma4-e2b
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coca.extraction import (  # noqa: E402
    extract_hidden_states,
    find_concept_span,
    load_model,
    slugify,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SEED = 42


@dataclass(frozen=True)
class Probe:
    row_id: int
    concept: str
    frame_id: str
    frame_lang: str
    full_text: str
    cluster: str
    category: str
    ambiguity_type: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Hugging Face model id.")
    p.add_argument("--layers", required=True, help="Comma-separated layer indices.")
    p.add_argument("--out", required=True, help="Output directory for this model.")
    p.add_argument("--probes", default=str(DATA / "probes.csv"))
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size for the forward pass (default 1 — safest for offsets).")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device-map", default="auto")
    p.add_argument("--revision", default=None, help="Pin HF revision (optional).")
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def load_probes(path: Path) -> list[Probe]:
    df = pd.read_csv(path)
    required = {
        "concept", "frame_id", "frame_lang", "full_text",
        "cluster", "category", "ambiguity_type",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"probes.csv missing columns: {missing}")
    return [
        Probe(
            row_id=i,
            concept=str(r.concept),
            frame_id=str(r.frame_id),
            frame_lang=str(r.frame_lang),
            full_text=str(r.full_text),
            cluster=str(r.cluster),
            category=str(r.category),
            ambiguity_type=str(r.ambiguity_type),
        )
        for i, r in df.reset_index(drop=True).iterrows()
    ]


def batched(seq: list[Probe], n: int) -> list[list[Probe]]:
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    layers = sorted({int(x) for x in args.layers.split(",") if x.strip()})
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading model: %s  dtype=%s  device_map=%s", args.model, args.dtype,
             args.device_map)
    tok, model = load_model(args.model, revision=args.revision, dtype=args.dtype,
                            device_map=args.device_map,
                            trust_remote_code=args.trust_remote_code)

    num_hidden = getattr(model.config, "num_hidden_layers", None)
    if num_hidden is not None:
        log.info("num_hidden_layers=%s  requested layers=%s", num_hidden, layers)
        for L in layers:
            if L > num_hidden:
                log.warning("requested layer %d exceeds num_hidden_layers=%d", L, num_hidden)

    probes = load_probes(Path(args.probes))
    log.info("loaded %d probes from %s", len(probes), args.probes)

    if args.batch_size != 1:
        log.warning("batch size is fixed at 1 (as in every Coca run); ignoring --batch-size")
    t0 = time.time()
    spans = [find_concept_span(p.full_text, p.concept) for p in probes]
    vectors, info = extract_hidden_states(tok, model, [p.full_text for p in probes], spans,
                                          layers)
    meta_rows = [
        {
            "row_id": p.row_id,
            "concept": p.concept,
            "frame_id": p.frame_id,
            "frame_lang": p.frame_lang,
            "full_text": p.full_text,
            "cluster": p.cluster,
            "category": p.category,
            "ambiguity_type": p.ambiguity_type,
            "t_start": inf["t_start"],
            "t_end": inf["t_end"],
            "n_subwords": inf["n_subwords"],
        }
        for p, inf in zip(probes, info, strict=True)
    ]
    log.info("extracted %d probes in %.1fs", len(probes), time.time() - t0)

    meta = pd.DataFrame(meta_rows)
    meta_path = out_dir / "metadata.parquet"
    meta.to_parquet(meta_path, index=False)
    log.info("wrote metadata → %s  (%d rows)", meta_path, len(meta))

    concepts = meta["concept"].to_numpy()
    frame_ids = meta["frame_id"].to_numpy()
    clusters = meta["cluster"].to_numpy()
    categories = meta["category"].to_numpy()
    ambiguity_types = meta["ambiguity_type"].to_numpy()
    row_ids = meta["row_id"].to_numpy()

    for L in layers:
        arr = vectors[L]
        out_path = out_dir / f"embeddings_layer{L}.npz"
        np.savez_compressed(
            out_path,
            vectors=arr,
            concepts=concepts,
            frame_ids=frame_ids,
            clusters=clusters,
            categories=categories,
            ambiguity_types=ambiguity_types,
            row_ids=row_ids,
            layer=np.array([L]),
        )
        log.info("wrote %s  shape=%s", out_path, arr.shape)

    run_meta = {
        "model": args.model,
        "revision": args.revision,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "layers": layers,
        "num_hidden_layers": num_hidden,
        "seed": SEED,
        "batch_size": 1,
        "n_probes": len(probes),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tokenizer_class": type(tok).__name__,
        "model_slug": slugify(args.model),
    }
    with (out_dir / "run_meta.json").open("w") as f:
        json.dump(run_meta, f, indent=2)
    log.info("wrote run_meta.json")

    log.info("done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
