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
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def slugify(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-").lower()


def resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def find_concept_span(full_text: str, concept: str) -> tuple[int, int]:
    start = full_text.find(concept)
    if start < 0:
        raise ValueError(f"concept {concept!r} not found in: {full_text!r}")
    end = start + len(concept)
    if full_text.find(concept, end) >= 0:
        log.warning("concept %r appears more than once in %r — using first occurrence",
                    concept, full_text)
    return start, end


def char_span_to_token_span(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int]:
    t_start: int | None = None
    t_end: int | None = None
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue  # special tokens / padding
        if e <= char_start:
            continue
        if s >= char_end:
            break
        if t_start is None:
            t_start = i
        t_end = i + 1
    if t_start is None or t_end is None:
        raise RuntimeError(
            f"could not map char span [{char_start}, {char_end}) to tokens via offsets {offsets}"
        )
    return t_start, t_end


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

    log.info("loading tokenizer: %s", args.model)
    tok = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if not tok.is_fast:
        raise RuntimeError("need a fast tokenizer for offset mapping; got slow tokenizer")

    dtype = resolve_dtype(args.dtype)
    log.info("loading model: %s  dtype=%s  device_map=%s", args.model, dtype, args.device_map)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        output_hidden_states=True,
    )
    model.eval()

    num_hidden = getattr(model.config, "num_hidden_layers", None)
    if num_hidden is not None:
        log.info("num_hidden_layers=%s  requested layers=%s", num_hidden, layers)
        for L in layers:
            if L > num_hidden:
                log.warning("requested layer %d exceeds num_hidden_layers=%d", L, num_hidden)

    probes = load_probes(Path(args.probes))
    log.info("loaded %d probes from %s", len(probes), args.probes)

    # Pre-allocate per-layer vector storage once we know hidden size (after first pass).
    per_layer_vecs: dict[int, list[np.ndarray]] = {L: [] for L in layers}
    meta_rows: list[dict] = []

    t0 = time.time()
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    with torch.inference_mode():
        for bi, batch in enumerate(batched(probes, args.batch_size)):
            texts = [p.full_text for p in batch]
            enc = tok(
                texts,
                return_offsets_mapping=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                add_special_tokens=True,
            )
            offsets_batch = enc.pop("offset_mapping")

            device_inputs = {k: v.to(model.device) for k, v in enc.items()}
            outputs = model(**device_inputs, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states  # tuple[len = num_hidden + 1]

            for j, probe in enumerate(batch):
                char_start, char_end = find_concept_span(probe.full_text, probe.concept)
                offsets = offsets_batch[j].tolist()
                t_start, t_end = char_span_to_token_span(offsets, char_start, char_end)

                for L in layers:
                    vec = hidden_states[L][j, t_start:t_end, :].float().mean(dim=0)
                    norm = vec.norm().clamp(min=1e-12)
                    vec = (vec / norm).detach().cpu().numpy().astype(np.float32)
                    per_layer_vecs[L].append(vec)

                meta_rows.append({
                    "row_id": probe.row_id,
                    "concept": probe.concept,
                    "frame_id": probe.frame_id,
                    "frame_lang": probe.frame_lang,
                    "full_text": probe.full_text,
                    "cluster": probe.cluster,
                    "category": probe.category,
                    "ambiguity_type": probe.ambiguity_type,
                    "t_start": t_start,
                    "t_end": t_end,
                    "n_subwords": t_end - t_start,
                })

            if (bi + 1) % 20 == 0 or (bi + 1) == len(batched(probes, args.batch_size)):
                log.info("  batch %d / %d  elapsed=%.1fs", bi + 1,
                         len(batched(probes, args.batch_size)), time.time() - t0)

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
        arr = np.stack(per_layer_vecs[L], axis=0)
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
        "batch_size": args.batch_size,
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
