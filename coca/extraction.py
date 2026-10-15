"""Hidden-state extraction: forward pass, concept-span mean pooling, L2 norm.

This is the code path of ``scripts/02_extract_embeddings.py`` factored out so
that downstream projects can import it (pinned by tag) instead of copying it.
The numerics are unchanged: bf16 forward pass with ``output_hidden_states``,
mean over the concept's token span, cast to float32, L2-normalised.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

import numpy as np
import torch

log = logging.getLogger(__name__)


def slugify(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-").lower()


def resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def find_concept_span(full_text: str, concept: str) -> tuple[int, int]:
    """Character span of the first occurrence of ``concept`` (Coca v1 behaviour)."""
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


# Round-trip probes: spaces, accents, apostrophes, punctuation, Catalan and English.
ROUNDTRIP_TEXTS = (
    "A la festa major del poble, els gegants van sortir a la plaça.",
    "Once upon a time, there was a giant who lived in the enchanted forest.",
    "L'àvia va dir: coca de recapte, pa amb tomàquet i calçots!",
)


def _norm(s: str) -> str:
    return " ".join(s.split())


def tokenizer_roundtrip_ok(tok, texts=ROUNDTRIP_TEXTS) -> bool:
    """decode(encode(text)) must give back the text (special tokens skipped)."""
    return all(_norm(tok.decode(tok(t)["input_ids"], skip_special_tokens=True)) == _norm(t)
               for t in texts)


def _tokenizer_from_json(model_id: str, revision: str | None):
    """Rebuild the fast tokenizer straight from the repo's tokenizer.json.

    transformers 5.x converts some ``LlamaTokenizerFast`` repos with a byte-level
    BPE ``tokenizer.json`` (e.g. deepseek-llm-67b-base) into a Metaspace
    ``LlamaTokenizer`` that drops every space, so the model sees
    "Alafestamajor...". The original tokenizer.json is correct.
    """
    import json

    from huggingface_hub import hf_hub_download
    from tokenizers import processors
    from transformers import PreTrainedTokenizerFast

    cfg = json.load(open(hf_hub_download(model_id, "tokenizer_config.json", revision=revision)))

    def tok_str(x):
        return x.get("content") if isinstance(x, dict) else x

    tok = PreTrainedTokenizerFast(
        tokenizer_file=hf_hub_download(model_id, "tokenizer.json", revision=revision),
        bos_token=tok_str(cfg.get("bos_token")), eos_token=tok_str(cfg.get("eos_token")),
        unk_token=tok_str(cfg.get("unk_token")), pad_token=tok_str(cfg.get("pad_token")),
    )
    if cfg.get("add_bos_token") and tok.bos_token:
        bos = (tok.bos_token, tok.bos_token_id)
        tok.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{bos[0]} $A", pair=f"{bos[0]} $A {bos[0]} $B", special_tokens=[bos])
    return tok


def load_model(model_id: str, revision: str | None = None, dtype: str = "bf16",
               device_map: str = "auto", trust_remote_code: bool = True):
    """Fast tokenizer + causal LM in eval mode, as Coca loads them.

    Since v1.1.0 the tokenizer must pass a round-trip check; if the one built by
    ``AutoTokenizer`` fails it, it is rebuilt from ``tokenizer.json``, and if that
    also fails loading aborts. ``tok.coca_tokenizer_source`` records which one
    was used ("auto" or "tokenizer.json"). Models whose AutoTokenizer passes are
    loaded exactly as in v1.0.0.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True,
                                        trust_remote_code=trust_remote_code)
    source = "auto"
    if not tokenizer_roundtrip_ok(tok):
        log.warning("%s: AutoTokenizer (%s) fails the round trip; rebuilding from "
                    "tokenizer.json", model_id, type(tok).__name__)
        tok = _tokenizer_from_json(model_id, revision)
        source = "tokenizer.json"
        if not tokenizer_roundtrip_ok(tok):
            raise RuntimeError(f"{model_id}: no tokenizer passes the round-trip check")
    tok.coca_tokenizer_source = source
    if not tok.is_fast:
        raise RuntimeError("need a fast tokenizer for offset mapping; got slow tokenizer")
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=resolve_dtype(dtype), device_map=device_map,
        trust_remote_code=trust_remote_code, output_hidden_states=True,
    )
    model.eval()
    return tok, model


def extract_hidden_states(
    tok,
    model,
    texts: Sequence[str],
    spans: Sequence[tuple[int, int]],
    layers: Sequence[int],
    max_length: int = 256,
) -> tuple[dict[int, np.ndarray], list[dict]]:
    """Mean-pooled, L2-normalised hidden states of ``texts[i][spans[i]]`` at ``layers``.

    ``spans`` are character spans of the target inside each text; pass them
    explicitly (``find_concept_span`` reproduces Coca v1, which takes the first
    substring match). One text per forward pass, as in Coca (batch size 1), so no
    padding enters the computation.

    Returns ``({layer: (n, d) float32}, [{"t_start", "t_end", "n_subwords",
    "token_ids"} per text])``.
    """
    per_layer: dict[int, list[np.ndarray]] = {L: [] for L in layers}
    info: list[dict] = []
    with torch.inference_mode():
        for text, (c0, c1) in zip(texts, spans, strict=True):
            enc = tok([text], return_offsets_mapping=True, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length, add_special_tokens=True)
            offsets = enc.pop("offset_mapping")[0].tolist()
            t0, t1 = char_span_to_token_span(offsets, c0, c1)
            inputs = {k: v.to(model.device) for k, v in enc.items()}
            hs = model(**inputs, output_hidden_states=True, use_cache=False).hidden_states
            for L in layers:
                vec = hs[L][0, t0:t1, :].float().mean(dim=0)
                vec = vec / vec.norm().clamp(min=1e-12)
                per_layer[L].append(vec.detach().cpu().numpy().astype(np.float32))
            info.append({"t_start": t0, "t_end": t1, "n_subwords": t1 - t0,
                         "token_ids": enc["input_ids"][0, t0:t1].tolist()})
    return {L: np.stack(v) for L, v in per_layer.items()}, info
