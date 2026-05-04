# Limitations and pre-registered next steps

> What this study does **not** claim, what would strengthen it,
> and what is queued for future revisions of the lexicon and panel.

---

## Methodological caveats

- **Mean-pooling over subwords** smears the representation of
  multi-token concepts (e.g.\ *caga tió*, *pa amb tomàquet*,
  *rabassa morta*). The token-span span identified by
  `offset_mapping` is correct, but averaging within it loses
  positional information. Layer choice partly compensates (deep
  layers integrate context across the span), but a per-subword
  vector probe is queued for a future revision.

- **Hand-curated anchor lists.** The Anglo, Catalan and noise
  clusters are 22, 23 and 13 entries respectively; results are
  sensitive to that curation. Direction (3) of the rigour suite
  sub-classifies the Anglo cluster (fairy_tale / medieval /
  character) so the per-concept analysis can pinpoint *which*
  English neighbourhood catches each focus concept; the noise
  cluster (drug / brand / beverage) was added specifically for the
  title test.

- **Cross-model raw cosines are not directly comparable.** The
  latent spaces are not aligned across models. The only
  within-study comparisons that survive that constraint are the
  normalised `bias_ratio`, the rank-based
  `top10_catalan_frac`, and the Procrustes alignment of
  Section 8.5 of [`methodology.md`](methodology.md) (small-tier
  only, by design — cross-tier alignment requires methods that
  handle dimension-mismatched spaces and MoE expert routing).

- **Layer-selection circularity.** The selection rule shares its
  anchor pool with the metrics computed on the chosen layer.
  Mitigated by reporting every metric across all three committed
  layers per model (see `out/{slug}/layer_selection.csv` and the
  layer-depth × frame plot). Section 8.1 also added the paired
  bootstrap CI per concept so the headline numbers come with
  explicit uncertainty rather than point estimates. A
  leakage-free variant that splits each anchor cluster into a
  select-half and a metric-half is straightforward and is
  queued for a future revision.

- **Quantization is excluded a priori.** fp8 / int8 / NVFP4
  reorganises the very representation geometry the metrics
  measure (see Dettmers et al. 2022). The methodology requires
  bf16 on every model in the panel — that is what excludes
  DeepSeek-V3 (671B) and DeepSeek-V4-Flash (584B) from this
  study; the dense **DeepSeek-LLM-67B-Base** does fit on a
  single H200 141 GB and *is* in the panel. Apertus 70B-Base
  was attempted on the same RunPod orchestrator but the run
  died before its weights finished downloading and is therefore
  also out of the panel reflected in the current artifacts.

- **Instruction-tuned variants are excluded a priori.** RLHF /
  SFT collapses the representation toward "respond to user", not
  toward the geometry the probe is designed to read. Whatever
  pretraining-bias signal exists in the base is masked by the
  instruction tuning. The methodology probes only `-base`
  variants throughout; instruct comparisons would require a
  separate study.

- **One language.** The methodology is language-independent and
  the F1 / F2 / F3 taxonomy applies to any low-resource language
  with cognate neighbours and intra-language polysemy. A
  15-concept replication in Occitan or Aragonese is queued and
  is a weekend of work given the existing scripts.

- **Cross-generation analysis is partial.** The within-family
  delta heatmap (`out/figs/cross_generation_heatmap.png`) only
  covers Gemma 2→4 and Qwen 2→3.5 because those are the only
  families in the panel with both a previous and a current
  generation at comparable size. Mistral Small 24B-Base-2501
  and DeepSeek-LLM-67B-Base are single-point measurements per
  family; tracking how they evolve generationally requires a
  follow-up extraction with the predecessor weights.

---

## What this study does not claim

- That a Catalan-aware model is *worse* than a general-purpose
  one. The H6 verdict is that intentional Catalan training does
  not, by itself, erase the F1 / F2 / F3 collisions; that is *not*
  the same as "Salamandra/ALIA is no better than Gemma". The
  Salamandra family reaches a perfect Catalan-frame ceiling on
  multiple strata; the result is rather that a Catalan-trained
  model does not cancel the deployment-time requirement of
  supplying a Catalan carrier prompt.

- That the title-test result generalises beyond *coca*. The
  noise-cluster collision is a literal property of *coca*'s
  latent placement; whether the analogous test would surface a
  drug-cluster collision for any other Catalan word that
  happens to share a spelling with an English word is an
  empirical question the present study does not answer.

- That the operational recommendation ("supply a Catalan
  carrier prompt") solves the deployment problem. It rescues
  *retrieval-style* probes (the model puts the right
  neighbours within reach). It says nothing about generation
  quality, factual accuracy, or downstream tasks. It is a
  necessary intervention, not a sufficient one.

- That `top10_catalan_frac = 1.0` under `catalan_frame` is the
  same as "the model has correctly identified the cultural
  domain". It is the same as "the top-10 nearest anchors come
  from the curated Catalan list". The two are not the same
  claim — see the audit history of F1 in
  [`methodology.md`](methodology.md) § 1.

---

## Pre-registered next steps

- **Replicate in Occitan and Aragonese.** Same scripts, same
  five-method rigour suite, ~15 concepts each. Tests whether
  the F1 / F2 / F3 taxonomy and the framing-rescue mechanism
  generalise to other Romance low-resource languages.

- **Lift the single-H200 cap to multi-GPU.** Required to fold
  in DeepSeek-V4-Flash-Base (~584 GB bf16), which sits ~4× past
  the single-H200 limit and would need ≥4× H200. The
  predictions queued for that extension would become its
  pre-registered hypotheses verbatim.

- **Re-attempt Apertus 70B-Base.** A swiss-ai Apertus run was
  queued in the panel and died during weight download before
  any extraction completed; queued for the next extraction
  pass.

- **Per-subword vector probe.** Replace the mean-pooled span
  vector with a per-subword vector probed at the same layer.
  Tests whether the F1 / F2 / F3 collisions concentrate at
  particular subword positions or are spread uniformly across
  the target span.

- **Anchor-split cross-validation for layer selection.** Split
  each anchor cluster into a select-half (used for KNN purity)
  and a metric-half (used for `bias_ratio` and
  `top10_catalan_frac`). Eliminates the circularity flagged in
  *methodological caveats* above.

- **Third-party pre-registration timestamping.** The H1–H6
  hypotheses and falsifier thresholds were drafted before each
  extraction run but only as committed text in the project; no
  external registry filing was made for the initial release. A
  future revision may file the same hypothesis set with a
  third-party pre-registration registry (e.g.\ OSF Registrations)
  so future readers can verify intent prior to extraction
  independently of the project repo's own history.
