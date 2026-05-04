# Methodology

> Full method, lexicon design, anchor sets, metrics and the
> five-method statistical-rigour suite. The high-level summary lives
> in [`../README.md`](../README.md); reproduction commands live in
> [`reproduce.md`](reproduce.md).

---

## 1. The F1/F2/F3 taxonomy of lexical-cultural collisions

A culturally-marked target word *c* in a low-resource language can
fail in any one of three structurally different ways when a probe
asks where *c* sits in a multilingual base LLM's latent space.
Single-axis cultural-bias probes (e.g. CAMeL,
[Naous et al. 2024](https://arxiv.org/abs/2305.14456)) collapse all
three modes onto one number, which hides what is actually going on.
The 56-concept Catalan lexicon at `data/concepts_ca.csv` is
stratified by an `ambiguity_type` column that tags each target with
exactly one of:

| code | type                  | n  | failure mode                              | examples                                  |
| ---- | --------------------- | -- | ----------------------------------------- | ----------------------------------------- |
| **F1** | `cross_lingual_bias` | 10 | direct EN cognate, different universe     | gegant ↔ giant, castell ↔ castle, drac ↔ dragon, coca ↔ coca, Pasqua ↔ Easter, Sant Jordi ↔ Saint George, capgròs ↔ big-head, pilar ↔ pillar, diables ↔ devils, Mercè ↔ Mercy |
| **F2** | `domain_specific`    | 29 | no EN cognate — pure domain depth          | correfoc, casteller, cercavila, panellet, grallers, pa amb tomàquet, samfaina, calçotada, romesco, … |
| **F3** | `internal_polysemy`  | 17 | mundane CA sense dominates festive sense   | pinya (pinecone ‖ castell-base), tió (log ‖ Christmas-log), folre (lining ‖ castell padding), manilles (handcuffs ‖ castell padding), terra (earth ‖ political territory), … |

### Why each stratum needs a different intervention

- **F1** confounds *anglification* with *cognate semantics*: when
  *gegant* lands near *giant*, that is partly because they
  share a referent in English (the large mythical figure) and
  partly because English-dominant pretraining drags the Catalan
  word toward its English cognate. A pull is observed but its
  cause is ambiguous by construction.

- **F2** is the cleanest evidence stratum. *Correfoc* has no
  English cognate; if it lands inside the Anglo fairy-tale cluster
  the pull is structural — there is no lexical shortcut to fall
  onto.

- **F3** can produce a high pull *without* anglification: if
  *pinya* lands near *pinecone*, the model is just reflecting that
  the mundane Catalan sense (the fruit / cone) is more frequent in
  the training mix than the festive one (the base of a *castell*).
  The fix is not "teach the model that *pinya* ≠ *giant*"; the
  fix is "teach the model *pinya* has a second sense that is not a
  pinecone".

### F1 audit

F1 is deliberately kept *tight* (10 concepts) and conservative.
A concept enters F1 only if it has a direct English cognate
**and** that cognate lives in a clearly different cultural
universe. The F1 audit removed three concepts (*mare*,
*port*, *vall*) whose cognates were culturally co-extensive with
their Catalan senses (a *port* is a port in both languages), and
moved *terra* from F1 to F3 because its interesting case (*la
terra* as political territory vs.\ *terra* as soil) is
intra-Catalan polysemy rather than cross-lingual collision.
Choosing 10 pure F1 over 14 contaminated F1 is the right call
because F1 signal is supposed to be *not confounded by cognate
semantics*; if half the stratum has cognates that mean the same
thing in both languages, there is no signal to read.

---

## 2. Probe frame design

For each target concept *c* the pipeline emits four probes:

| frame_id        | sentence                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------ |
| `bare`          | *c* alone, no surrounding context.                                                         |
| `anglo_en`      | *Once upon a time, there was a c who lived in the enchanted forest.*                       |
| `catalan_frame` | *A la festa major del poble, els c van sortir a la plaça acompanyats dels grallers.*       |
| `neutral_ca`    | *El c és un element important.*                                                            |

We deliberately leave the Catalan lemma verbatim in `anglo_en`
even when *c* has a direct English cognate. Substituting *gegant*
→ *giant* would (a) embed the English word, so the resulting
vector measures *giant*'s geometry, not the cross-lingual pull on
*gegant*; and (b) collide directly with the Anglo anchor set,
which contains *giant, castle, dragon*.

---

## 3. Anchor sets

| file                              | n  | role                                                          |
| --------------------------------- | -: | ------------------------------------------------------------- |
| `data/anchors_anglo.csv`          | 22 | Anglo fairy-tale / medieval vocabulary (the dominant cluster) |
| `data/anchors_catalan.csv`        | 23 | Catalan folkloric vocabulary (the target cluster)             |
| `data/anchors_noise.csv`          | 13 | Drug / brand / beverage cluster (used for the title test only)|

The Anglo file has a `subcluster` column that splits its 22
entries into `fairy_tale` (giant, ogre, dwarf, witch, dragon,
fairy, troll, goblin, monkey, beanstalk, enchanted forest),
`medieval` (knight, castle, sword, kingdom, jester, peasant,
dungeon, tavern) and `character` (princess, Jack, Cinderella).
The decomposition lets the per-concept analysis report *which*
English neighborhood catches each focus concept when it drifts.

The noise cluster does **not** enter the binary `bias_ratio`. It
is used solely for the title test (Section 5).

---

## 4. Hidden-state extraction

For each `(c, f)` probe and each requested layer *L*:

1. Tokenise the probe sentence with the model's fast tokenizer
   (`use_fast=True`, `return_offsets_mapping=True`).
2. Find the character span of *c* in `full_text` and map it onto
   its token span via `offset_mapping`.
3. Forward pass with `output_hidden_states=True`, `use_cache=False`,
   no gradients.
4. Slice `hidden_states[L][0, t_start:t_end, :]`, mean-pool over
   the subword positions, L2-normalize the result.

All extractions use `torch_dtype=torch.bfloat16`,
`device_map="auto"`. Quantization is forbidden — it reorganises
the representation geometry the metrics measure
(see `Don'ts` in `../README.md`).

Edge cases: a concept that does not appear verbatim in
`full_text` is logged and skipped; a concept whose span tokenises
to multiple subwords is mean-pooled across the whole span (the
common case for Catalan on these tokenizers); a concept that
appears more than once in `full_text` uses the first occurrence
and emits a warning.

---

## 5. Metrics

### `bias_ratio`

For each `(c, f)`:

```
pull_anglo(c, f)   = mean_a∈ANGLO   cos( v(c, f), v(a, bare) )
pull_catalan(c, f) = mean_a∈CATALAN cos( v(c, f), v(a, bare) )
bias_ratio(c, f)   = pull_anglo / (pull_anglo + pull_catalan)
```

with 95 % CIs from 1000 bootstrap resamples over the anchor sets.
Self-matches by surface string are excluded (e.g. when *diada*
appears as both target and Catalan anchor).

### `top10_catalan_frac` (complementary metric)

Raw cosines saturate near 0.5 on both clusters at deep layers
(generic token frequency makes the target similar to most
anchors). That compresses `bias_ratio`'s dynamic range. So we
also report a rank-based metric: of the 10 anchors nearest to
*v(c, f)* by cosine, what fraction belong to the Catalan
cluster? `1.0` = the entire top-10 KNN is Catalan folkloric;
`0.0` = the entire top-10 is Anglo fairy-tale.

### Frame differential `Δ`

```
Δ(c) = bias_ratio(c, anglo_en) − bias_ratio(c, catalan_frame)
```

`Δ` is the headline diagnostic. Large positive `Δ` ⇒ the model
has retrievable Catalan geometry that an explicit Catalan frame
unlocks. `Δ ≈ 0` ⇒ the representation is structurally
Anglo-anchored regardless of framing.

The CI on `Δ` uses **paired** bootstrap: each iteration draws a
single anchor index per cluster and reuses it for both frames
before differencing. Marginal CIs on the two frame-bias values do
not cancel the shared anchor noise; the paired version does and
is materially tighter.

---

## 6. Layer selection

For each model we sweep three layers at 25 / 60 / 90 % depth and
pick the best by *deepest layer whose anchor-KNN purity is within
0.15 of the model's per-layer max*. Early layers tend to win raw
purity by surface morphology (English vs Catalan orthography),
which is not what we want to probe; the tolerance keeps us in the
semantic regime while preserving objectivity.

The layer-selection rule shares its anchor pool with the metrics
computed on the chosen layer, so it is mildly circular. We mitigate
in two ways: every metric we report is also reported across all
three committed layers per model (see `out/{model}/layer_selection.csv`
and the layer-depth × frame plot), and the post-extraction iteration added the paired
bootstrap CI per concept so the headline numbers come with
explicit uncertainty rather than point estimates.

---

## 7. Pre-registered title test

The project's title makes a literal claim (*coca is not cocaine*),
not just an F2 metaphor (*coca drifts toward fairy-tale
vocabulary*). To test the literal claim we add the noise anchor
cluster (`data/anchors_noise.csv`: 13 drug / brand / beverage
words) and count how many of them land in the top-10 KNN of
*coca* per `(model, frame)`. The pre-specified decision rule
(committed in advance of the cloud extraction):

| outcome                                        | rule                                                     | how the result is reported     |
| ---------------------------------------------- | -------------------------------------------------------- | ------------------------------ |
| **falsified**                                  | `max n_noise < 2` across every cell                      | reframe title to *Coca is not a Giant* |
| **partially supported (context-induced)**      | `n_noise ≥ 2` *only* under `anglo_en`                    | report as a prompt artifact    |
| **strongly supported (structural)**            | `n_noise ≥ 2` in `bare`, `catalan_frame` *or* `neutral_ca` | keep title; report as a property of latent space |

The extraction returns the **strongly supported (structural)**
branch on all eight general-purpose models (Gemma 2 / 4, Qwen 2 /
3.5 / 3.6, Mistral Small 24B, DeepSeek 67B). See
[`findings.md`](findings.md) for the full data.

---

## 8. The five-method statistical-rigour suite

Every model in the 11-model panel goes through the same five
additional analyses, applied uniformly. All five are derivable
from the committed bias / KNN / NPZ artifacts without retraining
or re-extracting. The NPZ files are kept on disk and committed
to the repository, so the large-tier CI is a proper paired
bootstrap throughout — no conservative-marginal upper bound
anywhere in the current artifact set.

1. **Per-concept paired-bootstrap CI on Δ.** 1000 resamples per
   concept, 95 % CI excludes-zero flag.
   Artifacts: `out/delta_per_concept.csv`,
   `out/figs/delta_forest.png`.
2. **Layer-depth × frame interaction.** Δ at all three committed
   layers per model, by ambiguity_type.
   Artifacts: `out/layer_depth_delta.csv`,
   `out/figs/layer_depth_delta.png`.
3. **Anglo subcluster pull.** Top-10 KNN composition of the
   focus concepts (coca, gegant, castell) split by the
   fairy_tale / medieval / character sub-clusters of
   `data/anchors_anglo.csv`.
   Artifacts: `out/anglo_subcluster_pull.csv`,
   `out/figs/subcluster_{coca,gegant,castell}.png`.
4. **Stratified-bootstrap CIs on the (model × ambiguity_type ×
   frame) median table.** Resamples concepts within each
   `ambiguity_type` so the 72-cell median has uncertainty
   attached.
   Artifacts: `out/ambiguity_stratified_ci.csv`,
   `out/figs/ambiguity_stratified_ci.png`.
5. **Procrustes cross-model alignment.** Per-model PCA → 30
   dims, then orthogonal Procrustes against a reference (the
   smallest model's anchor PCA) using the shared bare-anchor
   set. Focus concepts (coca, gegant, castell) traced as
   polylines across the aligned models. The combined UMAP renders
   the small-tier chronological pair (Gemma 2→4, Qwen 2→3.5).
   Cross-tier alignment with the large dense / MoE models is out
   of scope (dimension mismatch, MoE expert routing); the
   large-tier additions (Mistral Small 24B, DeepSeek 67B) appear
   in every other figure but not in this one.
   Artifacts: `out/figs/procrustes_combined_umap.png`,
   `out/figs/procrustes_combined_umap_full.png`.

---

## References

- Naous, Ryan, Ritter & Xu (2024). *Having Beer after Prayer? Measuring
  Cultural Bias in Large Language Models.* ACL 2024 — primary methodological
  prior. [arXiv:2305.14456](https://arxiv.org/abs/2305.14456)
- Tao, Viberg, Baker & Kizilcec (2024). *Cultural Bias and Cultural Alignment
  of Large Language Models.* PNAS Nexus.
  [arXiv:2311.14096](https://arxiv.org/abs/2311.14096)
- Li, Shafiq, Mathur & Roy (2023). *GeoMLAMA: Geographical and Cultural
  Biases in Multilingual Pre-trained Language Models.*
  [arXiv:2205.12247](https://arxiv.org/abs/2205.12247)
- Rystrøm, Kirk & Hale (2025). *Multilingual ≠ multicultural.*
  [arXiv:2502.16534](https://arxiv.org/abs/2502.16534)
- Wang, Saar-Tsechansky, Sun & Wang (2023). *Not all countries celebrate
  Thanksgiving: cultural dominance in LLMs.*
  [arXiv:2310.12481](https://arxiv.org/abs/2310.12481)
- Bolukbasi, Chang, Zou, Saligrama & Kalai (2016). *Man is to computer
  programmer as woman is to homemaker? Debiasing word embeddings.*
  [arXiv:1607.06520](https://arxiv.org/abs/1607.06520)
- Dettmers et al. (2022). *LLM.int8(): 8-bit matrix multiplication for
  transformers at scale.* NeurIPS 2022.
  [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- Ada Lovelace Institute (2025). *Tokenising culture: causes and consequences
  of cultural misalignment in LLMs.*
  [adalovelaceinstitute.org/blog/cultural-misalignment-llms](https://www.adalovelaceinstitute.org/blog/cultural-misalignment-llms/)
