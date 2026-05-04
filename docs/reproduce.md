# Reproduce the experiment

> Step-by-step recipe to regenerate every artifact under `out/`
> from a fresh clone. Methodology details live in
> [`methodology.md`](methodology.md); detailed findings live in
> [`findings.md`](findings.md).

---

## Hardware tiers

| Tier             | Models                                                                                       | GPU                                                                       | Wall-clock           |
| ---------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | -------------------- |
| Small (local)    | Gemma 2 2B, Gemma 4 E2B, Qwen2 1.5B, Qwen 3.5 4B, Salamandra 2B                              | 16 GB consumer (RTX 4060 Ti / 4080 / similar)                             | ~2 minutes per model |
| Large (cloud)    | Gemma 4 26B-A4B, Mistral Small 24B, Qwen 3.6 35B-A3B, Salamandra 7B, ALIA 40B, DeepSeek 67B  | 1× H200 SXM 141 GB *(or H100 80 GB for everything except DeepSeek 67B)*   | ~5–10 minutes per model |

The whole 11-model panel runs end-to-end on a single H200 in
~70-90 minutes. DeepSeek 67B-base is dense (~134 GB bf16) and
does *not* fit a single H100 80 GB; an H200 is required for that
one model.

---

## 1. Setup

```bash
git clone https://github.com/xaviviro/coca-is-not-cocaine
cd coca-is-not-cocaine
uv sync
```

`uv` resolves `pyproject.toml` and creates `.venv` with the pinned
versions of `transformers`, `torch`, `umap-learn`, `accelerate`,
`pyarrow`, `tabulate`, `matplotlib`, `seaborn`, `scikit-learn`,
`pandas`, `numpy`, `sentencepiece`, and `bitsandbytes` (optional;
unused under our methodology because quantization is forbidden).

Hugging Face authentication: Gemma 2, Gemma 4 and Mistral Small 24B
Base are gated on HF. Generate a fresh read-scope token at
<https://huggingface.co/settings/tokens>, accept the Gemma 2,
Gemma 4 and Mistral Small license terms by clicking through on
each model card, then:

```bash
export HF_TOKEN=hf_...
```

Do not paste the token into chat logs or commit it to a file. The
project's other models (Qwen, Salamandra/ALIA) are non-gated.

---

## 2. Build the probe set

```bash
uv run python scripts/01_build_dataset.py
```

Reads:

- `data/concepts_ca.csv` (56 Catalan targets with `ambiguity_type`)
- `data/anchors_anglo.csv` (22 + subcluster column)
- `data/anchors_catalan.csv` (23)
- `data/anchors_noise.csv` (13 drug / brand / beverage anchors;
  used only in the title test)

Writes `data/probes.csv` (282 rows: 4 frames × 56 targets + 1
bare × 58 anchors). `data/probes.csv` is gitignored — regenerate
it from the curated CSVs above.

---

## 3. Extract embeddings

`scripts/02_extract_embeddings.py` is the GPU-heavy stage. It
loads a base model in bf16, runs forward passes over every probe
with `output_hidden_states=True`, mean-pools over the target
concept's token span at the requested layers, L2-normalises, and
saves an NPZ per layer plus a metadata sidecar.

### Small + previous-generation tier (local)

```bash
uv run python scripts/02_extract_embeddings.py \
    --model google/gemma-2-2b      --layers 6,15,22 --out out/gemma2-2b
uv run python scripts/02_extract_embeddings.py \
    --model google/gemma-4-E2B     --layers 6,15,22 --out out/gemma4-e2b
uv run python scripts/02_extract_embeddings.py \
    --model Qwen/Qwen2-1.5B        --layers 7,17,26 --out out/qwen2-1.5b
uv run python scripts/02_extract_embeddings.py \
    --model Qwen/Qwen3.5-4B-Base   --layers 9,22,32 --out out/qwen3.5-4b-base
uv run python scripts/02_extract_embeddings.py \
    --model BSC-LT/salamandra-2b   --layers 6,14,21 --out out/salamandra-2b
```

### Large + current-generation tier (single H200 141 GB)

```bash
uv run python scripts/02_extract_embeddings.py \
    --model google/gemma-4-26B-A4B                  --layers 8,18,27  --out out/gemma4-26b-a4b
uv run python scripts/02_extract_embeddings.py \
    --model mistralai/Mistral-Small-24B-Base-2501   --layers 10,24,37 --out out/mistral-small-24b
uv run python scripts/02_extract_embeddings.py \
    --model Qwen/Qwen3.6-35B-A3B                    --layers 10,24,36 --out out/qwen3.6-35b-a3b
uv run python scripts/02_extract_embeddings.py \
    --model BSC-LT/salamandra-7b                    --layers 8,19,29  --out out/salamandra-7b
uv run python scripts/02_extract_embeddings.py \
    --model BSC-LT/ALIA-40b                         --layers 12,29,43 --out out/ALIA-40b
uv run python scripts/02_extract_embeddings.py \
    --model deepseek-ai/deepseek-llm-67b-base       --layers 18,48,92 --out out/deepseek-llm-67b
```

> **Keep the NPZ files on disk** for the entire analysis run. The
> paired-bootstrap CI on Δ requires the per-concept vectors per
> frame, so deleting `out/{slug}/embeddings_layer*.npz` between
> 02 and 03 collapses the proper paired CI into a conservative
> marginal upper bound.

---

## 4. Analyse

```bash
uv run python scripts/03_analysis.py --model all
```

For each model in `out/`, `03_analysis.py` reads the NPZ files,
selects the best layer (deepest within 0.15 of the per-layer KNN
purity max), and writes:

- `out/{slug}/best_layer.json`
- `out/{slug}/layer_selection.csv`
- `out/{slug}/layer{L}/bias.csv` (per-(concept, frame) bias_ratio
  + bootstrap CI + top10_catalan_frac)
- `out/{slug}/layer{L}/bias_by_ambiguity.csv`
- `out/{slug}/layer{L}/delta.csv` (paired-bootstrap Δ + 95 % CI)
- `out/{slug}/layer{L}/knn.csv` (top-10 anchors per target)
- `out/{slug}/layer{L}/umap.png`

---

## 5. Compose the report and figures

```bash
uv run python scripts/04_report.py
```

Reads every `out/{slug}/best_layer.json`, joins on the per-layer
artifacts, and writes:

- `out/report.md`
- `out/figs/headline_umap.png` (multi-panel grid sized for the
  number of probed models)
- `out/figs/cross_generation_heatmap.png`
- `out/figs/delta_forest.png`
- `out/figs/layer_depth_delta.png`
- `out/figs/ambiguity_stratified_ci.png`
- `out/figs/procrustes_combined_umap.png` (clean) and
  `out/figs/procrustes_combined_umap_full.png` (every anchor labeled)
- `out/figs/coca_lexical_collision.png` (the title test)
- `out/figs/focus_{coca,gegant,castell}_{bar,trajectory}.png`
- `out/figs/subcluster_{coca,gegant,castell}.png`
- `out/coca_lexical_collision.csv`
- `out/delta_per_concept.csv`
- `out/anglo_subcluster_pull.csv`
- `out/ambiguity_stratified_ci.csv`
- `out/layer_depth_delta.csv`
- `out/median_*.csv` plus `coca_case_study.csv` and
  `cross_generation_deltas.csv`

Both `out/{slug}/embeddings_layer*.npz` and
`out/{slug}/metadata.parquet` are committed for fully reproducible
re-analysis without re-running the GPU-heavy stage.

---

## Output structure

```
.
├── README.md
├── pyproject.toml       uv-managed Python 3.11+ project
├── .python-version
├── data/
│   ├── concepts_ca.csv          56 Catalan targets
│   ├── anchors_anglo.csv        22 anchors + subcluster column
│   ├── anchors_catalan.csv      23 anchors
│   ├── anchors_noise.csv        13 drug / brand / beverage anchors
│   └── probes.csv               (gitignored) 282 generated probes
├── docs/
│   ├── methodology.md           full method & metrics
│   ├── reproduce.md             this file
│   ├── findings.md              detailed results & verdicts
│   └── limitations.md           caveats & pre-specified next steps
├── scripts/
│   ├── 01_build_dataset.py
│   ├── 02_extract_embeddings.py
│   ├── 03_analysis.py
│   └── 04_report.py
└── out/                  report + figures + CSVs (committed);
                          *.npz and *.parquet ignored
```

---

## Don'ts

- **No `-it` / instruct variants.** RLHF masks raw pretraining
  bias; instruction tuning collapses the representation toward
  "respond to user", not toward the geometry the probe is
  designed to read.
- **No raw cross-model cosine comparisons.** The latent spaces
  are unaligned; only the rank-based metric
  (`top10_catalan_frac`) and the normalised `bias_ratio` are
  comparable across models.
- **No input-embedding-table probes.** Decontextualised, defeats
  the hypothesis.
- **No cherry-picking layers.** Publish the full sweep and let
  the layer-selection rule of `methodology.md` § 6 pick the best
  per model. Section 8.2 of `methodology.md` reports `Δ` across
  all three committed layers regardless.
- **No quantization.** fp8 / int8 / NVFP4 reorganise the
  representation geometry the metrics measure (see Dettmers et
  al. 2022). Stay in bf16.
