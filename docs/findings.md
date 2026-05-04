# Detailed findings

> Full results across the eleven-model panel (eight general-purpose
> + three Catalan-aware control), with the verdict on each
> pre-registered hypothesis. The high-level summary lives in
> [`../README.md`](../README.md); the method behind the metrics is
> in [`methodology.md`](methodology.md).

---

## 1. Headline geometry

![headline UMAP](../out/figs/headline_umap.png)

UMAP of bare-frame contextualized hidden states at each model's
best layer, arranged in a 3×4 grid ordered by family (Gemma →
Qwen → Mistral → DeepSeek → Salamandra → ALIA) then by size.
*coca* is highlighted with a yellow callout in every panel —
follow it across models to see the title concept's journey
through eleven different latent spaces.

---

## 2. The two families moved in opposite directions

![cross-generation heatmap](../out/figs/cross_generation_heatmap.png)

`top10_catalan_frac(newer) − top10_catalan_frac(older)`: blue =
the newer model places the concept more Catalan than the older
one; red = the newer regressed.

- **Google (Gemma 2 → 4): more context-invariant.** English-frame
  floor rises (+0.20 on `cross_lingual_bias`), Catalan-frame
  ceiling drops (−0.10 on `domain_specific` /
  `internal_polysemy`). The newer Gemma is harder to push around
  but cannot reach perfect Catalan placement under explicit
  native context.
- **Alibaba (Qwen 2 → 3.5): more context-amplified.**
  Catalan-frame placement climbs to a perfect `1.0` across every
  stratum, but English-frame placement collapses by ~0.30 on
  `cross_lingual_bias` and ~0.40 on `internal_polysemy`. The
  newer Qwen reaches the ceiling under native framing but can be
  dragged into the Anglo cluster by an English sentence.

There is no single "best LLM for Catalan". The right model
depends on whether the deployment can guarantee Catalan framing
(then Qwen) or must survive mixed / untrusted surrounding text
(then Gemma).

---

## 3. Scale alone is not the answer — Gemma 4 26B is the most native-frame-responsive

| model                | anglo_en  | bare  | catalan_frame | neutral_ca |
| -------------------- | --------- | ----- | ------------- | ---------- |
| gemma2-2b            | 0.504     | 0.495 | 0.484         | 0.494      |
| gemma4-e2b           | 0.458     | 0.476 | 0.435         | 0.448      |
| **gemma4-26b-a4b**   | **0.435** | 0.483 | **0.370**     | **0.402**  |
| qwen2-1.5b           | 0.468     | 0.501 | 0.458         | 0.464      |
| qwen3.5-4b-base      | 0.495     | 0.458 | 0.401         | 0.408      |
| qwen3.6-35b-a3b      | 0.473     | 0.488 | 0.427         | 0.448      |
| mistral-small-24b    | 0.485     | 0.482 | 0.426         | 0.457      |
| deepseek-llm-67b     | 0.492     | 0.488 | 0.458         | 0.476      |
| salamandra-2b        | 0.503     | 0.495 | 0.463         | 0.482      |
| salamandra-7b        | 0.501     | 0.495 | 0.465         | 0.484      |
| ALIA-40b             | 0.504     | 0.490 | 0.457         | 0.483      |

Median `bias_ratio` pooled across ambiguity types (audited lexicon,
n = 56 concepts). Gemma 4 26B has the **deepest native-frame
drop** (0.483 bare → 0.370 catalan_frame) and the **lowest
absolute Anglo pull** under Catalan framing of any model probed.
The biggest Qwen (3.6 35B) is noticeably *less* responsive to
Catalan framing than the smaller Qwen 3.5 4B — more parameters
did not translate into more recoverable Catalan geometry. The
two new general-purpose families behave differently from each
other: Mistral Small 24B sits mid-pack (drop 0.056) and DeepSeek
67B-base is **the least native-frame-responsive of the entire
panel** (drop 0.030, comparable to Salamandra-7B and Salamandra-2B
despite having no targeted Catalan training). DeepSeek's signature
in this panel is consistent fairy-tale displacement (see § 4) but
weak prompt-rescue — the worst of both worlds.

---

## 4. *Coca Is Not Cocaine* — title case (F1 grounded by the metric)

![coca lexical-collision test](../out/figs/coca_lexical_collision.png)

Purple = noise-cluster (drug / brand / beverage) hits in the
top-10 KNN of *coca*. The pre-specified threshold of
`n_noise ≥ 2` (committed in advance of the RunPod extraction) is
reached in `bare` and `neutral_ca` frames on multiple models — the
**strongly supported (structural)** branch of the decision rule.

**Under `anglo_en`** — *"Once upon a time, there was a coca who
lived in the enchanted forest"* — **Salamandra 7B puts 9 of 10
nearest anchors in the noise cluster**, ALIA-40B puts 8, and
Mistral Small 24B / Qwen 3.5 4B / Salamandra 2B each put 7. Six
more models (Gemma 2/4-26B, Qwen 3.6, gemma2-2b) come in at 6.
Two outliers: Gemma 4 E2B and DeepSeek 67B-base put **zero**
noise anchors in the top-10 — DeepSeek's top-3 is *enchanted
forest, fairy, princess*, drifting cleanly into the fairy-tale
cluster instead (pure F2). The title's F1 reading is not
metaphor for the rest of the panel: the nearest-neighbour set
for *coca* literally contains *cocaine*.

**Under `bare`** (no carrier sentence) the effect persists across
the panel: 7 noise hits on ALIA-40B, Mistral Small 24B,
Salamandra 2B, 6 on DeepSeek 67B / Gemma 2 / Qwen 2 / Salamandra
7B, 5 on Gemma 4 26B and Qwen 3.6 35B, 3 on Qwen 3.5 4B, 0 on
Gemma 4 E2B. This is what *structural* means in the decision
rule — the collision is a property of where *coca* lives in
latent space, not induced by an English scaffold.

**Under `catalan_frame`** — *"A la festa major del poble, els
coca van sortir a la plaça acompanyats dels grallers"* — noise
drops to 0 on three models (DeepSeek 67B, Gemma 4 E2B, Qwen 3.5 4B),
to 1 on Gemma 4 26B and Mistral Small 24B (only *Coca-Cola*
survives), and Gemma 4 26B's top-3 becomes *plaça, enxaneta,
ajuntament*. The collision is rescuable, but only by a strong
native frame, and the depth of the rescue is family-dependent —
the Salamandra family and ALIA still carry 3-6 noise neighbours
under `catalan_frame`, which is the cleanest evidence that
intentional Catalan training does not, by itself, dissolve the
F1 collision.

![coca bar chart](../out/figs/focus_coca_bar.png)

Anglo-anchor count (blue side of the binary `bias_ratio`) in the
top-10 KNN of *coca* per (model, frame). The complement to the
noise-cluster chart above: together they tell you which cluster
catches *coca* when it drifts, and which frame rescues it.

![coca trajectory](../out/figs/focus_coca_trajectory.png)

Each panel is one model's UMAP at its best layer, showing only
the anchors plus the four frame-versions of *coca* as colored
stars. The spread of the four stars per panel is exactly how
much the surrounding context can move the same Catalan word —
Gemma's stars cluster tightly (context-invariant); Qwen 3.5's
stars spread across the cloud (context-amplified).

---

## 5. Same treatment for *gegant* and *castell*

The two cross_lingual_bias archetypes (gegant ↔ giant, castell ↔
castle) behave very differently from *coca* under the same
probes. Neither concept triggers the dramatic Qwen 3.5 failure we
see for *coca*, which is why the project takes its title from the
latter. See `out/report.md` for the per-concept tables and
`out/figs/focus_{gegant,castell}_{bar,trajectory}.png` for the
visuals.

---

## 6. The Catalan-aware control (Salamandra / ALIA from BSC)

The H6 verdict from the Catalan-aware extraction: intentional
Catalan training does not, by itself, erase the F1 / F2 / F3
collisions. The collision modes survive on the Salamandra family
up to ALIA-40B, suggesting they are properties of the
**contextualised-embedding pipeline** (tokenisation, mean-pooling,
training-objective geometry) rather than of training-data mixture.
Three of four pre-specified H6 sub-hypotheses were falsified. The
BSC's open-Catalan model series is the indispensable causal control
that allows this conclusion: without the Salamandra/ALIA panel we
could not have distinguished the data-mix hypothesis from the
pipeline hypothesis. The full falsifier verdict is committed at
`out/h6_verdict.json`.

---

## 7. Conclusions

1. **The F1/F2/F3 taxonomy is necessary.** Under `anglo_en` on
   Qwen 3.5 4B the pooled median `top10_catalan_frac` is **0.50**
   — a reader scanning a single-axis number would conclude the
   model is unbiased. Stratifying reveals three distinct
   stratum-level stories coexisting inside that pool: **F1 = 0.55**
   (cognate arrow partially offset by well-anchored Catalan
   scaffolding), **F2 = 0.50** (no cognate, lands at the median),
   **F3 = 0.40** (the mundane polysemic sense dominates and is
   *more* Anglo-adjacent than F1's cognate pull). The right
   intervention for F3 is *teach the model* `pinya` *has a second
   sense that is not a pinecone*, not *teach the model `gegant` ≠
   `giant`*.

2. **The two families moved in opposite directions, and this
   survives at the large tier.** Gemma becomes
   *context-invariant* (floor rises under English framing,
   ceiling drops under Catalan framing). Qwen becomes
   *context-amplified* (`top10_catalan_frac` climbs to 1.0 under
   Catalan framing across every stratum, but English framing
   collapses). There is no global "best LLM for Catalan" — it
   depends on whether the deployment can guarantee Catalan
   framing.

3. **Scale is not the answer.** The biggest Qwen (3.6 35B-A3B)
   is *less* responsive to Catalan framing than the smaller Qwen
   3.5 4B. The Catalan-aware ALIA-40B has the deepest absolute
   drop and the strongest response to native framing —
   suggesting it is not size but the multilingual pretraining
   diet that matters, *combined with* the framing at inference
   time.

4. ***Coca is not cocaine* is grounded by the metric, not
   metaphor.** The pre-registered title test (`n_noise ≥ 2` in
   the top-10 KNN of *coca*, outside the `anglo_en` frame)
   returns the **strongly supported (structural)** branch on
   ten of eleven panel models. The single exception is Gemma
   4 E2B, which carries zero noise neighbours across every
   frame (its *coca* drifts into anglo fairy-tale terms instead,
   so the title test misses it but F2 catches it). Under `bare`
   frame *coca*'s nearest neighbours include *Coca-Cola*,
   *cocaine*, *coke* and *soda* on every other model; under
   `anglo_en` Salamandra 7B puts **9 of 10** noise-cluster
   anchors in the top-10, ALIA 40B puts 8, and Mistral Small
   24B / Qwen 3.5 4B / Salamandra 2B each put 7. Under
   `catalan_frame` three models rescue completely (DeepSeek
   67B, Gemma 4 E2B, Qwen 3.5 4B) and Gemma 4 26B-A4B / Mistral
   Small 24B drop to 1 noise neighbour. The title's F1 reading
   is a literal property of latent space, not a metaphor; the
   collision is rescuable, but only by a strong native frame,
   and the rescue depth is family-dependent — the Catalan-aware
   control tier carries the heaviest residual noise.

---

## 8. Conclusions (català)

1. **La taxonomia F1/F2/F3 és necessària.** Sota `anglo_en` al
   Qwen 3.5 4B la mediana *pooled* de `top10_catalan_frac` és
   **0,50** — un lector que només miri un sol número conclouria
   que el model no té biaix. Estratificar revela tres històries
   convivint dins d'aquell pool: **F1 = 0,55** (l'arrossegament
   cognat queda parcialment compensat pel bon ancoratge català
   del concepte), **F2 = 0,50** (sense cognat, cau a la
   mediana), **F3 = 0,40** (el sentit polisèmic mundà domina i
   és *més* anglo-adjacent que l'arrossegament cognat de F1). La
   intervenció correcta per a la F3 és **ensenyar al model que**
   `pinya` *té un segon sentit que no és el fruit del pi*, i no
   **ensenyar-li que** `gegant` ≠ `giant`.

2. **Les dues famílies s'han mogut en direccions oposades, i la
   cosa sobreviu al nivell gran.** Gemma es fa *context-invariant*
   (el sòl puja sota frame anglès, el sostre baixa sota frame català).
   Qwen es fa *context-amplificat* (`top10_catalan_frac` puja a
   1.0 sota frame català a tots els estrats, però el frame
   anglès col·lapsa). No hi ha cap "millor LLM per al català"
   global — depèn de si el desplegament pot garantir un framing
   català.

3. **L'escala no és la resposta.** El Qwen més gran (3.6
   35B-A3B) respon menys al framing català que el Qwen
   3.5 4B: més paràmetres no equivalen a més geometria catalana
   recuperable. ALIA-40B (amb consciència del català) és l'excepció — té la
   caiguda més profunda i la resposta més forta al framing natiu —
   cosa que suggereix que el que importa és la dieta
   multilingüe del preentrenament *combinada amb* el framing a
   inferència.

4. ***La coca no és cocaïna*, i el test preregistrat ho demostra
   literalment.** El test del títol (`n_noise ≥ 2` al top-10
   KNN de *coca*, fora del frame `anglo_en`) retorna la branca
   **fortament recolzada (estructural)** a deu d'onze models del
   panell. L'única excepció és Gemma 4 E2B, que té zero veïns de
   soroll a qualsevol frame (la seva *coca* deriva cap a les
   àncores anglo de contes de fades, així que el test del títol no
   l'agafa però la F2 sí). Sota frame `bare` (sense oració
   portadora) els veïns més pròxims de *coca* inclouen
   *Coca-Cola*, *cocaine*, *coke* i *soda* a la resta de models;
   sota `anglo_en` Salamandra 7B col·loca **9 de 10** àncores al
   clúster de soroll, ALIA 40B en col·loca 8, i Mistral Small
   24B / Qwen 3.5 4B / Salamandra 2B en col·loquen 7 cadascun.
   Sota `catalan_frame` tres models rescaten el concepte del tot
   (DeepSeek 67B, Gemma 4 E2B, Qwen 3.5 4B) i Gemma 4 26B-A4B /
   Mistral Small 24B baixen a 1 veí de soroll. La lectura F1 del
   títol és una propietat literal de l'espai latent, no una
   metàfora; el rescat és possible però només amb un frame
   natiu fort, i la profunditat del rescat depèn de la família —
   el nivell de models amb consciència del català és precisament
   el que carrega el soroll residual més alt.
