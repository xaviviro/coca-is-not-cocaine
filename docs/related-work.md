# Treballs relacionats

> Antecedents i posicionament del projecte: el document metodològic propi
> (en català) i les recerques prèvies en què s'inspira aquesta feina.

---

## Document propi

- **Vinaixa Roselló, X.** (2026). *Coca no és cocaïna: tres modes de
  col·lisió lèxic-cultural en LLMs de pesos oberts, provats en català.*
  Memòria metodològica en català — [`metodologia.pdf`](../metodologia.pdf).

  Document de referència del projecte: explica la taxonomia F1 / F2 / F3,
  el disseny del lèxic de 56 conceptes, els tres clústers d'anchors i la
  bateria estadística de cinc mètodes. La versió canònica del codi i les
  dades viu en aquest mateix repositori.

---

## Inspiració metodològica directa

- **Naous, Ryan, Ritter & Xu (2024)** — *Having Beer after Prayer?
  Measuring Cultural Bias in Large Language Models.* ACL 2024.
  [arXiv:2305.14456](https://arxiv.org/abs/2305.14456)

  **CAMeL**: el marc de referència del qual partim. CAMeL prova el biaix
  cultural amb fill-in-the-blank en àrab. Aquest projecte el porta (i) del
  fill-in-the-blank a la geometria d'embeddings contextualitzats; (ii) de
  l'àrab al català; (iii) afegeix la taxonomia F1/F2/F3 que separa els
  modes de fallada que CAMeL agrega en una sola mètrica.

---

## Tronc de cultural-bias en LLMs

- **Tao, Viberg, Baker & Kizilcec (2024)** — *Cultural Bias and Cultural
  Alignment of Large Language Models.* PNAS Nexus.
  [arXiv:2311.14096](https://arxiv.org/abs/2311.14096)

  Mostra desalineament cultural sistemàtic en outputs d'LLMs sobre
  societats no-occidentals — el "perquè importa" macro de la nostra feina
  micro-geomètrica.

- **Li, Shafiq, Mathur & Roy (2023)** — *GeoMLAMA: Geographical and
  Cultural Biases in Multilingual Pre-trained Language Models.*
  [arXiv:2205.12247](https://arxiv.org/abs/2205.12247)

  Probes de biaix geogràfic en LMs multilingües. Confirma que el problema
  travessa modalitats de probe, no només la nostra.

- **Rystrøm, Kirk & Hale (2025)** — *Multilingual ≠ multicultural.*
  [arXiv:2502.16534](https://arxiv.org/abs/2502.16534)

  Argumenta que entrenar amb més llengües no implica capturar més
  cultures. És exactament la lectura "pipeline, no data-mix" que la
  família Salamandra/ALIA confirma empíricament en aquest projecte.

- **Wang, Saar-Tsechansky, Sun & Wang (2023)** — *Not all countries
  celebrate Thanksgiving: cultural dominance in LLMs.*
  [arXiv:2310.12481](https://arxiv.org/abs/2310.12481)

  Documenta dominància cultural anglo-americana al training mix.
  Motivació per fer servir el català com a cas límit.

---

## Tradició clàssica de geometria d'embeddings

- **Bolukbasi, Chang, Zou, Saligrama & Kalai (2016)** — *Man is to
  computer programmer as woman is to homemaker? Debiasing word
  embeddings.* [arXiv:1607.06520](https://arxiv.org/abs/1607.06520)

  Paper fundacional de probes de biaix per **proximitat cosinus a anchor
  sets**. La `bias_ratio = pull_anglo / (pull_anglo + pull_catalan)`
  d'aquest projecte és l'extensió contextualitzada d'aquell mètode,
  aplicada a una col·lisió cultural en lloc d'una col·lisió de gènere.

---

## Restricció metodològica (per què no quantization)

- **Dettmers et al. (2022)** — *LLM.int8(): 8-bit matrix multiplication
  for transformers at scale.* NeurIPS 2022.
  [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)

  Mostra que la quantització reorganitza la geometria de representació.
  És la base del "bf16 only, no fp8 / int8 / NVFP4" que apliquem a tots
  els models del panell (vegeu [`limitations.md`](limitations.md) i la
  secció *Don'ts* del [README](../README.md)).

---

## Context divulgatiu

- **Ada Lovelace Institute (2025)** — *Tokenising culture: causes and
  consequences of cultural misalignment in LLMs.*
  [adalovelaceinstitute.org](https://www.adalovelaceinstitute.org/blog/cultural-misalignment-llms/)

  Síntesi pública del problema des de policy/governance — útil per
  emmarcar el "per què hauria d'importar a algú que no fa NLP".

---

## Mapa ràpid de què aporta cada referència

| Treball                          | Què hereta aquest projecte                                       |
| -------------------------------- | ---------------------------------------------------------------- |
| CAMeL (Naous 2024)               | el marc del problema, anchor sets per cultura, falsifiers preregistrats |
| GeoMLAMA / Tao / Wang            | motivació + evidència que el problema és general                 |
| Rystrøm 2025                     | la lectura "pipeline ≠ data" que confirmem amb Salamandra/ALIA   |
| Bolukbasi 2016                   | la mètrica geomètrica `bias_ratio` (versió contextualitzada)     |
| Dettmers 2022                    | la restricció bf16 sobre tot el panell                           |
| Ada Lovelace 2025                | l'emmarcament policy/divulgatiu                                  |
