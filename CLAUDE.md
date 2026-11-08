# Paper 14 Reproduction: Random Truncation of Text Embeddings

CS F429 (Intro to NLP), BITS Pilani Hyderabad, Group 12. Built by one person (Aditya) end to end, so every choice must be easy for him to explain alone in the viva. Paper presentation held with Project Phase 3 (deadline 21 Nov 2026, target done by 12 Nov).

Paper: Takeshita et al., "Randomly Removing 50% of Dimensions in Text Embeddings has Minimal Impact on Retrieval and Classification Tasks", EMNLP 2025. PDF in `docs/paper.pdf`. Grading brief in `docs/assignment.pdf`. Every number we compare against is in `docs/PAPER_NOTES.md`; read it before writing any code.

## Your job

Build a scoped down, faithful, runnable reproduction of the paper's core claims plus a live demo app. Slides are made elsewhere from your `results/` and `figures/`, so those outputs must be clean, complete and self explanatory. Do not make slides.

Grading weight is 11/20 on reproduction, demo and viva. The rubric rewards: a working implementation that is more than calling a checkpoint, documented and justified scope reduction, real metrics tied to the paper's claims, explicit comparison with the paper's numbers, and the team being able to defend every implementation choice. Optimise for those.

## Hard rules

- Implement truncation, metrics, attribution and evaluation ourselves in numpy/torch. Pretrained encoders and HF datasets are fine as inputs. Do not use the authors' code (sotaro.io/papers/trunbed) as an implementation; you may read it only to confirm a protocol detail, and log anything you confirmed that way in `REPRO_NOTES.md`.
- Encode every corpus once, cache embeddings to `cache/` as float32 `.npy` with a metadata json (model, dataset, normalisation, prefix used). Every experiment after that is array slicing on cached embeddings. Nothing in the demo path should call an encoder except the optional live query box.
- Seeds fixed and logged. Random truncation uses 10 seeds as in the paper.
- Every experiment script writes machine readable output to `results/` (json or csv) and every figure to `figures/`. A figure must be reproducible from its csv alone.
- A local NVIDIA GPU is available; check `nvidia-smi` and VRAM at the start and run all encoding locally with fp16 where safe. The cached path (experiments and demo) must still work on CPU alone.
- Git: repo `embedding-truncation-repro`, already initialised by the user. Commit at each checkpoint with a short imperative message. Commit `results/` and `figures/`; never commit `cache/`, model weights, `.env` or tokens. Keep `.gitignore` current. Do not push; the user pushes.
- At each checkpoint below, record a short summary and the key numbers in `RESULTS.md`, then continue. Do not expand scope beyond this file.

## Code style (strict)

- No comments, no docstrings, no emojis, no print banners.
- Production grade, minimal, typed where useful, vectorised, no dead code, no notebooks.
- Small modules under `src/`, thin entry scripts under `scripts/`, one `run_all.sh`.
- `requirements.txt` pinned. Python 3.10+.

## Layout

```
docs/                  paper.pdf, assignment.pdf, PAPER_NOTES.md
src/
  data.py              NanoBEIR + MTEB classification loaders
  encode.py            model registry, pooling, prefixes, caching
  metrics.py           nDCG@10, accuracy, Spearman of rankings
  truncate.py          last-K, random-K, index-set removal, PCA
  attribution.py       leave-one-dimension-out, outliers, degrading sets
  theory.py            subsampling variance / rank flip model
  plotting.py          shared slide style
scripts/               one script per experiment, plus encode_all.py
cache/                 gitignored
results/               json/csv per experiment
figures/               png at 200 dpi plus pdf
demo/app.py            Streamlit demo
REPRO_NOTES.md         scope reductions, decisions, deviations, protocol details
RESULTS.md             our numbers vs paper numbers, one table per claim
run_all.sh
```

## Scope

### Models (sentence-transformers unless noted)

Core, must run:
1. `sentence-transformers/paraphrase-MiniLM-L3-v2` (384d, matches the paper's 17M Paraphrase-MiniLM)
2. `sentence-transformers/all-mpnet-base-v2` (768d, the paper's MPNet base)
3. `facebook/contriever` (768d, mean pooling, dot product, no normalisation; the paper's most fragile model, needed for the PCA reversal on BEIR)
4. `intfloat/e5-base-v2` (768d, needs `query: ` / `passage: ` prefixes; scaled down stand in for the paper's E5-large)

MRL baseline pair (the paper's limitation (iv), which they only checked informally):
5. `tomaarsen/mpnet-base-nli-matryoshka` vs `tomaarsen/mpnet-base-nli`. If either id fails to load, find the closest MRL / non-MRL pair with the same backbone and log it.

Document in `REPRO_NOTES.md` why we dropped E5-large, E5-Mistral (7B) and Sentence-T5.

### Data

- Retrieval: NanoBEIR from HF (`zeta-alpha-ai/Nano*`), all 13 subsets if encoding time allows, otherwise at least NanoSciFact, NanoNFCorpus, NanoFiQA2018, NanoArguAna, NanoSCIDOCS, NanoQuoraRetrieval. Metric nDCG@10 with graded qrels, implemented ourselves and sanity checked against `pytrec_eval` or the `mteb`/`ir_measures` value on one dataset.
- Classification: 4 MTEB English tasks: Banking77, Emotion, TweetSentimentExtraction, MTOPIntent (en). Protocol: logistic regression on embeddings following the MTEB classification protocol as closely as is cheap (train subset sampling, max_iter, n experiments). Write the exact protocol into `REPRO_NOTES.md`. Metric accuracy.

## Experiments, in order

### Phase 0: setup (checkpoint)
Environment, loaders, encoder registry, `scripts/encode_all.py`, cached embeddings for all models and datasets. Verify full dimension scores are in a sane range versus public MTEB/NanoBEIR numbers for the same models. Report full dimension scores table, encoding time, device.

### Phase 1: core claim (checkpoint)
E1. Last-K% and random-K% truncation, K in {0,10,...,90}, 10 seeds for random. Relative performance = truncated / full. Retrieval and classification. Reproduces Fig 2.
E2. Spearman correlation between full and truncated document rankings per query, averaged, last-K. Reproduces Fig 8.
E3. Random 50% truncation vs PCA to 50%. PCA fit on 20k sentences from `sentence-transformers/all-nli`. Reproduces Table 2.
E4. MRL vs non-MRL pair under last-K and random-K.
E5. Sanity check script proving that removing dimension i and zeroing dimension i give identical cosine and dot scores. This matters for the viva (the paper's justification for removal over zeroing is moot for cosine/dot similarity).

### Phase 2: explanation analyses (checkpoint)
E6. Outlier dimensions: average query embedding over NanoBEIR, outliers are dims beyond 3 sigma (paper's Kovaleva definition). Remove outliers vs 10 random sets of the same size. Reproduces Table 1.
E7. Leave-one-dimension-out attribution on NanoBEIR, vectorised: for cosine, score_without_i = (q.d - q_i d_i) / (sqrt(|q|^2 - q_i^2) sqrt(|d|^2 - d_i^2)); for dot product just subtract q_i d_i. Count degrading / improving / tie dims per model. Reproduces Fig 4a. For classification do it for 2 tasks and 2 models only (refit LR per dimension), run in background.
E8. Remove only degrading dims vs only improving dims vs last-K, ordered by attribution magnitude. Reproduces Fig 5 / Fig 12.
E9. Shared degrading dims across NanoBEIR datasets. Reproduces Fig 6.
E10. Anisotropy and dimensional collapse, cheap version: we do not retrain BERT/T5 contrastively. Instead compute IsoScore, uniformity loss and mean absolute inter dimension correlation for each of our models on 10k `sentence-transformers/simple-wiki` sentences, and correlate with each model's relative performance at 50%. Document this as a scope reduction.

### Phase 3: our critical extensions (checkpoint)
These are what make the viva and critique strong. Keep them rigorous and small.
X1. Selection leakage test. The paper identifies degrading dims and evaluates their removal on the same queries. Split each NanoBEIR dataset's queries 50/50 with a fixed seed (repeat over 5 splits). Identify degrading dims on split A, remove them, evaluate on split B. Compare to the same set protocol. Also report the correlation between per dimension deltas on A and on B.
X2. Significance. Bootstrap over queries (1000 resamples) for each dimension's delta; report how many degrading dims have a 95% CI excluding zero. Put the per dim delta size next to the query level nDCG standard error.
X3. Why random truncation works, a mechanism the paper does not give. For normalised embeddings, q.d is a sum of per dimension products. A random subset of m of n dims gives a scaled unbiased estimate of that sum, with variance set by the spread of the per dimension products (finite population correction). Implement in `src/theory.py`: predict the probability that two documents swap order, aggregate to a predicted top-10 overlap or Spearman, and compare predicted vs observed across K and models. Also compute a per model concentration measure of per dimension contributions (e.g. participation ratio) and test whether it predicts the 50% drop better than the anisotropy metrics from E10 (it should, and Contriever should be the outlier).

### Phase 4: stretch, your call
S1. LLM hidden state truncation on Qwen2.5-0.5B: drop first or last half of the final hidden state and the matching unembedding columns, score 200 HellaSwag and 200 MMLU items by log likelihood, and 50 GSM8K items by greedy generation. Reproduces the Table 3 pattern at small scale. Run it on the local GPU if Phases 0 to 3 are done and the GPU has the memory; otherwise skip and log why in `REPRO_NOTES.md`.

## Demo (build after Phase 1, extend after Phase 2/3)

`streamlit run demo/app.py`, fully offline from `cache/`, every interaction under 2 seconds on CPU.

- Sidebar: model, dataset, truncation method (last-K, random-K with seed, PCA, remove degrading, remove improving), K slider.
- Main panel: nDCG@10 or accuracy full vs truncated, relative %, mean Spearman, and the live curve with the current point highlighted against the paper's reported value for that setting.
- Query panel: pick or type a query, show top 5 documents full vs truncated side by side, with rank changes marked. Typing a new query is the only live encoder call.
- Tab with the leakage test (X1) and theory vs observed (X3) plots.
- Fallback: `python scripts/demo_cli.py --model ... --dataset ... --k 50` prints the same numbers if Streamlit misbehaves in the room.

## Figures spec (they go straight into slides)

- 16:9 friendly, 2400x1350 px at 200 dpi, white background, font size 16 or larger, no in-figure titles, axis labels with units, legend outside data.
- One fixed colour per model across every figure; degrading = blue, improving = orange, last-K = grey to match the paper.
- Each figure has a matching csv in `results/` with the exact plotted values.
- Required: fig_trunc_retrieval, fig_trunc_classification, fig_spearman, fig_pca_vs_random, fig_mrl, fig_outliers, fig_attribution_<model>, fig_degrading_removal, fig_shared_degrading, fig_anisotropy_vs_drop, fig_leakage, fig_bootstrap, fig_theory_vs_observed, fig_concentration_vs_drop.

## Deliverable docs

- `REPRO_NOTES.md`: every scope reduction with a one line justification, every protocol choice, deviations from the paper, compute used and runtimes, known weaknesses of our own setup.
- `RESULTS.md`: for each paper claim, our number, the paper's number, and one line on agreement. Include the full dimension scores table and all phase checkpoint summaries.
- Both written plainly, no em dashes or en dashes, no filler.

## Working style with the user

- Start by reading `docs/PAPER_NOTES.md`, `docs/assignment.pdf` and `docs/paper.pdf`. Write your plan and the exact model and dataset ids into `REPRO_NOTES.md`, then proceed.
- The user is away. Run all phases end to end without waiting for approval. At each checkpoint, append the summary and key numbers to `RESULTS.md`, commit, and continue.
- Long jobs (encoding, classification attribution) run in the background while you build the next independent piece.
- If a step fails, fix it and retry; if it is truly infeasible, take the closest faithful alternative, log the reason in `REPRO_NOTES.md`, and move on. Never leave the run blocked on a question.
- Before finishing, run `run_all.sh` from a clean cache-only state to confirm the cached path and the demo start without errors, then write a final summary at the top of `RESULTS.md` listing what was done, what was skipped and why.
- If something in this plan is wrong, apply the fix and record what changed and why under a "Deviations" heading in `REPRO_NOTES.md`. Never change the plan silently.
