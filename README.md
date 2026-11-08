# Random Truncation of Text Embeddings: a Reproduction

A scoped down, faithful reproduction of

> Sotaro Takeshita, Yurina Takeshita, Daniel Ruffinelli and Simone Paolo Ponzetto. *Randomly Removing 50% of Dimensions in Text Embeddings has Minimal Impact on Retrieval and Classification Tasks.* EMNLP 2025.

Prepared for CS F429 (Introduction to NLP), BITS Pilani Hyderabad, Group 12, paper 14. The repository contains our own implementation of every truncation method, metric, attribution analysis and evaluation in the paper, the cached embeddings needed to rerun all of it offline, three extension experiments that test the paper's explanation, and an interactive demo.

## 1. The paper in brief

The paper removes K% of the dimensions of sentence embeddings, either the last K% or a random K%, and measures how much retrieval (nDCG@10) and classification (accuracy) survive. Its main claim is that removing a random half of the dimensions costs less than 10% on retrieval and less than 5% on classification. It then tests three explanations (anisotropy, dimensional collapse, outlier dimensions), rejects all three, and proposes its own: leave-one-dimension-out attribution finds many "degrading" dimensions whose removal improves performance, so random removal discards good and bad dimensions alike.

## 2. What this repository reproduces

| Paper element | Our experiment | Status |
|---|---|---|
| Fig 2: last-K and random-K truncation | E1 | Reproduced |
| Fig 8: Spearman between full and truncated rankings | E2 | Reproduced |
| Table 2: random 50% vs PCA 50% | E3 | Reproduced |
| Limitation (iv): MRL vs non-MRL models | E4 | New result |
| Justification for removing rather than zeroing | E5 | Tested: the two are identical |
| Table 1: outlier dimensions | E6 | Reproduced |
| Fig 4: leave-one-dimension-out attribution | E7 | Reproduced (retrieval; classification for 2 models x 2 tasks) |
| Fig 5, Fig 12: removing only degrading or improving dims | E8 | Reproduced in sample |
| Fig 6: degrading dims shared across datasets | E9 | Reproduced |
| Sec 3.1, 3.2: anisotropy and dimensional collapse | E10 | Cross model version (no retraining) |
| Table 3: LLM hidden state truncation | S1 | Reproduced on Qwen2.5-0.5B |
| Not in the paper | X1 selection leakage, X2 significance, X3 sampling theory | Our extensions |

## 3. Headline results

| Claim | Ours | Paper |
|---|---|---|
| Random 50%, NanoBEIR nDCG@10 kept | 88.7 to 98.3% | 87.3 to 97.7% |
| Random 50%, MTEB accuracy kept | 93.1 to 97.0% | 95.4 to 97.7% |
| Full nDCG@10, MPNet / Contriever / MiniLM | 0.581 / 0.528 / 0.487 | 0.576 / 0.524 / 0.484 |
| Spearman at 50% last-K, MPNet / MiniLM | 0.899 / 0.885 | 0.90 / 0.88 |
| Outlier dims, MiniLM / MPNet / Contriever / E5 | 2 / 4 / 3 / 1 | 2 / 5 / 3 / 1 |
| PCA beats random 50% except Contriever retrieval | yes | yes |
| Removing degrading dims exceeds 100% | yes, in sample (up to 103.7%) | yes (up to about 103%) |

Our three main findings:

1. **The core claim reproduces.** Random removal of half the dimensions keeps most of the performance, with Contriever as the fragile outlier, as in the paper.
2. **The paper's explanation does not survive a held out test (X1, X2).** Degrading dimensions chosen on one half of the queries give no gain on the other half (best held out result 100.3%), and their per dimension effects on two disjoint query halves are uncorrelated (r between -0.05 and 0.01). The number of degrading dimensions with a bootstrap confidence interval excluding zero is at the level expected by chance, and the median effect of one dimension (3 to 6e-4 nDCG@10) is about 25 times smaller than the standard error of the benchmark score (0.012).
3. **A simpler mechanism explains the robustness (X3).** A score q.d is a sum of per dimension products, and a random subset of dimensions gives an unbiased, scaled estimate of that sum. Rankings survive whenever contributions are spread over many dimensions. This model predicts the observed rank correlation at 50% within 0.03 for 5 of 6 models, and identifies Contriever as the model with the most concentrated contributions.

Every claim with our number, the paper's number and a note on agreement is in [RESULTS.md](RESULTS.md).

## 4. Scope

| Paper | This reproduction |
|---|---|
| 6 encoders | 4 core encoders (Paraphrase-MiniLM-L3, all-mpnet-base-v2, Contriever, E5-base-v2) plus an MRL / non-MRL MPNet pair |
| 14 BEIR datasets | All 13 NanoBEIR datasets |
| 12 MTEB classification tasks | Banking77, Emotion, TweetSentimentExtraction, MTOPIntent (en) |
| Contrastive retraining of BERT and T5 for anisotropy | Cross model comparison of the same metrics on our 6 models |
| Llama 3.1 8B, Qwen 2.5 7B | Qwen2.5-0.5B |

Every reduction is justified, and every protocol choice and deviation is recorded, in [REPRO_NOTES.md](REPRO_NOTES.md).

## 5. Method and implementation

All truncation, metrics, attribution and evaluation code is our own (numpy and torch). Pretrained encoders and Hugging Face datasets are used only as inputs. The authors' code was not used.

**Encoding.** Each corpus is encoded once per model with sentence-transformers (fp16 on GPU) and cached as float32 `.npy` files with a metadata file recording model, dataset, prefix and normalisation. E5 uses its `query: ` and `passage: ` prefixes. All later experiments only slice these arrays.

**Truncation** (`src/truncate.py`). Last-K keeps the first D minus round(K x D) dimensions. Random-K draws round(K x D) dimensions to remove with a seeded generator, over 10 seeds (0 to 9) as in the paper. PCA is our own SVD on 20,000 all-nli sentences, keeping D/2 components.

**Similarity.** Cosine for all models except Contriever, which uses the raw dot product. Cosine models are renormalised after truncation by the cosine itself.

**Metrics** (`src/metrics.py`). nDCG@10 with linear gain, as in pytrec_eval and BEIR, excluding a document whose id equals the query id, as BEIR and MTEB do. It matches pytrec_eval to 4e-8 per query. Classification follows the MTEB protocol: 10 experiments, each fitting a logistic regression (max_iter 100) on a class balanced subsample of the training set (8, 16 or 32 examples per label depending on the task), scored by accuracy on the test set. The same 10 subsamples are reused for every truncation setting.

**Attribution** (`src/attribution.py`). Leave-one-dimension-out is computed in closed form for all dimensions at once instead of rescoring the corpus D times. For cosine,

    score_without_i = (q.d - q_i d_i) / ( sqrt(|q|^2 - q_i^2) * sqrt(|d|^2 - d_i^2) )

and for the dot product the term q_i d_i is simply subtracted. A dimension is degrading if removing it raises the NanoBEIR mean nDCG@10.

**Sampling theory** (`src/theory.py`). For a document pair (a, b) and query q, let r_j = q_j (a_j - b_j). Keeping a random m of n dimensions estimates the score difference sum_j r_j with variance n^2 sigma_r^2 / m x (n - m) / (n - 1) (sampling without replacement). The probability that the pair swaps order follows from a normal approximation; averaging over pairs gives a predicted Kendall tau, converted to Spearman through the Gaussian relation.

## 6. Repository layout

```
docs/                     paper, assignment brief, reference numbers (PAPER_NOTES.md)
src/
  data.py                 NanoBEIR, MTEB classification and sentence set loaders
  encode.py               model registry, prefixes, encoding and caching
  metrics.py              nDCG@10, Spearman, MTEB logistic regression protocol
  truncate.py             last-K, random-K, index removal, PCA
  evaluate.py             retrieval and classification evaluators over cached arrays
  attribution.py          leave-one-dimension-out, outliers, anisotropy metrics
  theory.py               subsampling variance and rank swap model
  paper.py                every reference number taken from the paper
  plotting.py             shared figure style and model colours
scripts/
  encode_all.py           encode every model and dataset into cache/
  e0_full_scores.py       full dimension scores and pytrec_eval check
  e1 ... e10              paper experiments (section 2)
  x1, x2, x3              extensions (section 2)
  s1_llm_truncation.py    Qwen2.5-0.5B hidden state truncation
  demo_cli.py             command line fallback for the demo
demo/app.py, demo/core.py Streamlit demo
results/                  machine readable output of every experiment (csv, json)
figures/                  every figure as 200 dpi png (2400 x 1350) and pdf
run_all.sh                regenerates all results and figures
RESULTS.md                our numbers vs the paper, claim by claim
REPRO_NOTES.md            scope reductions, protocol, deviations, runtimes, weaknesses
```

## 7. Setup

Python 3.12 or later (the pinned numpy 2.5.2 requires it; tested on 3.13). An NVIDIA GPU is used only for encoding and S1; everything else runs on CPU.

```bash
pip install -r requirements.txt
```

The cache (`cache/`, about 2.7 GB) is not tracked by git. On a fresh clone, `run_all.sh` encodes everything first (about 50 minutes on a laptop RTX 4060, longer on CPU).

## 8. Running

Reproduce every result and figure (encodes first if `cache/emb/READY` is missing; about 1.5 hours on CPU once the cache exists):

```bash
bash run_all.sh
```

Include the Qwen2.5-0.5B experiment, which needs a GPU and downloads the model and benchmark items:

```bash
RUN_S1=1 bash run_all.sh
```

Run the interactive demo (offline, CPU only; the first load takes about 30 seconds):

```bash
streamlit run demo/app.py
```

The sidebar selects the model, dataset, truncation method (last-K, random-K with seed, PCA, remove degrading, remove improving) and K. The main panel shows full and truncated scores, relative performance, rank correlation, the truncation curve against the paper's value, and the top 5 documents for a chosen or typed query, before and after truncation. Typing a new query is the only step that calls an encoder. A second tab shows the leakage and theory results.

Command line fallback with the same numbers:

```bash
python scripts/demo_cli.py --model mpnet --dataset NanoSciFact --k 50 --method random --seed 0
```

## 9. Outputs

Each figure is generated from the csv listed next to it, which contains the plotted values.

| Figure | Data |
|---|---|
| fig_trunc_retrieval | results/fig_trunc_retrieval.csv |
| fig_trunc_classification | results/fig_trunc_classification.csv |
| fig_spearman | results/fig_spearman.csv |
| fig_pca_vs_random | results/fig_pca_vs_random.csv |
| fig_mrl | results/fig_mrl.csv |
| fig_outliers | results/e6_outliers.csv |
| fig_attribution_{model} | results/e7_attribution_retrieval.csv |
| fig_degrading_removal | results/e8_degrading_removal.csv |
| fig_shared_degrading | results/e9_shared_degrading.csv |
| fig_anisotropy_vs_drop | results/e10_anisotropy.csv, results/e10_correlations.csv |
| fig_leakage | results/x1_leakage_plot.csv |
| fig_bootstrap | results/x2_bootstrap_dims.csv, results/x2_bootstrap.csv |
| fig_theory_vs_observed | results/x3_theory_vs_observed.csv |
| fig_concentration_vs_drop | results/x3_concentration.csv, results/x3_correlations.csv |

Other results: `e0_*` (full scores, encoding times, metric check), `e1_*` (all truncation runs per dataset and seed), `e2_*` (Spearman per dataset), `e3_pca.csv`, `e4_mrl.csv`, `e5_remove_vs_zero.json`, `e7_attribution_classification.csv`, `x1_*`, `x2_*`, `x3_*`, `s1_*` (LLM results and generated GSM8K answers).

## 10. Key design decisions

- **NanoBEIR instead of full BEIR.** The paper itself uses NanoBEIR for its largest model and for all attribution analyses, and its 13 datasets cover every BEIR domain at 50 queries each.
- **Closed form attribution.** Removing a dimension changes q.d by exactly q_i d_i, so all D leave-one-out scores follow from one score matrix; this makes E7, X1 and X2 take minutes instead of hours.
- **Removal equals zeroing.** For cosine and dot similarity, a removed coordinate and a zeroed coordinate contribute nothing to the dot product or the norms, so the two give identical scores (E5 confirms this to 1e-13). The paper's stated reason for preferring removal therefore does not apply.
- **Dot product for Contriever.** Checked against the paper: dot product reproduces its full score, random 50% and PCA numbers, while cosine does not (RESULTS.md, C3).
- **One device for all reported numbers.** CPU and GPU arithmetic differ by up to 2e-4 nDCG@10, the same size as single dimension effects, so every reported number comes from one CPU run.
- **Held out evaluation (X1).** Selecting dimensions and evaluating on the same queries cannot separate a real effect from selection on noise; splitting queries can.

## 11. Limitations

Four of twelve MTEB tasks, 50 queries per NanoBEIR dataset, a base sized stand in for E5-large, a 0.5B model for the LLM experiment, and cross model correlations over only 6 models. Details and further weaknesses are listed in [REPRO_NOTES.md](REPRO_NOTES.md).

## 12. Hardware

Developed and run on a laptop: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), 16 logical CPU cores, 16 GB RAM, Windows 11, Python 3.13. Per step runtimes are in REPRO_NOTES.md.
