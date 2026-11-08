# Results

## Summary

Done: all of Phase 0 to 4 in CLAUDE.md. 6 encoders (4 core plus an MRL / non-MRL MPNet pair) on 13 NanoBEIR datasets and 4 MTEB classification tasks; experiments E1 to E10, extensions X1 to X3, S1 on Qwen2.5-0.5B; Streamlit demo and CLI fallback. Every number below comes from one clean run of `run_all.sh` on CPU from the cached embeddings (S1 ran separately on the local GPU). Skipped: E5-large, E5-Mistral, Sentence-T5, full BEIR, 8 of 12 MTEB tasks, and retraining BERT/T5 for anisotropy; reasons are in `REPRO_NOTES.md`.

Headline findings:

1. The core claim reproduces. Random 50% removal keeps 88.7 to 98.3% of NanoBEIR nDCG@10 (paper 87.3 to 97.7) and 93.1 to 97.0% of accuracy (paper 95.4 to 97.7). Contriever is again the fragile outlier.
2. The paper's explanation (many "degrading" dimensions) does not survive a held out test. Removing degrading dims gains up to 2.7% only on the queries used to find them; on held out queries the best gain is +0.3% and the per dimension deltas of two disjoint query halves are uncorrelated (r between -0.05 and 0.01). Only 8 to 22 degrading dims per model have a bootstrap CI excluding zero, close to the 10 to 19 expected by chance, and the median per dimension effect (3 to 6e-4) is about 25 times smaller than the standard error of NanoBEIR nDCG@10 (0.012).
3. A sampling view explains the robustness. A random subset of dimensions is an unbiased, scaled estimate of the dot product, so rankings survive when contributions are spread over many dimensions. The model predicts observed Spearman at 50% within 0.03 for 5 of 6 models, and Contriever has by far the most concentrated contributions (participation ratio 0.04 vs 0.19 to 0.92). Across only 6 models, however, the participation ratio is not a better single predictor of the 50% drop than IsoScore (r 0.50 vs -0.76); both are driven mostly by Contriever.

## Phase 0: full dimension scores

| Model | NanoBEIR nDCG@10 ours | Paper (Fig 4a) | Mean accuracy, 4 tasks |
|---|---|---|---|
| Para-MiniLM-L3 | 0.4871 | 0.4835 | 0.586 |
| MPNet base | 0.5807 | 0.5755 | 0.617 |
| Contriever | 0.5283 | 0.524 | 0.585 |
| E5-base-v2 | 0.6168 | 0.5765 (E5-large) | 0.624 |
| MPNet NLI MRL | 0.3945 | | 0.641 |
| MPNet NLI | 0.4047 | | 0.626 |

Our full scores are within 0.005 of the paper for the three models we share. E5-base-v2 beats the paper's E5-large, which suggests the paper may not have used E5's `query:` / `passage:` prefixes. Per dataset: `results/e0_full_scores.csv`. Our nDCG@10 matches pytrec_eval exactly on NanoSciFact; on NanoArguAna one query differs because of tied scores between duplicate documents (`results/e0_pytrec_check.json`). Encoding: RTX 4060 Laptop, fp16, 50% duty throttle, 2 to 14 min per model (`results/e0_encoding_times.csv`).

## C1. Random 50% removal keeps over 90% (retrieval) and 95% (classification)

| Model | NanoBEIR ours | Paper BEIR | MTEB ours | Paper MTEB |
|---|---|---|---|---|
| MPNet | 98.3 ± 0.6 | 97.7 | 95.5 ± 0.4 | 96.6 |
| Contriever | 88.7 ± 1.6 | 87.3 | 93.1 ± 0.4 | 95.4 |
| E5 (base vs paper large) | 95.5 ± 1.0 | 95.9 | 97.0 ± 0.4 | 97.7 |
| Para-MiniLM | 94.1 ± 0.8 | 94.6 | 96.8 ± 0.4 | 97.4 |

± is the std over 10 random seeds. Agreement: retrieval within 1.4 points for every model. Classification is 0.6 to 2.3 points lower, consistent with MTEB's few shot protocol (8 to 32 examples per label) on 4 of the 12 tasks. As in the paper, Contriever fails the "over 90%" retrieval bar; it also misses ">95%" for classification by 2 points. Figures: `fig_trunc_retrieval`, `fig_trunc_classification`.

## C2. Last-K truncation

| Model | NanoBEIR 50% | 80% | 90% | MTEB 50% | 80% | 90% |
|---|---|---|---|---|---|---|
| MPNet | 97.5 | 92.9 | 83.0 | 95.3 | 88.4 | 80.4 |
| E5 | 95.9 | 84.5 | 65.8 | 96.4 | 89.9 | 83.3 |
| Para-MiniLM | 94.8 | 80.4 | 64.9 | 96.9 | 87.8 | 77.9 |
| Contriever | 88.3 | 68.1 | 43.4 | 93.4 | 83.0 | 72.1 |

Agreement: as in the paper, relative performance stays above 80% up to 80% removal for every model except Contriever, which degrades fastest on retrieval (paper: about 5 to 35% at 90% on most BEIR datasets; ours 43% on average).

## C3. Spearman between full and truncated rankings (Fig 8)

| Model | Ours 50% | Paper 50% | Ours 90% | Paper 90% |
|---|---|---|---|---|
| MPNet | 0.899 | 0.90 | 0.592 | 0.59 |
| Para-MiniLM | 0.885 | 0.88 | 0.559 | 0.56 |
| E5 (base vs large) | 0.884 | 0.88 | 0.541 | 0.55 |
| Contriever | 0.788 | 0.85 | 0.411 | 0.43 |

Agreement: within 0.01 everywhere except Contriever at 50% (0.06 lower; still the lowest model, as in the paper). Figure `fig_spearman`.

Check on the Contriever gap: rerunning Contriever with cosine instead of dot gives Spearman 0.847 (paper 0.85), but its full nDCG@10 drops to 0.451 (paper 0.524), random 50% rises to 94.8% (paper 87.3) and PCA to 100.5% (paper 77.3). Dot product matches the paper on three of four numbers, so the paper used dot product for Contriever and the remaining Spearman gap is most likely a detail of how Fig 8 computes the ranking correlation (for example over a top-k list rather than the whole corpus).

## C4, C5. Anisotropy and dimensional collapse

Cross model version on 6 models, 10k simple-wiki sentences (`results/e10_*.csv`, `fig_anisotropy_vs_drop`):

| Metric | Pearson r with 50% last-K | with 50% random-K | Paper (within model, over checkpoints) |
|---|---|---|---|
| IsoScore | -0.93 (p 0.007) | -0.76 (p 0.08) | T5 -0.47, BERT 0.39 |
| Uniformity loss | -0.15 | -0.14 | T5 0.36, BERT -0.35 |
| Mean abs inter dim correlation | 0.87 (p 0.02) | 0.64 | T5 0.45, BERT -0.30 |

Agreement: partial. Uniformity is uncorrelated, as in the paper. IsoScore and inter dimension correlation correlate strongly across models, in the direction opposite to the "anisotropy wastes space" intuition (the most isotropic model, Contriever, is the most fragile). With n = 6 and Contriever as a single high leverage point, the Spearman correlations (-0.71, 0.60) are not significant, so we read this as "not explained by anisotropy" rather than as a counter claim. The design differs from the paper's (across models vs across checkpoints of one model).

## C6. Outlier dimensions (Table 1)

| Model | # outliers ours / paper | Full | Without outliers | Without 10 random sets |
|---|---|---|---|---|
| Para-MiniLM | 2 / 2 | 0.4873 | 0.4856 | 0.4875 ± 0.0008 |
| MPNet | 4 / 5 | 0.5807 | 0.5823 | 0.5808 ± 0.0007 |
| Contriever | 3 / 3 | 0.5284 | 0.5301 | 0.5280 ± 0.0012 |
| E5 (base vs large) | 1 / 1 | 0.6170 | 0.6175 | 0.6169 ± 0.0006 |

Agreement: outlier counts match for 3 of 4 models; removing outliers moves nDCG@10 by under 0.002, about two random set standard deviations, as in the paper.

## C7. Many degrading dimensions (Fig 4)

| Model | NanoBEIR degrading / improving / tie, ours | Paper |
|---|---|---|
| Para-MiniLM | 230 / 154 / 0 | 117 / 267 / 0 |
| MPNet | 382 / 382 / 4 | 595 / 169 / 4 |
| Contriever | 341 / 427 / 0 | 498 / 270 / 0 |
| E5 (base vs large) | 244 / 524 / 0 | 308 / 716 / 0 |

Classification (2 models x 2 tasks, LR refit per removed dimension, `results/e7_attribution_classification.csv`): Para-MiniLM Banking77 29 / 348 / 7, Emotion 247 / 124 / 13; MPNet Banking77 222 / 464 / 82, Emotion 369 / 353 / 46.

Agreement: every model has hundreds of degrading dims, as the paper reports, but the degrading vs improving split does not match the paper for any model and flips between tasks. X1 and X2 show why: the per dimension effects are at noise level, so the split is close to a coin flip per dimension. Figures `fig_attribution_<model>`.

## C8. Removing only degrading dims (Fig 5, 12)

Best relative NanoBEIR when removing degrading dims in 5% steps, same queries used to find them: Contriever 103.7%, E5 101.9%, MPNet 101.3%, Para-MiniLM 101.2%. Removing improving dims drops faster than last-K for every model (`results/e8_degrading_removal.csv`, `fig_degrading_removal`). Agreement: reproduces the paper's in sample pattern, including gains above 100%. X1 shows the gains do not transfer.

## C10. Shared degrading dims (Fig 6)

No dimension is degrading on all 13 datasets for any model (paper: none). For MPNet, the share of dims degrading on exactly k datasets peaks at k = 4 (24%) and k = 5 (18%); paper: more than 20% on 5 datasets. Agreement: yes, same shape (`results/e9_shared_degrading.csv`, `fig_shared_degrading`).

## C11. Random 50% vs PCA 50% (Table 2)

| Model | BEIR trunc ours / paper | BEIR PCA ours / paper | MTEB trunc ours / paper | MTEB PCA ours / paper |
|---|---|---|---|---|
| MPNet | 98.3 / 97.7 | 99.6 / 99.4 | 95.5 / 96.6 | 99.7 / 99.6 |
| Contriever | 88.7 / 87.3 | 85.8 / 77.3 | 93.1 / 95.4 | 96.4 / 98.5 |
| E5 (base vs large) | 95.5 / 95.9 | 101.7 / 90.6 | 97.0 / 97.7 | 99.0 / 99.8 |
| Para-MiniLM | 94.1 / 94.6 | 99.2 / 99.3 | 96.8 / 97.4 | 99.3 / 99.5 |

Agreement: PCA beats random truncation everywhere except Contriever retrieval, where random truncation wins, exactly the paper's exception. Our Contriever PCA drop is smaller (85.8 vs 77.3). E5 differs: our E5-base gains from PCA, while the paper's E5-large lost 9 points. Figure `fig_pca_vs_random`.

## C12. LLM hidden state truncation (Table 3), Qwen2.5-0.5B

| Task | Full | First half removed | Last half removed | Paper Qwen2.5-7B (full / first / last) |
|---|---|---|---|---|
| MMLU acc, 200 items, 0-shot | 0.480 | 0.415 (86%) | 0.365 (76%) | 0.718 / 0.709 / 0.709 |
| HellaSwag acc, 200 items | 0.370 | 0.280 (76%) | 0.320 (86%) | 0.718 / 0.709 / 0.709 (copied from MMLU in the paper) |
| HellaSwag acc_norm | 0.500 | 0.405 (81%) | 0.375 (75%) | |
| GSM8K strict EM, 50 items, 5-shot | 0.320 | 0.000 | 0.000 | 0.766 / 0.045 / 0.011 |

Agreement: same pattern at small scale. Multiple choice log likelihood tasks keep 75 to 86% of performance, while generation (GSM8K) collapses to 0. The 0.5B model keeps less than the paper's 7B on MMLU, consistent with less redundancy in a smaller hidden state (896 dims). `results/s1_llm.csv`.

## E4. MRL vs non-MRL (paper limitation iv)

| Model | NanoBEIR random 50% | last 50% | last 90% | MTEB random 50% | last 90% |
|---|---|---|---|---|---|
| MPNet NLI MRL | 95.9 | 99.1 | 77.4 | 98.1 | 87.8 |
| MPNet NLI | 94.8 | 97.0 | 68.6 | 97.9 | 81.7 |

MRL helps only for last-K, the truncation it was trained for (+9 points at 90% on retrieval, +6 on classification). Under random-K the two models are within 1.1 points at 50%. This matches the paper's informal remark that MRL and non-MRL MPNet behave comparably under random removal, and adds that the MRL advantage is specific to prefix truncation. `fig_mrl`.

## E5. Removing vs zeroing a dimension

Max absolute difference between removing and zeroing the same dimensions, cosine and dot, synthetic data and all 6 cached models: 9.9e-14 (`results/e5_remove_vs_zero.json`). The two are identical for both similarity functions, so the paper's stated reason for removing rather than zeroing does not apply to its own setup.

## Phase 3 extensions

### X1. Selection leakage (5 random 50/50 query splits)

| Model | Peak in sample (%) | Held out at that K (%) | Best held out (%) | Corr of per dim deltas, split A vs B |
|---|---|---|---|---|
| Para-MiniLM | 101.9 | 100.2 | 100.2 | -0.03 ± 0.03 |
| MPNet | 101.1 | 99.7 | 100.1 | -0.05 ± 0.02 |
| Contriever | 102.7 | 97.9 | 100.3 | 0.01 ± 0.02 |
| E5 | 101.3 | 99.2 | 100.0 | 0.00 ± 0.04 |

Removing "degrading" dims chosen on half the queries does not help the other half: held out curves track last-K (`fig_leakage`). The above 100% results in the paper's Fig 5 and Fig 12 are consistent with selection on test noise.

### X2. Significance (1000 bootstrap resamples over queries, stratified by dataset)

| Model | Degrading dims | CI excludes 0 | Expected by chance (2.5% per side) | Median abs delta | SE of NanoBEIR mean nDCG@10 |
|---|---|---|---|---|---|
| Para-MiniLM | 230 | 8 | 9.6 | 6e-4 | 0.013 |
| MPNet | 382 | 22 | 19.2 | 3e-4 | 0.013 |
| Contriever | 341 | 8 | 19.2 | 5e-4 | 0.012 |
| E5 | 244 | 8 | 19.2 | 5e-4 | 0.012 |

The number of significant degrading dims is at or below chance level for every model (`fig_bootstrap`).

### X3. Sampling theory vs observation

Spearman at random 50% on 300 sampled documents per query, predicted vs observed: Para-MiniLM 0.890 vs 0.886, MPNet 0.905 vs 0.897, Contriever 0.786 vs 0.760, MRL 0.926 vs 0.923, non-MRL 0.914 vs 0.913, E5 0.789 vs 0.881. The theory fits within 0.03 for 5 of 6 models across K (`fig_theory_vs_observed`). It underpredicts E5 because it ignores renormalisation after truncation, which matters when a few dimensions carry a large shared component.

Participation ratio of per dimension score contributions: Para-MiniLM 0.92, E5 0.78, MPNet 0.66, MRL 0.21, non-MRL 0.19, Contriever 0.04. Correlation with relative nDCG@10 at random 50% (n = 6): participation ratio 0.50, predicted Spearman 0.61, IsoScore -0.76, inter dim correlation 0.64 (`results/x3_correlations.csv`, `fig_concentration_vs_drop`). Our prediction in CLAUDE.md, that concentration would beat the anisotropy metrics, is not supported with 6 models: every metric's correlation is driven by Contriever, and none separates the other five models.

## Phase checkpoints

- Phase 0: 6 models x 13 NanoBEIR + 4 MTEB tasks + 2 sentence sets encoded (166,078 texts per model). Full scores within 0.005 of the paper's.
- Phase 1: C1, C2, C3 and C11 reproduce, with Contriever as the fragile model. Removing and zeroing are identical. MRL only helps last-K.
- Phase 2: outlier counts and effects match Table 1. Degrading dims are numerous but their identity does not match the paper's counts. In sample degrading removal exceeds 100%. Anisotropy shows no clean explanation.
- Phase 3: degrading dim gains vanish on held out queries; significance is at chance level; the sampling theory predicts rank stability for 5 of 6 models.
- Phase 4: the Qwen2.5-0.5B pattern matches Table 3 (multiple choice survives, GSM8K collapses).
