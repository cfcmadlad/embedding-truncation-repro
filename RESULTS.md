# Results

## Summary

Done: Phase 0 (6 models encoded, 13 NanoBEIR sets, 4 MTEB tasks), Phase 1 E1, E2, E5, Phase 2 E6 to E9, Phase 3 X1 to X3, demo and CLI. In progress when this was written: E3 (PCA), E4 (MRL), E7 classification attribution, E10, S1 (Qwen2.5-0.5B); their csv and figures land in `results/` and `figures/`. Not yet done: clean rerun of `run_all.sh` from cache. Skipped models and why: see `REPRO_NOTES.md`.

Headline findings:
1. The core claim reproduces. Random 50% removal keeps 88.7 to 98.3% of NanoBEIR nDCG@10 (paper 87.3 to 97.7) and 93.1 to 97.0% of accuracy (paper 95.4 to 97.7). Contriever is again the fragile outlier.
2. The paper's explanation does not survive a held out test. Removing "degrading" dimensions gains up to 2.8% only on the queries used to find them; on held out queries the gain vanishes (at most +0.4%) and the A vs B per dimension delta correlation is about 0. Bootstrap: significant degrading dims are near the chance count, and the median per dimension effect (3 to 6e-4) is about 20 times smaller than the query level standard error of NanoBEIR nDCG@10 (0.012).
3. A sampling mechanism explains the robustness: a random subset of dimensions is an unbiased estimate of the dot product. Predicted Spearman at 50% matches observed within 0.03 for 5 of 6 models, and Contriever has by far the most concentrated per dimension contributions (participation ratio 0.04 vs 0.19 to 0.92).

## Phase 0: full dimension scores

NanoBEIR mean nDCG@10: MiniLM 0.4871 (paper Para-MiniLM 0.4835), MPNet 0.5807 (paper 0.5755), Contriever 0.5283 (paper 0.524), E5-base-v2 0.6168 (paper E5-large 0.5765), MRL 0.3945, non-MRL 0.4047. Mean accuracy over our 4 tasks: 0.586, 0.617, 0.585, 0.624, 0.641, 0.626. Per dataset: `results/e0_full_scores.csv`. Metric check vs pytrec_eval: `results/e0_pytrec_check.json`. Encoding times: `results/e0_encoding_times.csv` (RTX 4060 Laptop, fp16, 50% duty throttle, 2 to 14 min per model).

## C1. Random 50% removal (Fig 2c, 2d; Table 2 Trunc columns)

| Model | NanoBEIR ours | Paper BEIR | MTEB ours | Paper MTEB |
|---|---|---|---|---|
| MPNet | 98.3 ± 0.6 | 97.7 | 95.5 ± 0.4 | 96.6 |
| Contriever | 88.7 ± 1.6 | 87.3 | 93.1 ± 0.4 | 95.4 |
| E5 (base vs paper large) | 95.5 ± 1.0 | 95.9 | 97.0 ± 0.4 | 97.7 |
| Para-MiniLM | 94.1 ± 0.8 | 94.6 | 96.8 ± 0.4 | 97.4 |

Agreement: retrieval within 1.4 points for every model. Classification is 0.6 to 2.3 points lower, consistent with MTEB's few shot logistic regression (8 to 32 examples per label) on 4 of the 12 tasks; Contriever misses the paper's ">95%" by 2 points.

## C2. Last-K (Fig 2a, 2b)

Relative NanoBEIR at 80% last-K: MPNet 92.9, E5 84.5, MiniLM 80.4, Contriever 68.1; at 90%: 83.0, 65.8, 64.9, 43.4. Agreement: as in the paper, only the fragile model (Contriever) drops below 80% before 80% removal; Contriever degrades fastest. Figure: `fig_trunc_retrieval`, `fig_trunc_classification`.

## C3. Spearman of rankings (Fig 8)

| Model | Ours 50% | Paper 50% | Ours 90% | Paper 90% |
|---|---|---|---|---|
| MPNet | 0.899 | 0.90 | 0.592 | 0.59 |
| MiniLM | 0.885 | 0.88 | 0.559 | 0.56 |
| E5 | 0.884 | 0.88 | 0.541 | 0.55 |
| Contriever | 0.788 | 0.85 | 0.411 | 0.43 |

Agreement: within 0.01 except Contriever at 50% (0.06 lower). Figure `fig_spearman`.

## C6. Outliers (Table 1)

| Model | # outliers ours / paper | Without outliers | Without random sets |
|---|---|---|---|
| MiniLM | 2 / 2 | 0.4856 | 0.4875 ± 0.0008 |
| MPNet | 4 / 5 | 0.5823 | 0.5808 ± 0.0007 |
| Contriever | 3 / 3 | 0.5301 | 0.5280 ± 0.0012 |
| E5 | 1 / 1 | 0.6175 | 0.6169 ± 0.0006 |

Agreement: outlier counts match; removing them changes nDCG by under 0.002, as in the paper.

## C7. Degrading dims (Fig 4a)

Degrading / improving / tie on NanoBEIR: MiniLM 230/154/0 (paper 117/267/0), MPNet 382/382/4 (paper 595/169/4), Contriever 341/427/0 (paper 498/270/0), E5 244/524/0 (paper E5-large 308/716/0). Agreement: many degrading dims in every model, but the split differs from the paper for each model, which is what X1 and X2 predict for noise level effects.

## C8. Removing only degrading dims (Fig 5, 12)

See `results/e8_degrading_removal.csv`, `fig_degrading_removal`. In sample we reproduce gains above 100%; X1 shows they do not transfer to held out queries.

## C10, C11, C4, C5, C12

Shared degrading dims: `results/e9_shared_degrading.csv`, `fig_shared_degrading`. PCA vs random (Table 2): `results/e3_pca.csv` (pending at time of writing). Anisotropy and collapse: `results/e10_anisotropy.csv`, `results/e10_correlations.csv` (pending). LLM truncation: `results/s1_llm.csv` (pending).

## E5. Removing vs zeroing

Max absolute difference between removing and zeroing a set of dimensions, cosine and dot, over synthetic and all cached models: 9.9e-14. They are identical, so the paper's reason for preferring removal is moot.

## Phase 3 extensions

| Model | Peak in sample (%) | Held out at that K (%) | Best held out (%) | Corr of deltas A vs B | Sig. degrading / chance | Median abs delta | SE of NanoBEIR mean |
|---|---|---|---|---|---|---|---|
| MiniLM | 101.9 | 100.2 | 100.2 | -0.03 | 19 / 9.6 | 6e-4 | 0.013 |
| MPNet | 101.1 | 99.6 | 100.1 | -0.05 | 22 / 19.2 | 3e-4 | 0.013 |
| Contriever | 102.8 | 97.9 | 100.4 | 0.01 | 10 / 19.2 | 5e-4 | 0.012 |
| E5 | 101.3 | 99.2 | 100.0 | 0.00 | 18 / 19.2 | 5e-4 | 0.012 |

X3 theory, Spearman at random 50% on sampled documents, predicted vs observed: MiniLM 0.890 vs 0.886, MPNet 0.905 vs 0.897, Contriever 0.786 vs 0.761, MRL 0.926 vs 0.923, non-MRL 0.914 vs 0.913, E5 0.789 vs 0.881 (the approximation ignores renormalisation after truncation and fails for E5). Participation ratio: MiniLM 0.92, E5 0.78, MPNet 0.66, MRL 0.21, non-MRL 0.19, Contriever 0.04. Files: `results/x1_*.csv`, `results/x2_*.csv`, `results/x3_*.csv`.
