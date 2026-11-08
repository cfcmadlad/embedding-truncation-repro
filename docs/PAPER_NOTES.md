# Paper 14 reference numbers and protocol

Takeshita, Takeshita, Ruffinelli, Ponzetto. EMNLP 2025 (arXiv 2508.17744v2). Use these as the comparison targets. Figure values are read off plots, so treat them as approximate.

## Setup in the paper

- 6 encoders: MPNet base (109M, 768d), Contriever (110M, 768d), E5-large (335M, 1024d), E5-Mistral (7B, 4096d), Paraphrase-MiniLM (17M, 384d), Sentence-T5 base (110M, 768d). All contrastively trained.
- Retrieval: 14 BEIR datasets (NanoBEIR for E5-Mistral), metric nDCG@10. Classification: 12 MTEB tasks (AmazonCounterfactual, AmazonPolarity, AmazonReviews, Banking77, Emotion, Imdb, MassiveIntent, MassiveScenario, MTOPDomain, MTOPIntent, ToxicConversations, TweetSentimentExtraction).
- Truncation: last K% and random K% (10 runs, std reported).
- Attribution: remove one dimension at a time (not zeroing), on 13 NanoBEIR and 12 MTEB datasets.
- Outliers: dims of the average NanoBEIR query embedding deviating more than 3 sigma (Kovaleva et al. 2021).
- PCA: scikit-learn, fit on 20k all-nli sentences, reduce to 50%.
- Anisotropy / collapse: contrastively train BERT and T5 on SNLI+MultiNLI with InfoNCE, save checkpoints, measure uniform loss, IsoScore, mean inter dimension correlation on 10k simple-wiki paragraphs, correlate with relative performance at 50% last-K.

## Claims and numbers

C1. Random 50% removal keeps more than 90% retrieval and more than 95% classification performance (Fig 1, 2c, 2d). Std across 10 random runs is small.

C2. Last-K: relative performance falls below 80% only at 80% removal for 5 of 6 models. E5-Mistral keeps 90% with 90% removed (Fig 2a, 2b). Contriever degrades fastest on BEIR (Fig 9: drops to roughly 5 to 35% at 90% on most datasets).

C3. Spearman between full and truncated rankings stays above 0.8 at 50% last-K for all models (Fig 8). Approximate values at 50%: E5-Mistral 0.95, Sentence-T5 0.92, MPNet 0.90, E5-large 0.88, Paraphrase-MiniLM 0.88, Contriever 0.85. At 90%: Contriever about 0.43, others 0.55 to 0.63, E5-Mistral 0.74.

C4. Anisotropy does not explain it. Pearson between relative performance at 50% and metrics over training checkpoints: T5 uniform loss 0.36, IsoScore -0.47; BERT uniform loss -0.35, IsoScore 0.39. Relative performance stays roughly flat (T5 98 to 102%, BERT 94.5 to 98%) while full performance rises.

C5. Dimensional collapse does not explain it. Pearson with Corr Mean: T5 0.45, BERT -0.30.

C6. Outlier dims do not explain it (Table 1, NanoBEIR nDCG@10):

| Model | # outliers | Without outliers | Without random non-outliers |
|---|---|---|---|
| MPNet | 5 | 0.576 | 0.576 ± 9e-4 |
| Contriever | 3 | 0.525 | 0.524 ± 1e-3 |
| E5-large | 1 | 0.577 | 0.577 ± 4e-4 |
| E5-Mistral | 32 | 0.651 | 0.626 ± 6e-4 |
| Para-MiniLM | 2 | 0.483 | 0.483 ± 6e-4 |
| ST5 | 2 | 0.489 | 0.489 ± 9e-4 |

C7. Many degrading dims, uniformly spread (Fig 4). Counts (degrading / improving / tie):

| Model | dims | NanoBEIR | MTEB |
|---|---|---|---|
| MPNet | 768 | 595 / 169 / 4 | 246 / 522 / 0 |
| Contriever | 768 | 498 / 270 / 0 | 398 / 370 / 0 |
| E5-large | 1024 | 308 / 716 / 0 | 430 / 594 / 0 |
| E5-Mistral | 4096 | 2652 / 1419 / 25 | 1932 / 2164 / 0 |
| Para-MiniLM | 384 | 117 / 267 / 0 | 182 / 202 / 0 |
| ST5 | 768 | 204 / 564 / 0 | 277 / 491 / 0 |

Full NanoBEIR nDCG@10 (red line in Fig 4a, approx): MPNet 0.5755, Contriever 0.524, E5-large 0.5765, E5-Mistral 0.6264, Para-MiniLM 0.4835, ST5 0.4895. Per dimension effects are of order 1e-3 nDCG or smaller, except E5-Mistral outliers (up to about +0.006).

C8. Removing only degrading dims: performance rises above 100% for Sentence-T5 (up to about 103%), E5-large, E5-Mistral (about 102%), Paraphrase-MiniLM; decays slower than last-K for Contriever and MPNet. Removing only improving dims drops faster than last-K (Fig 5, Fig 12).

C9. E5-Mistral has 19 outlier degrading dims (beyond 3 sigma of attribution), 12 of which are also value outliers (Fig 13).

C10. No degrading dim is shared across all 13 NanoBEIR datasets; for MPNet more than 20% of dims degrade on 5 datasets (Fig 6).

C11. Random 50% truncation vs PCA 50%, relative performance % (Table 2):

| Model | BEIR Trunc | BEIR PCA | MTEB Trunc | MTEB PCA |
|---|---|---|---|---|
| MPNet | 97.7 | 99.4 | 96.6 | 99.6 |
| Contriever | 87.3 | 77.3 | 95.4 | 98.5 |
| E5-large | 95.9 | 90.6 | 97.7 | 99.8 |
| E5-Mistral (NanoBEIR) | 99.6 | 100.6 | 98.2 | 100.4 |
| Para-MiniLM | 94.6 | 99.3 | 97.4 | 99.5 |
| ST5 | 95.9 | 100.2 | 97.8 | 99.9 |

C12. LLMs (Llama 3.1 8B, Qwen 2.5 7B), half of last hidden state plus unembedding removed (Table 3, 4):

| Model | Task | Full | First half | Last half |
|---|---|---|---|---|
| Llama | MMLU acc | 0.681 | 0.580 | 0.586 |
| Llama | SQuAD v2 best exact | 51.87 | 50.07 | 50.07 |
| Llama | GSM8K strict EM | 0.764 | 0.009 | 0.014 |
| Qwen | MMLU acc | 0.718 | 0.709 | 0.709 |
| Qwen | SQuAD v2 best exact | 50.12 | 50.07 | 50.07 |
| Qwen | GSM8K strict EM | 0.766 | 0.045 | 0.011 |
| Llama | HellaSwag acc | 0.681 | 0.580 | 0.586 |
| Qwen | HellaSwag acc | 0.718 | 0.709 | 0.709 |
| Llama | COPA F1 / DROP F1 | 0.194 / 0.194 | 0.094 / 0.094 | 0.038 / 0.038 |
| Qwen | COPA F1 / DROP F1 | 0.003 / 0.003 | 0.001 / 0.001 | 0.001 / 0.001 |

C13. Limitations the authors state: cause of degrading dims unknown; single dimension attribution only; English only; MRL relation untested (informal check found MRL and non-MRL MPNet comparable).

## Things we noticed that the paper does not address (for critique and viva)

1. Removal vs zeroing: the paper says it removes rather than zeroes because zeroing has varying impact. For cosine and dot similarity the two are mathematically identical, so this justification is moot.
2. Degrading dims are identified and then evaluated on the same queries (Fig 5, 12). Gains above 100% may be selection on test noise. Per dimension deltas (about 1e-3) are tiny next to the query sampling error of nDCG@10 on 50 query NanoBEIR sets. No significance tests anywhere.
3. No mechanism for why random removal works. A random subset of dims is an unbiased scaled estimate of the dot product, so rankings survive whenever per dimension contributions are spread out. This predicts which models are fragile.
4. Table 4 looks inconsistent: the HellaSwag column equals the MMLU row of Table 3 exactly, and COPA F1 equals DROP F1 for every row. Likely copy errors.
5. SQuAD v2 truncated scores are 50.07 for both models and both halves, which is about the share of unanswerable questions: the truncated model likely collapses to always answering "no answer". So "retains 96 to 99%" on SQuAD v2 is not retained ability.
6. Qwen full COPA and DROP F1 of 0.003 means the baseline is near zero, so relative numbers there are meaningless.
7. "Uniformly distributed" degrading dims is claimed visually, not tested.
8. Anisotropy test varies checkpoints of only two small models; across model correlations are not reported.
9. Contriever is the only non normalised dot product model and the clear outlier in C2, C3, C11; the paper does not connect these.
