# Viva questions and answers

Every answer points to a file or a number in this repository. Numbers are ours unless marked "paper". "Cached" means the full embedding cache behind RESULTS.md; "live" means the run of `scripts/live_small.py`.

## The paper and the reproduction

**1. What does the paper claim?**
Removing a random 50% of the dimensions of a text embedding keeps more than 90% of retrieval quality (nDCG@10) and more than 95% of classification accuracy, across 6 encoders and 26 tasks. It rejects anisotropy, dimensional collapse and outlier dimensions as explanations and proposes that many "degrading" dimensions exist, so a random cut removes good and bad dimensions in equal measure.

**2. What did you implement yourself, and what did you only use?**
We wrote truncation (`src/truncate.py`), nDCG@10, Spearman and the MTEB classification protocol (`src/metrics.py`), leave-one-dimension-out attribution (`src/attribution.py`), the sampling theory (`src/theory.py`) and every evaluation loop. We only use pretrained encoders and Hugging Face datasets as inputs. The authors' code was not used.

**3. Why MiniLM-L3 in the live run?**
It is the paper's Paraphrase-MiniLM (17M parameters, 384 dimensions), the weights are 69 MB, and on a CPU it encodes about 90 to 120 texts per second. The paper reports it as a model with a clear but moderate drop at 50% (94.6% on BEIR, 97.4% on MTEB), so it is a fair test and not a cherry-picked robust model.

**4. Why NanoBEIR and why NanoSciFact?**
The paper itself uses NanoBEIR for E5-Mistral and for every attribution analysis. NanoSciFact has 50 queries and 2,919 documents, small enough to encode live in about 25 s, and every query has judged documents. The cost is noise: with 50 queries per dataset the standard error of the benchmark mean is about 0.012 (X2).

**5. Why did you drop E5-large, E5-Mistral and Sentence-T5?**
E5-Mistral needs about 14 GB just for fp16 weights and the laptop GPU has 8 GB. The course HPC was unusable: on 8 Nov 2026 the course QOS gave no full GPU, all 5 GPUs of the open partition were held by multi-day jobs and 22 jobs were queued (REPRO_NOTES.md, known weaknesses). E5-large was replaced by E5-base-v2 (same family and prefixes, a third of the cost). Sentence-T5 adds an architecture but no new claim.

**6. Why 4 of 12 MTEB classification tasks?**
Each task needs 100 truncation settings times 10 logistic regression fits per model. Banking77, Emotion, TweetSentimentExtraction and MTOPIntent span 3 to 113 classes and three domains. The relative numbers are 0.6 to 2.3 points below the paper's, which fits the few-shot MTEB protocol (8 to 32 examples per label).

## Why removal equals zeroing, and the method

**7. Why does removing a dimension equal zeroing it for cosine and dot product?**
The dot product is a sum of per-dimension products, and a zeroed coordinate contributes 0 to that sum, exactly as a missing one does. The norms are sums of squares, so a zero coordinate adds nothing there either. Hence both the dot product and the cosine are identical. We checked this on synthetic data and all six cached models: the largest difference is 9.9e-14 (`results/e5_remove_vs_zero.json`), and the live run prints the same check on its own embeddings (largest score difference about 4e-7 in float32). The paper's stated reason for preferring removal ("zeroing may have varying impact") does not apply to cosine or dot-product similarity.

**8. Do cosine models need renormalising after truncation?**
The cosine divides by the norm of the truncated vector, so it renormalises by itself. Contriever uses the raw dot product, which shrinks scores but keeps the document order for a fixed query.

**9. How exactly does random-K work?**
`keep_random(dim, frac, seed)` draws `dim - round(frac * dim)` dimensions to keep without replacement with `numpy.random.default_rng(seed)`. Seeds 0 to 9 give 10 different subsets; we report the mean and the standard deviation over the 10 seeds, as the paper does. Each K and seed is a fresh draw, not nested subsets.

**10. How do you know your nDCG@10 is right?**
It matches `pytrec_eval` to below 1e-7 per query on NanoSciFact (MiniLM, 50 queries), NanoFiQA2018 (MiniLM and MPNet), NanoNFCorpus (E5) and NanoSCIDOCS (Contriever), 0 queries differing by more than 1e-6. On NanoArguAna one query differs by 0.044 because duplicate documents get exactly tied scores near rank 10 and the two tools break ties differently; the mean differs by 0.0009. The test suite repeats the pytrec_eval comparison on random graded data.

**11. How is classification scored?**
Logistic regression (`max_iter=100`, default C, lbfgs) on embeddings, accuracy on the test split, mean over 10 class-balanced training subsamples (MTEB's protocol). One deviation: we draw the subsamples with `numpy.random.default_rng(42)`, while MTEB seeds numpy's legacy generator with 42, so the exact training examples differ. On Banking77 with MiniLM this changes the mean accuracy from 0.779 to 0.783 (std 0.005 to 0.009), inside the run-to-run noise. All stored classification numbers use our draw consistently.

**12. Why do you use the dot product for Contriever?**
Because it reproduces the paper: dot gives full nDCG@10 0.528 (paper 0.524), random 50% 88.7% (paper 87.3) and PCA 85.8% (paper 77.3). Cosine gives 0.451, 94.8% and 100.5% (RESULTS.md, C3). It is also the only non-normalised model, and the most fragile; the paper does not connect these two facts.

## Where we differ from the paper

**13. Where do your numbers differ from the paper, and why?**
- Contriever Spearman at 50%: 0.788 against 0.85. The paper may compute the correlation over a top-k list instead of the whole corpus; with cosine we get 0.847 but then three other numbers stop matching.
- E5 PCA: 101.7% for E5-base-v2 against 90.6% for E5-large. Different model size, so only indicative.
- Classification: 0.6 to 2.3 points below the paper, from the few-shot protocol and 4 of 12 tasks.
- Degrading/improving split: every model has hundreds of degrading dimensions as in the paper, but the split differs (MPNet 382 / 382 against 595 / 169). X1 and X2 show the split is close to noise.
- Full-size E5: our E5-base-v2 scores 0.617 and the paper's E5-large 0.5765, which suggests the paper did not use E5's `query:` and `passage:` prefixes.
- Live run: 94.2% of nDCG@10 on NanoSciFact against the paper's 94.6% for Para-MiniLM averaged over BEIR; one dataset against an average, so agreement is approximate.

**14. How good is the agreement overall?**
Random 50% removal keeps 88.7 to 98.3% of NanoBEIR nDCG@10 (paper 87.3 to 97.7) and 93.1 to 97.0% of accuracy (paper 95.4 to 97.7). Contriever is the fragile outlier in both. Full scores are within 0.006 of the paper for the three shared models.

**15. PCA against random truncation?**
PCA to 50% beats random 50% for MPNet, MiniLM and E5 and loses for Contriever on retrieval (85.8% against 88.7%), the same exception as the paper's. Random truncation needs no fitting and no extra data.

**16. What does the MRL experiment show?**
A Matryoshka model (`mpnet-base-nli-matryoshka`) beats its non-MRL twin only for last-K (77.4% against 68.6% at 90% removed on retrieval, 87.8% against 81.7% on classification). Under random removal the two are within 1.1 points at 50%. This confirms the paper's informal remark and adds that the MRL advantage is specific to prefix truncation.

## Our extensions

**17. What does the leakage test (X1) show?**
The paper finds degrading dimensions and evaluates their removal on the same queries. We split queries 50/50 (5 splits), choose dimensions on half A and test on half B. In sample the gain peaks at 101 to 103%, held out it is 97.9 to 100.2% at the same cut, and the per-dimension effects on the two halves are uncorrelated (r between -0.05 and 0.01). So the gains above 100% in the paper's Fig 5 and 12 are consistent with selection on test noise.

**18. What does the bootstrap (X2) show?**
Resampling queries 1,000 times, only 8 to 22 degrading dimensions per model have a 95% interval excluding zero, about what chance gives (9.6 to 19.2). The median effect of one dimension (3 to 5e-4 nDCG@10) is 23 to 43 times smaller than the standard error of the benchmark score (0.012).

**19. Does that prove the paper is wrong?**
No. It shows that the paper's evidence, counting dimensions whose removal helps on the same queries, cannot separate a real effect from noise. A model could still have truly harmful dimensions; our test finds no evidence of them at NanoBEIR's 50-query scale. The robustness result itself (random 50% keeps most performance) reproduces.

**20. What is the sampling theory (X3)?**
A score q.d is a sum of n per-dimension products. A random subset of m dimensions gives an unbiased estimate of that sum (scaled by n/m) with variance given by the spread of the products times a finite-population correction (n - m)/(n - 1). Two documents swap order when the noise exceeds their score gap, which we turn into a predicted rank correlation (`src/theory.py`). It predicts the observed Spearman at random 50% within 0.03 for 5 of 6 models. It fails for E5 (0.789 predicted against 0.881 observed) because it ignores renormalisation after truncation.

**21. Did the participation ratio beat the anisotropy metrics?**
No, and we report it as such. Contriever has by far the most concentrated contributions (0.04 against 0.19 to 0.92), but with 6 models every correlation is driven by Contriever: participation ratio 0.50, IsoScore -0.76, inter-dimension correlation 0.64 with the 50% drop. Our own prediction in the project plan is not supported.

**22. What does the anisotropy analysis (E10) do, and what is its weakness?**
The paper retrains BERT and T5 and tracks checkpoints. We compute IsoScore, uniformity and inter-dimension correlation on 10k simple-wiki sentences for our six trained models and correlate them with the 50% drop. Different design, n = 6, and Contriever is a high-leverage point. We read it as "not explained by anisotropy", not as a counter claim.

**23. What does the LLM experiment (S1) show?**
Qwen2.5-0.5B with half the final hidden state (and matching unembedding columns) removed keeps 75 to 86% on 200 MMLU and 200 HellaSwag items but GSM8K falls from 0.32 to 0 (50 items), the same pattern as the paper's 7B models (GSM8K 0.766 to 0.045 and 0.011). The 0.5B model keeps less than the 7B on MMLU. Sample sizes are small.

## The live run

**24. What is the live run, and what does it scale down?**
`scripts/live_small.py` encodes NanoSciFact (50 queries, 2,919 documents) with MiniLM-L3 on the laptop CPU, runs last-K and random-K (10 seeds) for K from 0 to 90%, checks removal against zeroing, prints the paper's Figure 1 average next to every row, and then does the same for Banking77 (2 MTEB subsamples, 1,000 test texts, 5 seeds). It runs in about a minute. `--fast` cuts the corpus to 400 documents (all 55 judged ones kept), so absolute nDCG is higher than on the full corpus; only the relative columns are comparable with the paper. Live embeddings differ from the cached ones by at most 1.1e-3 because the cache was encoded in fp16 on the GPU and the live run uses fp32 on the CPU; the seed-0 result at 50% is the same to 4 decimals (0.4447).

**25. What was not reproduced, and what should we say if asked?**
E5-large, E5-Mistral and Sentence-T5 (compute), 14 full BEIR datasets (NanoBEIR instead), 8 of 12 MTEB tasks, retraining BERT and T5 for the anisotropy checkpoints, and the 7B and 8B LLMs of Table 3. The paper's E5-Mistral outlier analysis (Fig 13) and the SQuAD, COPA and DROP rows of Tables 3 and 4 were also left out. Everything is listed in REPRO_NOTES.md, with the reason for each.

## Extra questions the examiner may ask

**26. What looks inconsistent in the paper itself?**
Table 4 repeats Table 3's MMLU numbers in its HellaSwag column (0.718, 0.709, 0.709 for Qwen) and COPA F1 equals DROP F1 in every row, likely copy errors. SQuAD v2 scores 50.07 for every truncated model, about the share of unanswerable questions, so the truncated model probably always answers "no answer".

**27. Why are the live numbers not identical on every laptop?**
Float32 sums depend on the number of threads and on the BLAS library, so the fourth decimal of an nDCG value can change. Seeds, subsamples and the 10 random subsets are fixed, so the picture and the verdict do not change.

**28. What would you do with more compute?**
Run the missing models and tasks, repeat X1 and X2 on full BEIR (more queries, smaller standard errors), and train a model with random dimension dropout so that any subset of dimensions works, not only the first ones as with MRL.
