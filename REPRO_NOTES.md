# Reproduction notes

Paper: Takeshita et al., "Randomly Removing 50% of Dimensions in Text Embeddings has Minimal Impact on Retrieval and Classification Tasks", EMNLP 2025.

## Plan

### Hardware

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM (8188 MiB), driver 592.82, CUDA 13.1 runtime, torch 2.9.0+cu126.
- CPU: 16 logical cores. OS: Windows 11. Python 3.13.7.
- Encoding runs on the GPU in fp16. Everything after encoding runs on cached float32 arrays and works on CPU.

### Models

| Key | HF id | Dim | Pooling | Similarity | Prefixes | Stands in for |
|---|---|---|---|---|---|---|
| minilm | sentence-transformers/paraphrase-MiniLM-L3-v2 | 384 | mean | cosine | none | Paraphrase-MiniLM (17M) |
| mpnet | sentence-transformers/all-mpnet-base-v2 | 768 | mean + normalise | cosine | none | MPNet base |
| contriever | facebook/contriever | 768 | mean | dot, no normalisation | none | Contriever |
| e5 | intfloat/e5-base-v2 | 768 | mean + normalise | cosine | `query: ` / `passage: ` | E5-large (scaled down) |
| mrl | tomaarsen/mpnet-base-nli-matryoshka | 768 | mean | cosine | none | MRL baseline |
| nonmrl | tomaarsen/mpnet-base-nli | 768 | mean | cosine | none | non-MRL twin |

### Data

- Retrieval: all 13 NanoBEIR subsets from `zeta-alpha-ai/Nano{ClimateFEVER, DBPedia, FEVER, FiQA2018, HotpotQA, MSMARCO, NFCorpus, NQ, QuoraRetrieval, SCIDOCS, ArguAna, SciFact, Touche2020}`, configs `corpus`, `queries`, `qrels`, split `train`. Metric nDCG@10.
- Classification: `mteb/banking77`, `mteb/emotion`, `mteb/tweet_sentiment_extraction`, `mteb/mtop_intent` (parquet files under `en/`). Metric accuracy on `test`.
- PCA fit set: 20,000 unique sentences from `sentence-transformers/all-nli` (pair config, train split), sampled with seed 0.
- Anisotropy set: 10,000 texts from `sentence-transformers/simple-wiki` (train split), sampled with seed 0.

### Classification protocol (copied from MTEB 1.12 AbsTaskClassification, reimplemented)

- For each of 10 experiments: shuffle the train indices (one persistent permutation reshuffled every experiment, generator seeded with 42), keep the first `samples_per_label` items of each label, fit `sklearn.linear_model.LogisticRegression(max_iter=100, random_state=42)` with default C=1 and lbfgs, predict on the full test split, accuracy. Report the mean over the 10 experiments.
- `samples_per_label`: Banking77 8, MTOPIntent 8, Emotion 16, TweetSentimentExtraction 32 (MTEB task metadata).
- Truncation is applied to train and test embeddings identically before fitting. The same 10 training subsets are reused for every truncation setting, so differences come only from the dimensions.

### Seeds

- Random truncation: seeds 0 to 9 (10 runs as in the paper).
- Outlier control sets: seeds 0 to 9.
- Leakage splits (X1): seeds 0 to 4.
- Bootstrap (X2): seed 0, 1000 resamples.
- Classification subsampling: 42 (MTEB default).

### Expected runtime

| Phase | Work | Expected |
|---|---|---|
| 0 | download, encode 6 models over about 56k NanoBEIR docs, 650 queries, 4 classification tasks, 20k NLI and 10k wiki sentences | 30 to 45 min on GPU |
| 1 | E1 to E5 on cached arrays, about 26k logistic regression fits run in parallel | 20 to 40 min |
| 2 | E6 to E10; retrieval attribution vectorised on GPU, classification attribution (2 models x 2 tasks x 10 fits per dimension) in background | 30 to 60 min |
| 3 | X1 to X3 on cached arrays | 15 to 30 min |
| 4 | Qwen2.5-0.5B, 200 HellaSwag + 200 MMLU + 50 GSM8K, three head variants | 20 to 40 min |
