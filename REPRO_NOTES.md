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

## Scope reductions

| Paper | Ours | Why |
|---|---|---|
| 6 encoders | 4 core encoders plus an MRL / non-MRL MPNet pair | Fits a laptop GPU and keeps every model under 1 GB |
| E5-large (335M, 1024d) | E5-base-v2 (110M, 768d) | Same family, prefixes and training recipe at a third of the cost; documented as a stand in, so its numbers are compared with E5-large only loosely |
| E5-Mistral (7B, 4096d) | dropped | 7B encoder needs about 14 GB in fp16 for weights alone, beyond the 8 GB GPU; the HPC GPUs were fully booked on the run day |
| Sentence-T5 base | dropped | Adds a third architecture family but no new claim; the four core models already span the full range of fragility (Contriever most fragile) |
| 14 full BEIR datasets | 13 NanoBEIR datasets (50 queries, 2k to 6k docs each) | Full BEIR has corpora up to 5M docs; NanoBEIR is what the paper itself uses for E5-Mistral and for all attribution analyses |
| 12 MTEB classification tasks | 4 tasks: Banking77, Emotion, TweetSentimentExtraction, MTOPIntent (en) | Cover 3 to 113 classes and three domains; logistic regression cost grows with the number of tasks times 100 truncation settings |
| Contrastive training of BERT and T5 checkpoints for anisotropy (Sec 3.1, 3.2) | Across model comparison of IsoScore, uniformity and correlation on our 6 models | Retraining is out of scope; the cross model version tests the same hypothesis with a different (and weaker, n=6) design |
| Llama 3.1 8B and Qwen 2.5 7B | Qwen2.5-0.5B, 200 HellaSwag, 200 MMLU, 50 GSM8K items | 7B to 8B models do not fit the GPU; same truncation operation on a smaller sibling |

## Protocol choices

- Embeddings are stored as the raw model output (`normalize_embeddings=False`). Models whose sentence-transformers pipeline ends in a Normalize module (all-mpnet-base-v2, e5-base-v2) are therefore stored already normalised. Truncation always happens on the stored vector; cosine models are renormalised after truncation by the cosine itself, Contriever uses the raw dot product.
- max_seq_length is the model default capped at 512 (MiniLM 128, MPNet 384, MPNet NLI pair 384 and 75, Contriever 512, E5 512).
- Encoding in fp16 on the GPU, stored as float32.
- nDCG@10 with linear gain (as pytrec_eval and BEIR), ideal DCG from all judged documents. A document whose id equals the query id is excluded from the ranking, as BEIR and MTEB do (`ignore_identical_ids`); this matters for NanoArguAna (20 queries) and one NanoClimateFEVER query.
- NanoBEIR score of a model = mean over the 13 datasets of the mean per query nDCG@10. Relative performance = mean over datasets (or tasks) of truncated / full, in percent. Std for random truncation = std over the 10 seeds of that dataset averaged relative score.
- Last-K removes the last round(K * D) dims. Random-K draws round(K * D) dims without replacement with `numpy.random.default_rng(seed)`.
- PCA: our own numpy SVD on the mean centred 20k all-nli embeddings of each model, keep D/2 components, project queries, documents and classification inputs with the same mean and components. Similarity after projection is the model's own (cosine or dot).
- Spearman (E2) is computed per query over the scores of the whole corpus (all 2k to 6k docs), full vs last-K truncated, then averaged over queries and datasets.

## Verification

- Our nDCG@10 equals pytrec_eval `ndcg_cut_10` to 4e-8 per query on NanoSciFact. On NanoArguAna one of 50 queries differs because two duplicate documents get exactly equal scores at ranks 9 and 10; pytrec_eval breaks ties by document id, torch `topk` does not. Mean differs by 0.003. Results in `results/e0_pytrec_check.json`.

## Deviations

- NanoBEIR qrels are binary, so the graded nDCG implementation runs with gains of 1. The code supports graded gains.
- The `mteb/mtop_intent` repository is a loading script dataset that `datasets` 5 no longer runs; we read its `en/train` and `en/test` parquet files directly. That test parquet has 2,185 rows, while MTEB metadata lists 4,386, so absolute MTOPIntent accuracy is not directly comparable with the MTEB leaderboard. Relative performance is unaffected because full and truncated use the same split.
- `mteb/banking77` currently has 3,076 test rows (MTEB lists 3,080).
- The plan said to encode on the local GPU. The user asked to use the course HPC instead. Its only partition open to our account (`gpu_rtx_pro_6000_6_csis_hyd`) had all 5 GPUs held by multi day jobs, with an estimated start the next day; CPU jobs are capped at 4 cores by the course QOS, and the compute nodes have Python 3.6 only. We therefore encoded locally on the RTX 4060 Laptop GPU in fp16, throttled to a 50% duty cycle at below normal priority because the GPU was in use for other work. Throughput numbers in `results/e0_encoding_times.csv` include that throttle.
- All cached experiments ran on CPU (16 cores, 6 parallel workers at below normal priority) so the local GPU stayed free.
- Added `src/evaluate.py` (shared retrieval and classification evaluators) and `src/paper.py` (paper reference numbers) to the layout in CLAUDE.md, so every script and the demo compare against one source of truth.
