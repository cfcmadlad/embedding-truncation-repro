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

- For each of 10 experiments: shuffle the train indices (one persistent permutation reshuffled every experiment, `numpy.random.default_rng(42)`; MTEB 1.12 seeds numpy's legacy global generator with 42 instead, so the stream and therefore the exact training examples differ from MTEB's, see Stage 1 audit below), keep the first `samples_per_label` items of each label, fit `sklearn.linear_model.LogisticRegression(max_iter=100, random_state=42)` with default C=1 and lbfgs, predict on the full test split, accuracy. Report the mean over the 10 experiments.
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

- Our nDCG@10 equals pytrec_eval `ndcg_cut_10` to 4e-8 per query on NanoSciFact for every model checked, and again to 1e-7 per query (0 queries differing by more than 1e-6) on five more model and dataset pairs in the Stage 1 audit. On NanoArguAna one or two of 50 queries differ because duplicate documents get exactly equal scores near rank 10; pytrec_eval breaks ties by document id, torch `topk` does not. Means differ by at most 0.003. Results in `results/e0_pytrec_check.json`.
- Contriever similarity: CLAUDE.md specifies dot product. We checked cosine as an alternative because the paper's Contriever Spearman (0.85) is higher than ours (0.79). Cosine gives Spearman 0.847 but full nDCG@10 0.451 (paper 0.524), random 50% 94.8% (paper 87.3) and PCA 100.5% (paper 77.3); dot gives 0.528, 88.7% and 85.8%. Dot product is therefore what the paper used, and we kept it.
- Removing and zeroing dimensions give identical cosine and dot scores (max difference 1e-13), `results/e5_remove_vs_zero.json`.

## Deviations

- NanoBEIR qrels are binary, so the graded nDCG implementation runs with gains of 1. The code supports graded gains.
- The `mteb/mtop_intent` repository is a loading script dataset that `datasets` 5 no longer runs; we read its `en/train` and `en/test` parquet files directly. That test parquet has 2,185 rows, while MTEB metadata lists 4,386, so absolute MTOPIntent accuracy is not directly comparable with the MTEB leaderboard. Relative performance is unaffected because full and truncated use the same split.
- `mteb/banking77` currently has 3,076 test rows (MTEB lists 3,080).
- The plan said to encode on the local GPU. The user asked to use the course HPC instead. Its only partition open to our account (`gpu_rtx_pro_6000_6_csis_hyd`) had all 5 GPUs held by multi day jobs, with an estimated start the next day; CPU jobs are capped at 4 cores by the course QOS, and the compute nodes have Python 3.6 only. We therefore encoded locally on the RTX 4060 Laptop GPU in fp16, throttled to a 50% duty cycle at below normal priority because the GPU was in use for other work. Throughput numbers in `results/e0_encoding_times.csv` include that throttle.
- Cached experiments run on CPU by default (`TRUNBED_DEVICE=cpu`). During development some scripts were also run on the GPU; CPU and GPU float arithmetic differ by up to 2e-4 nDCG@10, which is the same size as the per dimension attribution effects and flips the sign of some near zero deltas. All reported numbers therefore come from one clean `run_all.sh` run on CPU.
- Joblib parallelism uses the threading backend (`prefer="threads"`), because process workers each load torch and exhausted the Windows commit limit on the 16 GB machine (WinError 1455).
- E7: a dimension is a tie when |delta| < 1e-9; X1 and X2 set per query deltas below 1e-6 to zero to remove float noise from cosine renormalisation.
- E7 classification attribution covers 2 models (Para-MiniLM, MPNet) x 2 tasks (Banking77, Emotion), as CLAUDE.md specifies; the paper used 12 tasks and 6 models.
- E10 is a cross model comparison (n = 6) instead of the paper's within model comparison over training checkpoints; the paper's design needs contrastive retraining of BERT and T5.
- S1 runs Qwen2.5-0.5B (896 hidden dims) instead of Qwen2.5-7B and Llama 3.1 8B, on 200 HellaSwag validation items, 200 MMLU test items (0-shot, letter log likelihood) and 50 GSM8K test items (5-shot with the first 5 training items as shots, greedy, max 256 new tokens, strict match on `#### N`), sampled with seed 0. It does not use lm-evaluation-harness; scoring is our own. The final normed hidden state and the matching columns of the untied unembedding are sliced. It ran unthrottled on the local RTX 4060 Laptop GPU in fp16.
- The demo shares its logic with the CLI through `demo/core.py`, which is not in the CLAUDE.md layout. Full PCA fits per model are cached in `cache/pca/` and sliced for each K.

- CLAUDE.md asks for Python 3.10+. The pinned versions (numpy 2.5.2, scipy 1.16.3, matplotlib 3.11.0, transformers 5.22.2) need Python 3.12 or later, so the pinned environment requires Python 3.12+; it was developed and run on 3.13.7.
- `requirements-demo.txt` (live run and demo) uses version ranges that install on Python 3.10 to 3.13 on Windows, macOS and Linux; the pinned `requirements.txt` stays for regenerating the results and needs 3.12+.
- Text loaders (`load_retrieval`, `load_classification`, `load_sentences`) used to download from Hugging Face silently when a file was missing; they now raise `FileNotFoundError` unless called with `fetch=True`, which only `encode_all.py` and `fetch_lite.py` do.
- Phase 4 (S1) is opt in in `run_all.sh` (`RUN_S1=1`) because it needs a GPU and downloads a model and benchmark items; everything else runs offline from `cache/`.

## Runtimes

| Step | Hardware | Time |
|---|---|---|
| Encoding, 6 models x 166,078 texts | RTX 4060 Laptop, fp16, 50% duty | 49 min in total (2 to 14 min per model) |
| E1 retrieval, 6 models x 13 datasets x 110 settings | CPU | about 2 min |
| E1 classification, 6 models x 4 tasks x 100 settings x 10 LR fits | CPU, 8 threads | about 55 min |
| E7 retrieval attribution, 4 models | CPU | about 1.5 min |
| E7 classification attribution, 2 models x 2 tasks | CPU, threads | about 25 min |
| X1, X2, X3 | CPU | about 2, 1, 4 min |
| S1, 3 variants x 450 items | RTX 4060 Laptop, fp16 | about 30 min |
| Full `run_all.sh` from cache | CPU | about 1.5 h |

## Known weaknesses of our setup

- 4 of 12 MTEB classification tasks and NanoBEIR (50 queries per dataset) instead of full BEIR, so per dataset numbers are noisy; X2 shows the query level standard error of the NanoBEIR mean is 0.012.
- E5-base-v2 stands in for E5-large; its PCA result (101.7%) differs from the paper's E5-large (90.6%), so E5 comparisons are only indicative.
- MTEB few shot protocol (8 to 32 examples per label) makes classification accuracy noisy; our relative classification numbers are 0.6 to 2.3 points below the paper's.
- Cross model correlations in E10 and X3 use n = 6 models and are dominated by Contriever.
- The X3 theory ignores renormalisation after truncation and underpredicts E5's rank stability (0.79 vs 0.88).
- MTOPIntent test split is smaller than MTEB's (2,185 vs 4,386 rows).
- E5-Mistral (7B), Sentence-T5, E5-large, the remaining 8 MTEB tasks, the 7B to 8B LLMs and the BERT/T5 retraining were planned as an HPC extension. On 8 Nov 2026 at 14:30 the course QOS (`gpu_csis_course`) allowed no full GPU per job (`gres/gpu=0`), only a half GPU MPS share, 4 CPUs and 80 GB RAM; all 5 GPUs of `gpu_rtx_pro_6000_6_csis_hyd` were allocated to other users' multi day jobs and 22 jobs were pending. The extension was dropped and is left as future work.
- Added `src/evaluate.py` (shared retrieval and classification evaluators) and `src/paper.py` (paper reference numbers) to the layout in CLAUDE.md, so every script and the demo compare against one source of truth.

## Live run (`scripts/live_small.py`)

The assignment asks for a small version of the code run live on a subset of the data. The live run encodes the data with the encoder on the presenter's own CPU, runs the paper's core truncation experiment, and compares it with the paper's numbers. Everything it prints comes from the same code as the cached experiments: `RetrievalEval` and `ndcg_at_k` for scoring, `keep_last_k` and `keep_random` for truncation, `logreg_accuracy` and `mteb_subsamples` for classification, `FIG1_RANDOM_AVG` for the paper column. `src/live.py` adds only the subset function, the sweep loops, the removal-versus-zeroing check, the classification plan, the replay and the rendering. `tests/test_cache_parity.py` checks that the live sweep applied to cached embeddings reproduces the stored `e1_retrieval.csv` value, and `scripts/demo_cli.py` and the Streamlit app print the same numbers as the live run for the same setting (MiniLM, NanoSciFact, random 50%, seed 0: 0.4447; full 0.4713).

### What it scopes down

| Main reproduction | Live run | Why |
|---|---|---|
| 6 encoders | Para-MiniLM-L3 only (17M parameters, 69 MB of weights saved in `lite/models/minilm`) | The one paper model that encodes a corpus on a laptop CPU in seconds |
| 13 NanoBEIR datasets | NanoSciFact (50 queries, 2,919 documents) | Encoding the corpus is the whole cost; `--fast` cuts it to 400 documents, keeping all 55 judged ones |
| 4 MTEB tasks, 10 subsamples, full test split, 10 seeds | Banking77, the first 2 of the 10 MTEB subsamples (1,182 training texts), 1,000 random test texts (seed 0), random-K seeds 0 to 4, K in 0, 30, 50, 70, 90; `--fast`: 1 subsample, 300 test texts, 3 seeds, K in 0, 50, 90 | Fits within the 2 minute budget (the logistic regression sweep is about 5 to 12 s) |
| fp16 on the GPU for the cached embeddings | fp32 on the CPU | Live embeddings differ from the cached ones by at most 1.1e-3; the seed 0, K = 50 score agrees to 4 decimals (0.4447) |
| Paper numbers: Table 2 and Fig 2 per model | Figure 1 average over the paper's 6 encoders, read off the plot (`src/paper.py`), plus the Table 2 value for Para-MiniLM in the verdict line | Figure 1 is the only curve the paper gives per K for all models |

Retrieval keeps K from 0 to 90 in steps of 10 with the 10 seeds of the main experiment, so the live retrieval table is the cached MiniLM / NanoSciFact slice of E1 recomputed from scratch (stored 94.17% at random 50% against live 94.2%, stored 48.78% at 90% against live 48.7%). The classification result (stored 97.88% at 50% with 10 subsamples and the full test split, live 97.7%) differs by the smaller subsample and test set. The 50% row of each table is the one the verdict uses, with the paper's bars (retrieval above 90%, classification above 95%).

`--fast` keeps every judged document and fills the rest of the budget with the first unjudged documents by index. With fewer distractors the absolute nDCG@10 is higher (0.6543 against 0.4713); only relative numbers are comparable with the paper, and the script prints that it cut the corpus.

If anything in the live path raises an exception (model files missing, encoder error, memory), the script prints the exception and then the stored csv, labelled `STORED RESULT, NOT RECOMPUTED`. `--replay` prints the stored csv directly. The live path never uses the network or the GPU: it sets `HF_HUB_OFFLINE=1`, loads the encoder from `lite/models/minilm`, and forces `TRUNBED_ENCODE_DEVICE=cpu`.

### Rehearsal (laptop: 8 physical, 16 logical cores, Windows 11, fresh Python 3.12.4 environment in a path with spaces, GPU hidden with `CUDA_VISIBLE_DEVICES=-1`)

| Run | Wall time | Result |
|---|---|---|
| Full live run, 5 times | 48.2, 49.4, 55.5, 54.2, 48.6 s | identical every time: random 50% keeps 94.2% of nDCG@10 and 97.7% of accuracy, verdict holds. Runs 3 and 4 overlapped with my test runs |
| `--fast --threads 1`, 3 times | 35.2, 35.6, 50.1 s | random 50% keeps 93.4% of nDCG@10 and 99.3% of accuracy. The third run overlapped with test runs |
| Fallback (model folder hidden, no internet), 5 times | 21.7, 16.1, 14.7, 14.9, 15.3 s | error line, then the stored csv labelled as stored, 5 of 5 |
| `run.bat` from a clean unzip including `pip install` | 5 to 8 min the first time, 48 s after | |

Time breakdown of one full run: imports and encoder load 13 s (torch 6 s, sentence-transformers and transformers 6 s on this machine; the antivirus scans every file), encoding 2,969 texts 25 s (about 117 texts per second), the 110 retrieval scores 0.3 s, classification encoding 3 s and logistic regression sweep 5 to 12 s.

### Optimisation findings (numbers measured on this laptop, CPU only)

| Idea | Measurement | Decision |
|---|---|---|
| Thread count | Encoding 600 texts, median of 3: 4 threads 68 texts/s, 8 threads 107 to 122, 16 threads 113 to 126 | Default to the physical core count (8), the torch default; the 8 against 16 difference is within noise |
| Batch size | 32 against 128 at 8 threads: 107 against 122 texts/s | Keep the existing 128 |
| Sort by length | sentence-transformers already sorts by length inside `encode` | Nothing to add |
| `inference_mode` | `SentenceTransformer.encode` already runs under `torch.inference_mode` | Nothing to add |
| Cap `max_seq_length` | The cache used the model default of 128 for MiniLM; a lower cap would change the embeddings | Not applied |
| Vectorise the 10-seed sweep | The whole 110-score sweep takes 0.29 s | Not worth the complexity |
| Load the model once | `load_model` caches it in a dict; startup is now timed and printed separately | Done |
| Classification sweep | 50 logistic regression fits, 16.8 s serial, 12.2 s with 4 threads, identical numbers (max difference 0.0) | Applied (`workers`) |
| Demo first load | 13.8 s before, 9.3 s after (scikit-learn is imported only when a classifier is fitted) | Applied |
| Demo interaction | 1.05 s before, 0.9 s after (curve rendered once to cached PNG bytes, critique assets cached) | Applied |
| Before and after tables | Old and new retrieval tables for the same 2,969 texts differ by at most 5e-7 (the six printed decimals) | Numbers unchanged |

The honest conclusion is that the live run is bound by encoding and by the imports, which cannot be reduced without changing the numbers (a lower sequence cap, fp16 or int8 would change the embeddings). The measured gains are in the sweep, the app and the startup transparency, not in the encoder.

## Stage 1 audit

Checks run on 17 Nov 2026 on the Windows laptop with the GPU hidden:

| Check | Result |
|---|---|
| nDCG@10 against pytrec_eval, MiniLM NanoSciFact, MiniLM and MPNet NanoFiQA2018, E5 NanoNFCorpus, Contriever NanoSCIDOCS | 0 of 250 queries differ by more than 1e-6 (largest 1.0e-7); means identical to 6 decimals |
| Graded qrels | NanoBEIR qrels are binary (NFCorpus values 0 and 1); the code handles graded gains and a test compares it with pytrec_eval on random graded data |
| Random-K seeding | sizes, determinism, sorted unique indices, 10 distinct subsets for seeds 0 to 9 (tests) |
| Removal against zeroing | identical for cosine and dot: 9.9e-14 (float64) on all cached models, 4e-7 in float32 on the live embeddings (tests) |
| Classification protocol | Implements the MTEB structure (class balanced subsamples, `LogisticRegression(max_iter=100)`, mean of 10 experiments); the random stream differs from MTEB's legacy generator, see fixes |
| `src/paper.py` against `docs/paper.pdf` | Tables 1, 2 and 3, the Fig 4 counts, the Fig 8 approximate Spearman values, Table 4 and the anisotropy correlations match the PDF; Fig 1 values are read off the plot (approximate) |
| Every number in RESULTS.md against its csv | All tables recomputed from `results/`; 7 discrepancies found and fixed, below |
| Figures against csv | `fig_pca_vs_random`, `fig_theory_vs_observed` and the slide copies match their csv values; every figure is produced from the csv written by the same script |
| Windows path and encoding | All file I/O now passes `encoding="utf-8"`; the live and CLI scripts reconfigure stdout to UTF-8; tested with a non-UTF-8 console and a path with spaces |
| Deprecation warnings | The Streamlit 1.51 and 1.67 runs print none (the `use_container_width` warnings in `logs/` were from an earlier version of the app) |
| Silent internet or GPU use | Found and fixed: text loaders downloaded from Hugging Face when a file was missing (now explicit); the typed-query box fails with an explanation when the encoder is not on disk; no live path uses the GPU |

### Fixes (old value, new value, cause)

| # | Where | Old | New | Cause |
|---|---|---|---|---|
| 1 | RESULTS.md, Phase 0 table | Para-MiniLM 0.4871, Contriever 0.5283, E5-base 0.6168, MPNet NLI 0.4047 | 0.4873, 0.5284, 0.6170, 0.4052 | Values from an earlier run; the table disagreed with `e0_full_scores_summary.csv` and with Table C6 in the same file |
| 2 | RESULTS.md, README.md | "within 0.005 of the paper" | "within 0.006" | The largest gap is MPNet, 0.5807 against 0.5755 = 0.0052 |
| 3 | RESULTS.md X2 table | Para-MiniLM median absolute delta 6e-4 | 5e-4 (0.000500) | Typo against `x2_bootstrap.csv` |
| 4 | RESULTS.md and README.md headline | "3 to 6e-4, about 25 times smaller than the standard error" | "3 to 5e-4, 23 to 43 times smaller" | The ratio is 25, 43, 23 and 24 for the four models |
| 5 | REPRO_NOTES.md classification protocol | "copied from MTEB" with a generator seeded 42 | states that our `default_rng(42)` stream differs from MTEB's legacy `np.random.seed(42)` stream, so the 10 training subsets are not the exact MTEB ones | Checked: on Banking77 with MiniLM the mean accuracy is 0.783 (ours) against 0.779 (MTEB stream), standard deviations 0.005 and 0.009. Not regenerated, because the difference is inside the run to run noise and all stored classification numbers use one stream consistently |
| 6 | README.md | "the first typed query downloads the encoder from Hugging Face" | the demo runs offline, so the encoder must be on disk (`fetch_lite.py --models`) or `HF_HUB_OFFLINE=0` | The demo sets `HF_HUB_OFFLINE=1`; the sentence was wrong |
| 7 | `src/data.py` | missing text files were downloaded silently | `FileNotFoundError` with the fix, unless `fetch=True` | Silent internet use |
| 8 | `src/metrics.py` | `ndcg_at_k` crashed when a corpus had fewer than 10 documents | clamps k to the number of documents | Robustness for tiny subsets (no stored result is affected: all corpora have at least 2,000 documents) |
| 9 | `demo/core.py` | classification with "remove degrading / improving" used the NanoBEIR retrieval attribution | uses the classification attribution (minilm and mpnet on Banking77 and Emotion), otherwise reports that no result exists | The old behaviour applied retrieval attribution to a classification task |
| 10 | `src/metrics.py` | `torch.set_num_threads(6)` hard coded as the default | physical core count | Oversubscribed 4 core laptops |

No stored result in `results/` or `figures/` was wrong, so none was regenerated; items 1 to 4 and 6 were text errors in the documents. The slide deck `Claude outputs/Paper14_Replication_Group12.pdf` has no source in the repository (no pptx, html or tex); every numeric claim on its 11 pages was compared with `results/` and with the paper and none disagreed. It contains no mention of the live run.

## Live kit and any-laptop readiness

- `lite/models/minilm` (69 MB, git ignored) and `lite/data/{NanoSciFact,Banking77}.json` (5 MB) are everything the live run reads; `python scripts/fetch_lite.py` rebuilds them from Hugging Face (or copies the text from `cache/text`) and `--zip` writes `dist/embedding-truncation-lite.zip` (70 MB, with `src`, `scripts`, `demo`, `tests`, `docs`, `results`, `figures`, launchers).
- `requirements-demo.txt` holds version ranges instead of pins and no extra index: on Windows and macOS `pip install torch` already gives a CPU build; `run.sh` installs the CPU wheel first on Linux. Proven in fresh virtual environments on this laptop: Python 3.13.7 (installed torch 2.15.0+cpu, numpy 2.5.5, scikit-learn 1.9.1, sentence-transformers 6.3.0, streamlit 1.67.0; live run, tests and check all pass) and Python 3.12.4 (from the unzipped kit in a path with spaces, through `run.bat`: `check` all ok, live run 48 s). Python 3.10, 3.11, macOS and Linux were not run here; the ranges allow them, the code uses nothing newer than 3.10 syntax, and the self-check (`docs/SELF_CHECK.md`) is meant to prove them on the teammates' machines.
- The antivirus on this laptop intercepts HTTPS. pip 24.0 (bundled with Python 3.12.4) fails there with `CERTIFICATE_VERIFY_FAILED`; the launchers retry once with `--trusted-host pypi.org --trusted-host files.pythonhosted.org`, which worked. pip 25 and later use the Windows certificate store and need no retry.
- Windows PowerShell does not run `run.bat` by bare name when `NoDefaultCurrentDirectoryInExePath` is set; use `.\run.bat`.
- `.gitattributes` forces CRLF for `run.bat` (a batch file with LF line endings can break `goto`).
- Not verified: macOS (including Apple silicon), Linux, Python 3.10 and 3.11, a laptop with fewer than 8 cores, a real second laptop. Only a Windows laptop with an 8 core CPU was available.
