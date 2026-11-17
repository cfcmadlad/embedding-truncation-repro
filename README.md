# Random Truncation of Text Embeddings: a Reproduction

A scoped down, faithful reproduction of

> Sotaro Takeshita, Yurina Takeshita, Daniel Ruffinelli and Simone Paolo Ponzetto. *Randomly Removing 50% of Dimensions in Text Embeddings has Minimal Impact on Retrieval and Classification Tasks.* EMNLP 2025.

Prepared for CS F429 (Introduction to NLP), BITS Pilani Hyderabad, Group 12, paper 14. The repository contains an independent implementation of every truncation method, metric, attribution analysis and evaluation in the paper, precomputed embeddings for running all of it offline, three extension experiments that test the paper's explanation, and an interactive demo.

## 1. Quick start

There are two ways to run this repository. The lite kit runs the paper's core experiment live on your own CPU in about a minute and needs no large download. The full cache (1.7 GB) is only needed for the Streamlit demo and for regenerating every result.

### 1a. Lite kit: the live run

**What it is.** `scripts/live_small.py` encodes NanoSciFact (50 queries, 2,919 documents) with `paraphrase-MiniLM-L3-v2` on the CPU, removes the last K% and a random K% (10 seeds) of the dimensions for K = 0 to 90, and prints nDCG@10 next to the paper's Figure 1 average. It also checks that removing and zeroing dimensions give the same score, and repeats the experiment for classification on Banking77. It ends with a one-line verdict. No GPU is used.

**Requirements.** Python 3.10 to 3.13 (check with `python --version`), 3 GB of free disk space for the packages, and internet once to install them. After that the live run works offline.

**Option A: zip file.** Copy `embedding-truncation-lite.zip` from the USB stick, unzip it, open a terminal in the unzipped folder and run:

Windows (Command Prompt or PowerShell):

```bash
.\run.bat
```

macOS or Linux:

```bash
./run.sh
```

The first call creates `.venv` and installs `requirements-demo.txt` (5 to 8 minutes). Later calls start immediately. Do the first call at home on a good connection, not in the room.

**Option B: git clone.** The lite files (model weights, data) are not in git; fetch them once:

```bash
git clone https://github.com/cfcmadlad/embedding-truncation-repro.git
```

```bash
cd embedding-truncation-repro
```

```bash
python scripts/fetch_lite.py
```

then run `run.bat` or `./run.sh` as above. `python scripts/fetch_lite.py --zip` builds `dist/embedding-truncation-lite.zip` for the USB stick.

**Launcher modes.**

| Command | What it does | Time on a laptop CPU |
|---|---|---|
| `run.bat` | full live run: retrieval and classification | about 1 minute |
| `run.bat fast` | cuts the corpus to 400 documents (all judged ones kept); add `--threads 1` for a slow CPU | about 30 to 60 s |
| `run.bat replay` | prints the stored csv, labelled as stored | 1 s |
| `run.bat check` | prints ok or FAIL for Python, packages, files and a smoke run | 20 s |
| `run.bat selfcheck` | runs all of the above and writes `selfcheck_report_<host>.txt` | 3 minutes |
| `run.bat test` | pytest | 10 s |
| `run.bat streamlit` | Streamlit demo (needs the full cache, see 1b) | |

`./run.sh <mode>` takes the same modes. If the live run cannot encode (model files missing, any error), it prints the stored result instead, clearly labelled `STORED RESULT, NOT RECOMPUTED`.

The expected output, step by step, with the sentence to say for each part, is in [docs/RUN_SHEET.md](docs/RUN_SHEET.md). The 5-minute check for a teammate's own laptop is in [docs/SELF_CHECK.md](docs/SELF_CHECK.md).

### 1b. Full cache: Streamlit demo and all experiments

The demo and every experiment run on CPU from precomputed embeddings; no GPU is required. Setup takes about 15 minutes and 6 GB of free disk space.

**Requirements.** Python 3.12 or later (3.13 recommended; the pinned numpy 2.5.2 needs 3.12+) and Git. On Windows, the option "Add python.exe to PATH" must be ticked in the Python installer. The lite kit (1a) is enough for the live run; this section is only for the demo and for regenerating results.

**Step 1. Clone the repository.**

```bash
git clone https://github.com/cfcmadlad/embedding-truncation-repro.git
```

```bash
cd embedding-truncation-repro
```

**Step 2. Create and activate a virtual environment.**

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS or Linux:

```bash
source .venv/bin/activate
```

The terminal prompt then starts with `(.venv)`. The environment must be activated again in every new terminal before running the commands below.

**Step 3. Install the dependencies.** On a machine without an NVIDIA GPU, the small CPU build of PyTorch is installed first, then the rest:

```bash
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r requirements.txt
```

On a machine with an NVIDIA GPU, the second command alone is sufficient (it installs the CUDA build, about 2.5 GB).

**Step 4. Download the precomputed embeddings.** The embedding cache is too large for git, so it is published as two zip files (1.7 GB in total, 2.7 GB unpacked) on the repository's [`cache-v1` release](https://github.com/cfcmadlad/embedding-truncation-repro/releases/tag/cache-v1). The following command downloads both and unpacks them into `cache/`:

```bash
python scripts/fetch_cache.py
```

It prints download progress and ends with `cache/ ready`. If the download is interrupted, the same command with `--force` starts it again. Alternatively, `cache_part1.zip` and `cache_part2.zip` can be downloaded from the release page and extracted into the repository folder, so that the file `cache/emb/READY` exists.

**Step 5. Check the setup.** The command below prints full and truncated scores in about a second after the imports:

```bash
python scripts/demo_cli.py --model mpnet --dataset NanoSciFact --k 50 --method random --seed 0
```

Expected output: full nDCG@10 0.6967, truncated nDCG@10 0.6908, relative 99.15%.

**Step 6. Launch the demo.**

```bash
streamlit run demo/app.py
```

The demo opens in the browser. The first load takes about 10 seconds; later interactions take under 1.5 seconds, except that classification with PCA or the degrading and improving methods refits 10 logistic regressions (about 5 s the first time, then cached).

### Using the demo

The sidebar selects the model, the dataset, the truncation method (last-K, random-K with a seed, PCA, remove degrading dimensions, remove improving dimensions) and K, the percentage of dimensions removed. The main panel shows the full and truncated scores (nDCG@10 for retrieval, accuracy for classification), relative performance, the rank correlation between full and truncated rankings, and the truncation curve with the current setting marked against the paper's value. The query panel lists the top 5 documents for a chosen query before and after truncation, with rank changes marked. A new query can also be typed in; this is the only step that runs an encoder, and it needs that encoder on disk because the demo runs with `HF_HUB_OFFLINE=1`: save it once with `python scripts/fetch_lite.py --models mpnet` (any of minilm, mpnet, contriever, e5, mrl, nonmrl), or set `HF_HUB_OFFLINE=0` before launching to let it download from Hugging Face. The second tab, "Critique", shows the leakage, significance and sampling theory results.

### Command line fallback

The command line tool prints the same numbers as the demo from the cache and is a fallback if the browser app is unavailable (for the same setting it agrees with the live run, for example MiniLM, NanoSciFact, random 50%, seed 0: 0.4447):

```bash
python scripts/demo_cli.py --model mpnet --dataset NanoSciFact --k 50 --method random --seed 0
```

| Option | Values |
|---|---|
| `--model` | minilm, mpnet, contriever, e5, mrl, nonmrl |
| `--dataset` | any of the 13 NanoBEIR datasets (for example NanoSciFact, NanoFiQA2018) or a classification task (Banking77, Emotion, TweetSentimentExtraction, MTOPIntent) |
| `--method` | last, random, pca, degrading, improving |
| `--k` | percentage of dimensions removed, 0 to 90 |
| `--seed` | seed for random removal, 0 to 9 |

### Suggested live walkthrough

1. **Core claim.** MPNet base, NanoSciFact, random-K, K = 50: about 99% of nDCG@10 survives, and the current point lies on the paper's curve.
2. **The fragile model.** Contriever with K = 80: performance falls much faster, as in the paper.
3. **Rankings barely change.** In the query panel, the top 5 documents before and after truncation are nearly identical.
4. **The paper's explanation.** E5, NanoFiQA2018, remove degrading, K = 50: relative nDCG@10 exceeds 100% (108%). The Critique tab then shows that this gain disappears on held out queries (X1) and is not statistically significant (X2).
5. **Why random removal works.** In the same tab, the sampling theory prediction tracks the observed rank correlation (X3).

### Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognised | Reinstall Python with "Add python.exe to PATH" ticked, or use `py` instead of `python` on Windows |
| `No matching distribution found for numpy==2.5.2` | The Python version is older than 3.12; install Python 3.12 or later and recreate `.venv` |
| `FileNotFoundError` mentioning `cache/emb` | Step 4 has not completed; run `python scripts/fetch_cache.py --force` |
| `streamlit` is not recognised | The virtual environment is not active; run the activation command from Step 2 |
| `run.bat` says `FAIL: no Python 3.10 to 3.13` | Install Python 3.12 from python.org and tick "Add python.exe to PATH" |
| pip fails with `CERTIFICATE_VERIFY_FAILED` | Antivirus HTTPS scanning; the launcher retries once with certificate checks relaxed for pypi.org and files.pythonhosted.org only |
| The live run prints `STORED RESULT, NOT RECOMPUTED` | Encoding failed (model files missing or another error); run `run.bat check`, read the FAIL line, then `python scripts/fetch_lite.py` |
| Streamlit says the port is in use | Add `--server.port 8600` to the streamlit command |

## 2. The paper in brief

The paper removes K% of the dimensions of sentence embeddings, either the last K% or a random K%, and measures how much retrieval (nDCG@10) and classification (accuracy) survive. Its main claim is that removing a random half of the dimensions costs less than 10% on retrieval and less than 5% on classification. It then tests three explanations (anisotropy, dimensional collapse, outlier dimensions), rejects all three, and proposes its own: leave-one-dimension-out attribution finds many "degrading" dimensions whose removal improves performance, so random removal discards good and bad dimensions alike.

## 3. What this repository reproduces

| Paper element | Experiment | Status |
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
| Not in the paper | X1 selection leakage, X2 significance, X3 sampling theory | Extensions |

## 4. Headline results

| Claim | This reproduction | Paper |
|---|---|---|
| Random 50%, NanoBEIR nDCG@10 kept | 88.7 to 98.3% | 87.3 to 97.7% |
| Random 50%, MTEB accuracy kept | 93.1 to 97.0% | 95.4 to 97.7% |
| Full nDCG@10, MPNet / Contriever / MiniLM | 0.581 / 0.528 / 0.487 | 0.576 / 0.524 / 0.484 |
| Spearman at 50% last-K, MPNet / MiniLM | 0.899 / 0.885 | 0.90 / 0.88 |
| Outlier dims, MiniLM / MPNet / Contriever / E5 | 2 / 4 / 3 / 1 | 2 / 5 / 3 / 1 |
| PCA beats random 50% except Contriever retrieval | yes | yes |
| Removing degrading dims exceeds 100% | yes, in sample (up to 103.7%) | yes (up to about 103%) |

Main findings:

1. **The core claim reproduces.** Random removal of half the dimensions keeps most of the performance, with Contriever as the fragile outlier, as in the paper.
2. **The paper's explanation does not survive a held out test (X1, X2).** Degrading dimensions chosen on one half of the queries give no gain on the other half (best held out result 100.3%), and their per dimension effects on two disjoint query halves are uncorrelated (r between -0.05 and 0.01). The number of degrading dimensions with a bootstrap confidence interval excluding zero is at the level expected by chance, and the median effect of one dimension (3 to 5e-4 nDCG@10) is 23 to 43 times smaller than the standard error of the benchmark score (0.012).
3. **A simpler mechanism explains the robustness (X3).** A score q.d is a sum of per dimension products, and a random subset of dimensions gives an unbiased, scaled estimate of that sum. Rankings survive whenever contributions are spread over many dimensions. This model predicts the observed rank correlation at 50% within 0.03 for 5 of 6 models, and identifies Contriever as the model with the most concentrated contributions.

Every claim, with the reproduced number, the paper's number and a note on agreement, is in [RESULTS.md](RESULTS.md).

## 5. Scope

| Paper | This reproduction |
|---|---|
| 6 encoders | 4 core encoders (Paraphrase-MiniLM-L3, all-mpnet-base-v2, Contriever, E5-base-v2) plus an MRL / non-MRL MPNet pair |
| 14 BEIR datasets | All 13 NanoBEIR datasets |
| 12 MTEB classification tasks | Banking77, Emotion, TweetSentimentExtraction, MTOPIntent (en) |
| Contrastive retraining of BERT and T5 for anisotropy | Cross model comparison of the same metrics on 6 models |
| Llama 3.1 8B, Qwen 2.5 7B | Qwen2.5-0.5B |

Every reduction is justified, and every protocol choice and deviation is recorded, in [REPRO_NOTES.md](REPRO_NOTES.md).

## 6. Method and implementation

All truncation, metric, attribution and evaluation code is implemented independently in numpy and torch. Pretrained encoders and Hugging Face datasets are used only as inputs. The authors' code was not used.

**Encoding.** Each corpus is encoded once per model with sentence-transformers (fp16 on GPU) and cached as float32 `.npy` files with a metadata file recording model, dataset, prefix and normalisation. E5 uses its `query: ` and `passage: ` prefixes. All later experiments only slice these arrays.

**Truncation** (`src/truncate.py`). Last-K keeps the first D minus round(K x D) dimensions. Random-K draws round(K x D) dimensions to remove with a seeded generator, over 10 seeds (0 to 9) as in the paper. PCA is an SVD on 20,000 all-nli sentences, keeping D/2 components.

**Similarity.** Cosine for all models except Contriever, which uses the raw dot product. Cosine models are renormalised after truncation by the cosine itself.

**Metrics** (`src/metrics.py`). nDCG@10 with linear gain, as in pytrec_eval and BEIR, excluding a document whose id equals the query id, as BEIR and MTEB do. It matches pytrec_eval to 4e-8 per query. Classification follows the MTEB protocol: 10 experiments, each fitting a logistic regression (max_iter 100) on a class balanced subsample of the training set (8, 16 or 32 examples per label depending on the task), scored by accuracy on the test set. The same 10 subsamples are reused for every truncation setting.

**Attribution** (`src/attribution.py`). Leave-one-dimension-out is computed in closed form for all dimensions at once instead of rescoring the corpus D times. For cosine,

    score_without_i = (q.d - q_i d_i) / ( sqrt(|q|^2 - q_i^2) * sqrt(|d|^2 - d_i^2) )

and for the dot product the term q_i d_i is simply subtracted. A dimension is degrading if removing it raises the NanoBEIR mean nDCG@10.

**Sampling theory** (`src/theory.py`). For a document pair (a, b) and query q, let r_j = q_j (a_j - b_j). Keeping a random m of n dimensions estimates the score difference sum_j r_j with variance n^2 sigma_r^2 / m x (n - m) / (n - 1) (sampling without replacement). The probability that the pair swaps order follows from a normal approximation; averaging over pairs gives a predicted Kendall tau, converted to Spearman through the Gaussian relation.

## 7. Repository layout

```
docs/                     paper, assignment brief, reference numbers (PAPER_NOTES.md)
src/
  data.py                 NanoBEIR, MTEB classification and sentence set loaders
  encode.py               model registry, prefixes, encoding and caching
  metrics.py              nDCG@10, Spearman, MTEB logistic regression protocol
  truncate.py             last-K, random-K, index removal, PCA
  evaluate.py             retrieval and classification evaluators over cached or live arrays
  live.py                 subset, sweep, remove-vs-zero, classification plan, replay; shared by the live run
  runtime.py              thread count, CPU forcing, UTF-8 console
  attribution.py          leave-one-dimension-out, outliers, anisotropy metrics
  theory.py               subsampling variance and rank swap model
  paper.py                every reference number taken from the paper
  plotting.py             shared figure style and model colours
scripts/
  fetch_cache.py          download the precomputed embeddings from the release
  fetch_lite.py           save the MiniLM weights and the live-run data into lite/, build the zip
  live_small.py           live run: encode, truncate, compare with the paper, replay fallback
  check_setup.py          ok / FAIL check of Python, packages, files and a smoke run
  self_check.py           runs the checks and live runs and writes a report to send back
  encode_all.py           encode every model and dataset into cache/
  e0_full_scores.py       full dimension scores and pytrec_eval check
  e1 ... e10              paper experiments (section 3)
  x1, x2, x3              extensions (section 3)
  s1_llm_truncation.py    Qwen2.5-0.5B hidden state truncation
  demo_cli.py             command line fallback for the demo
demo/app.py, demo/core.py Streamlit demo
results/                  machine readable output of every experiment (csv, json)
figures/                  every figure as 200 dpi png (2400 x 1350) and pdf
run_all.sh                regenerates all results and figures
run.bat, run.sh           one-step launchers for the live run (venv, install, run)
requirements-demo.txt     CPU-only version ranges for the live run and the demo (Python 3.10 to 3.13)
tests/                    pytest: metrics, truncation, remove vs zero, subset, fallback
lite/                     MiniLM weights and live-run data (weights are not in git)
docs/RUN_SHEET.md         commands, expected output, what to say, fallbacks
docs/VIVA.md              likely viva questions with answers
docs/SELF_CHECK.md        5-minute check for a teammate's laptop
RESULTS.md                reproduced numbers vs the paper, claim by claim
REPRO_NOTES.md            scope reductions, protocol, deviations, runtimes, weaknesses
```

## 8. Reproducing all results

Every result and figure is regenerated from the cache with one script (about 1.5 hours on CPU). On Windows it runs in Git Bash. If `cache/emb/READY` is missing, it first encodes every dataset (about 50 minutes on a laptop RTX 4060, several hours on CPU).

```bash
bash run_all.sh
```

The Qwen2.5-0.5B experiment needs a GPU and downloads the model and benchmark items, so it runs only when requested:

```bash
RUN_S1=1 bash run_all.sh
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

Live run: `live_small_<model>_<dataset>.csv` and `live_small_cls_<model>_<task>.csv` (the `_fast` variants come from `--fast`). Other results: `e0_*` (full scores, encoding times, metric check), `e1_*` (all truncation runs per dataset and seed), `e2_*` (Spearman per dataset), `e3_pca.csv`, `e4_mrl.csv`, `e5_remove_vs_zero.json`, `e7_attribution_classification.csv`, `x1_*`, `x2_*`, `x3_*`, `s1_*` (LLM results and generated GSM8K answers).

## 10. Key design decisions

- **NanoBEIR instead of full BEIR.** The paper itself uses NanoBEIR for its largest model and for all attribution analyses, and its 13 datasets cover every BEIR domain at 50 queries each.
- **Closed form attribution.** Removing a dimension changes q.d by exactly q_i d_i, so all D leave-one-out scores follow from one score matrix; this makes E7, X1 and X2 take minutes instead of hours.
- **Removal equals zeroing.** For cosine and dot similarity, a removed coordinate and a zeroed coordinate contribute nothing to the dot product or the norms, so the two give identical scores (E5 confirms this to 1e-13). The paper's stated reason for preferring removal therefore does not apply.
- **Dot product for Contriever.** Checked against the paper: dot product reproduces its full score, random 50% and PCA numbers, while cosine does not (RESULTS.md, C3).
- **One device for all reported numbers.** CPU and GPU arithmetic differ by up to 2e-4 nDCG@10, the same size as single dimension effects, so every reported number comes from one CPU run.
- **Held out evaluation (X1).** Selecting dimensions and evaluating on the same queries cannot separate a real effect from selection on noise; splitting queries can.

## 11. Limitations

- Four of the twelve MTEB classification tasks are used, and NanoBEIR (50 queries per dataset) replaces full BEIR.
- E5-base-v2 stands in for E5-large; E5-Mistral (7B) and Sentence-T5 are not included.
- The LLM experiment uses Qwen2.5-0.5B instead of Qwen2.5-7B and Llama 3.1 8B.
- The anisotropy analysis compares 6 trained models instead of retraining BERT and T5 and tracking checkpoints.
- Cross model correlations rest on only 6 models.

The 7B to 8B models and the retraining experiment need more GPU memory than the 8 GB laptop GPU used here. The course HPC was considered for them, but on the run date the course account was limited to a shared half of one GPU (Slurm QOS `gres/gpu=0`, `mps=50`, 4 CPUs, 80 GB RAM), all five GPUs of the only partition open to it were allocated to multi day jobs, and 22 jobs were queued ahead. These experiments are therefore left as future work. Details and further weaknesses are listed in [REPRO_NOTES.md](REPRO_NOTES.md).

## 12. Hardware

Developed and run on a laptop: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), 16 logical CPU cores, 16 GB RAM, Windows 11, Python 3.13. Per step runtimes are in REPRO_NOTES.md.
