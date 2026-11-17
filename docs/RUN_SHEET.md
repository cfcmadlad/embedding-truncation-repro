# Run sheet for the live demo

The instructor asked for a small version of the code run live on a subset of the data, showing the essence of the paper. The essence is one claim: removing a random half of the dimensions of a text embedding costs little. The live run tests that claim in front of the audience, on the laptop's CPU, in about a minute.

## Before the day (once, with good internet)

1. Unzip `embedding-truncation-lite.zip` (or clone the repo and run `python scripts/fetch_lite.py`).
2. Run the first call, which installs the packages (5 to 8 minutes):

   ```bash
   .\run.bat check
   ```

   On macOS or Linux use `./run.sh check`. Every line must say `ok` (the lines about `pytrec_eval` and the full cache may say `absent`).
3. Run `run.bat selfcheck` once ([SELF_CHECK.md](SELF_CHECK.md)) and keep the report.
4. Close other programs on the day, plug in the charger, and make sure the laptop is not in battery saver mode.

## In the room

Open a terminal in the kit folder. Say: "We reproduce the paper's main claim live on this laptop, on the CPU, on a subset of the data."

```bash
.\run.bat
```

(`./run.sh` on macOS or Linux; `.\run.bat fast` if the laptop is slow.) The output appears in the order below. Every printed number is tied to a claim of the paper.

### Step 1. Model and startup (about 13 s)

```
model      sentence-transformers/paraphrase-MiniLM-L3-v2 (384 dimensions, cos similarity)
startup    imports and encoder load 12.8 s (model load alone 9.4 s)
```

Say: "This is the paper's Paraphrase-MiniLM, 17 million parameters, 384 dimensions. Loading it takes a few seconds; nothing is downloaded."
Paper: one of its 6 encoders (Table 5).
Fallback: if this step raises an error, the script prints the stored result by itself (step 8).

### Step 2. Live encoding (about 25 s)

```
dataset    NanoSciFact: 50 queries, 2919 of 2919 documents
encoded    2969 texts live in 25.4 s (117 texts per second, cpu, 8 threads)
```

Say: "These are 50 scientific claims and 2,919 abstracts from NanoSciFact, a small sample of BEIR. We encode all of them now, on the CPU."
Paper: BEIR retrieval, nDCG@10 (Sec 2.1); NanoBEIR is what the paper itself uses for its largest model and for attribution.

### Step 3. Full score and the cache check

```
cache      live document embeddings differ from the cached ones behind RESULTS.md by at most 1.1e-03
full       nDCG@10 0.4713
```

(The `cache` line appears only where the full cache is present.)
Say: "With all 384 dimensions the retrieval quality is 0.4713. The tiny difference to our stored embeddings comes from fp16 on the GPU against fp32 on the CPU."
Paper: full scores are the denominator of every "relative performance" in the paper (Fig 1, 2).

### Step 4. The truncation table (the main claim)

```
 removed    last-K   random-K     std  kept (%)  paper avg (%)
      0%    0.4713     0.4713  0.0000     100.0          100.0
     ...
     50%    0.4481     0.4438  0.0168      94.2           95.5
     ...
     90%    0.2637     0.2294  0.0525      48.7           62.0
```

Say: "Each row removes that percentage of the dimensions. `last-K` removes the last ones, `random-K` removes random ones, averaged over 10 seeds. `kept` is the score as a percentage of the full score. The last column is the paper's own curve from its Figure 1. At 50% removal we keep 94.2% of the quality; the paper's average is 95.5%, and for this model on BEIR its Table 2 says 94.6%."
Paper: Figure 1 and 2 (random K), Table 2 (94.6% for Para-MiniLM). The claim is "more than 90% kept at 50%".
Note: one dataset with 50 queries is noisy (std column), so single rows wobble above and below the paper's curve, and the paper's column is an average over 6 models and many datasets. Our 90% row (48.7%) is below that average; the paper's Figure 9 shows how widely single datasets spread around it.

### Step 5. Removing equals zeroing

```
remove vs zero   the same 192 dimensions removed or zeroed: nDCG@10 0.4447 and 0.4447, largest score difference 4.2e-07
stored     random 50% seed 0 in results/e1_retrieval.csv (cached embeddings): 0.4447; this live run: 0.4447
```

Say: "The paper removes dimensions instead of zeroing them, because it thinks zeroing may behave differently. For cosine and dot-product similarity both give exactly the same scores: a zero adds nothing to a dot product or to a norm. The difference here is float rounding. The second line shows the live number agrees with our stored result."
Paper: Sec 4 (method), the justification for removal. Our finding E5 in RESULTS.md.
Fallback: if the `stored` line is missing, the full cache is absent; that is normal in the lite kit.

### Step 6. Classification (about 10 s plus 5 s of fits)

```
task       Banking77: 1182 training texts (2 MTEB subsample(s) of 8 per label), 1000 test texts
full       accuracy 0.7750 (logistic regression, 5.1 s for the sweep)
     50%    0.7560     0.7569  0.0052      97.7           97.5
```

Say: "The title of the paper has two halves, retrieval and classification. Here we classify bank customer questions into 77 intents with a logistic regression on 8 examples per intent, following the MTEB protocol, on the same truncated embeddings. At 50% removal we keep 97.7% of the accuracy; the paper's average is 97.5%, and 97.4% for this model."
Paper: Fig 1 (classification curve), Fig 2b and 2d, Table 2 (MTEB columns: more than 95%).
Note: 2 of MTEB's 10 subsamples and 1,000 test texts, so the numbers are close to but not identical with RESULTS.md.

### Step 7. Verdict

```
retrieval       random 50% keeps 94.2% of nDCG@10 (paper bar: over 90%, paper Para-MiniLM: 94.6%)
classification  random 50% keeps 97.7% of accuracy (paper bar: over 95%, paper Para-MiniLM: 97.4%)
VERDICT  the paper's claim holds in this live run.
total      43.4 s
```

Say: "Both halves of the paper's claim hold on this laptop, in under a minute: more than 90% of retrieval and more than 95% of classification survive a random 50% cut. The paper then tries to explain why. We tested its explanation on our cached results, and that is the second half of our work."
If the verdict says a bar was missed: say it. The numbers above it are the evidence; a single small dataset can fall below a bar by a point or two, and that is the noise level we report in RESULTS.md (X2).

### Step 8. If anything goes wrong: the replay

If encoding fails for any reason the script prints the error and then the stored result of an earlier live run, labelled:

```
live run failed: OSError: ...
STORED RESULT, NOT RECOMPUTED (live encoding failed). This is the csv written by an earlier live run:
```

Say: "The live encoding failed on this machine, so this is the table from the live run we did earlier; it is a stored result, not a new measurement." Never present it as live.
You can also request it directly: `.\run.bat replay`.

## Optional: the Streamlit demo (needs the full cache)

```bash
.\run.bat streamlit
```

First load about 10 s, then every click under 1.5 s. Walkthrough: MPNet, NanoSciFact, random-K, K = 50 (99.2%); Contriever with K = 80 (falls fast); query panel (rankings barely change); E5, NanoFiQA2018, remove degrading, K = 50 (above 100%), then the Critique tab for the held-out test (X1), the bootstrap (X2) and the sampling theory (X3). Fallback: `python scripts/demo_cli.py --model mpnet --dataset NanoSciFact --k 50` prints the same numbers.

## Failure modes and responses

| Failure | What you see | Response |
|---|---|---|
| No internet in the room | Nothing: the live run never uses the network (HF offline mode, local weights and data) | Carry on. Only the first install needed internet |
| Model not on disk | `live run failed: OSError ...` then `STORED RESULT, NOT RECOMPUTED` | Say it is stored; later run `python scripts/fetch_lite.py` |
| Slow CPU | encoding takes longer than 1 minute | Use `.\run.bat fast`; with `--threads 1` it takes about 1 minute even on a slow CPU |
| Antivirus scans every file | slow imports (13 s here), pip `CERTIFICATE_VERIFY_FAILED` at install time | Install beforehand; the launcher retries pip with relaxed certificate checks for pypi.org and files.pythonhosted.org only; in the room run `.\run.bat` once beforehand so the files are already scanned |
| Python too old or missing | `FAIL: no Python 3.10 to 3.13 found` | Install Python 3.12 from python.org, tick "Add python.exe to PATH" |
| Streamlit port 8501 in use | Streamlit starts on 8502 by itself | Use the URL it prints; or add `--server.port 8600` |
| Console is not UTF-8 | The output contains only ASCII; the script sets UTF-8 itself | Nothing to do |
| Spaces in the folder path | Nothing: tested in `D:\venv test\lite kit\...` | Nothing to do |
| `run.bat` is not recognised in PowerShell | `'run.bat' is not recognized` | Type `.\run.bat` (with the dot and backslash) |
| macOS says the script cannot be opened | permission denied | `chmod +x run.sh` once |
| Windows SmartScreen on `run.bat` | warning dialog | "More info", then "Run anyway", or run `python scripts/live_small.py` directly |

## Rehearsal record (GPU hidden with `CUDA_VISIBLE_DEVICES=-1`)

See the table in REPRO_NOTES.md, section "Live run".
