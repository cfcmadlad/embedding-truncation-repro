# 5-minute self-check for your own laptop

Do this at home, on your own connection, at least two days before 21 Nov. It tells us whether the live run works on your machine, how long it takes, and what to do if it does not. Do not skip the first step: installing the packages needs internet and takes 5 to 8 minutes, and that must not happen in the room.

## Steps

1. Copy `embedding-truncation-lite.zip` from the USB stick to your laptop and unzip it. A path with spaces is fine.
2. Open a terminal in the unzipped folder (the one containing `run.bat`).
   - Windows: right click in the folder and choose "Open in Terminal", or type `cmd` in the folder's address bar.
   - macOS or Linux: open Terminal, `cd` into the folder.
3. Run the self-check. It installs the packages on the first call (5 to 8 minutes), then runs four steps (about 3 minutes):

   Windows:

   ```bash
   .\run.bat selfcheck
   ```

   macOS or Linux:

   ```bash
   ./run.sh selfcheck
   ```

   The four steps are: `check` (ok or FAIL for Python, packages, model files, data files and a smoke run), a fast live run with 1 thread (a slow CPU), the full live run, and the replay of the stored result.
4. When it ends it prints a summary like this and writes `selfcheck_report_<your host name>.txt` in the same folder:

   ```
   check                             18.0 s  ok
   fast live run, 1 thread           45.0 s  ok
   full live run                     60.0 s  ok
   replay of the stored result        2.0 s  ok
   ```
5. Send us the file `selfcheck_report_<host>.txt` (it contains the machine description, the timings and the full output of every step). If you cannot send the file, send a screenshot of the summary.

## What good looks like

- All four lines say `ok`.
- The full live run takes under 2 minutes; the fast run takes under 90 s even with 1 thread.
- In the full live run, `VERDICT  the paper's claim holds in this live run.` appears, random 50% keeps about 94% of nDCG@10 and about 98% of accuracy. The last decimal can differ from ours by a few thousandths on another CPU.

## If something fails

| What you see | What to do |
|---|---|
| `FAIL: no Python 3.10 to 3.13 found` | Install Python 3.12 from python.org (Windows: tick "Add python.exe to PATH"), run the command again |
| pip errors in the first call | Check the internet, run the same command again; with antivirus HTTPS scanning the launcher retries on its own with relaxed certificate checks for pypi.org and files.pythonhosted.org |
| `FAIL  model files lite/models/minilm` | The zip was not unzipped completely; unzip again, or run `python scripts/fetch_lite.py` with internet |
| the live step prints `STORED RESULT, NOT RECOMPUTED` | Encoding failed on your machine. The report contains the error line (`live run failed: ...`); send it |
| a step takes more than 5 minutes | Close other programs; send the report anyway |

Whatever happens, send the report: a failing report is as useful to us as a passing one.
