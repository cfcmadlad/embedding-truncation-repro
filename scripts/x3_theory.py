from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

from src.encode import MODELS
from src.evaluate import retrieval_suite
from src.metrics import n_jobs, spearman_rows
from src.plotting import MODEL_COLORS, RES_DIR, figure, save
from src.theory import contribution_energy, copula_spearman, observed_tau, pair_moments, participation_ratio, predicted_tau
from src.truncate import SEEDS, keep_random, n_removed

FRACS = [round(0.1 * i, 1) for i in range(1, 10)]
SAMPLE = 300


def run_model(model: str) -> tuple[list[dict], float]:
    suite = retrieval_suite(model)
    n = MODELS[model].dim
    acc: dict[tuple, list] = {}
    energy = []
    for i, ev in enumerate(suite.values()):
        rng = np.random.default_rng(i)
        nq, nd = ev.q.shape[0], ev.d.shape[0]
        idx = torch.from_numpy(np.stack([rng.choice(nd, SAMPLE, replace=False) for _ in range(nq)])).to(ev.q.device)
        q, d = ev.q, ev.d
        if ev.sim == "cos":
            q = torch.nn.functional.normalize(q, dim=-1)
            d = torch.nn.functional.normalize(d, dim=-1)
        docs = d[idx]
        delta, s2 = pair_moments(q, docs)
        energy.append(contribution_energy(q, docs).cpu())
        full = ev.scores()
        full_s = torch.gather(full, 1, idx)
        for frac in FRACS:
            m = n - n_removed(n, frac)
            tau_p = predicted_tau(delta, s2, n, m).mean().item()
            acc.setdefault((frac, "pred_tau"), []).append(tau_p)
            obs_tau, obs_rho, obs_full = [], [], []
            for seed in SEEDS:
                t = ev.scores(keep_random(n, frac, seed))
                t_s = torch.gather(t, 1, idx)
                obs_tau.append(observed_tau(full_s, t_s).mean().item())
                obs_rho.append(spearman_rows(full_s, t_s).mean().item())
                obs_full.append(spearman_rows(full, t).mean().item())
            acc.setdefault((frac, "obs_tau"), []).append(np.asarray(obs_tau))
            acc.setdefault((frac, "obs_rho"), []).append(np.asarray(obs_rho))
            acc.setdefault((frac, "obs_full"), []).append(np.asarray(obs_full))
    rows = []
    for frac in FRACS:
        pred_tau = float(np.mean(acc[(frac, "pred_tau")]))
        seed_tau = np.mean(acc[(frac, "obs_tau")], axis=0)
        seed_rho = np.mean(acc[(frac, "obs_rho")], axis=0)
        seed_full = np.mean(acc[(frac, "obs_full")], axis=0)
        rows.append(
            {
                "model": model,
                "frac": frac,
                "pred_tau": pred_tau,
                "obs_tau": seed_tau.mean(),
                "obs_tau_std": seed_tau.std(ddof=1),
                "pred_spearman": float(copula_spearman(pred_tau)),
                "obs_spearman_sample": seed_rho.mean(),
                "obs_spearman_sample_std": seed_rho.std(ddof=1),
                "obs_spearman_full": seed_full.mean(),
                "obs_spearman_full_std": seed_full.std(ddof=1),
            }
        )
    pr = participation_ratio(torch.stack(energy).mean(0).numpy())
    return rows, pr


def plot_theory(df: pd.DataFrame) -> None:
    fig, ax = figure()
    for model in df.model.unique():
        d = df[df.model == model]
        x = 100 * d.frac
        ax.plot(x, d.obs_spearman_sample, color=MODEL_COLORS[model], marker="o")
        ax.plot(x, d.pred_spearman, color=MODEL_COLORS[model], ls="--", marker="x")
    ax.set_xlabel("Randomly removed dimensions (%)")
    ax.set_ylabel("Spearman, full vs truncated ranking")
    handles = [Line2D([], [], color=MODEL_COLORS[m], lw=4, label=MODELS[m].label) for m in df.model.unique()]
    handles += [
        Line2D([], [], color="black", marker="o", label="Observed"),
        Line2D([], [], color="black", ls="--", marker="x", label="Predicted"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    save(fig, "fig_theory_vs_observed")


def correlations(conc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["participation_ratio", "pred_spearman50", "isoscore", "uniformity", "corr_mean"]:
        if metric not in conc:
            continue
        d = conc.dropna(subset=[metric, "rel50_random"])
        if len(d) < 3:
            continue
        rows.append(
            {
                "metric": metric,
                "n_models": len(d),
                "pearson": pearsonr(d[metric], d.rel50_random)[0],
                "spearman": spearmanr(d[metric], d.rel50_random)[0],
            }
        )
    return pd.DataFrame(rows)


def plot_concentration(conc: pd.DataFrame, corr: pd.DataFrame) -> None:
    metrics = [
        ("participation_ratio", "Participation ratio"),
        ("pred_spearman50", "Predicted Spearman, 50%"),
        ("isoscore", "IsoScore"),
        ("corr_mean", "Mean |inter-dim corr|"),
    ]
    r = corr.set_index("metric")
    fig, axes = figure(2, 2, sharey=True)
    for ax, (metric, label) in zip(axes.flat, metrics):
        if metric not in conc:
            ax.set_visible(False)
            continue
        d = conc.dropna(subset=[metric])
        for _, row in d.iterrows():
            ax.scatter(row[metric], row.rel50_random, color=MODEL_COLORS[row.model], s=120, zorder=3)
        txt = f"r = {r.loc[metric, 'pearson']:.2f}" if metric in r.index else "r = n/a"
        ax.set_xlabel(f"{label}\n{txt}")
    fig.supylabel("Relative nDCG@10 at 50% random removal (%)", fontsize=18)
    handles = [Line2D([], [], color=MODEL_COLORS[m], marker="o", ls="", markersize=12, label=MODELS[m].label) for m in conc.model]
    axes[0, 1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    save(fig, "fig_concentration_vs_drop")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(n_jobs())
    pr_path = RES_DIR / "x3_participation.csv"
    if args.reuse and pr_path.exists():
        df = pd.read_csv(RES_DIR / "x3_theory_vs_observed.csv")
        prs = pd.read_csv(pr_path).set_index("model").participation_ratio.to_dict()
    else:
        rows, prs = [], {}
        for model in MODELS:
            r, pr = run_model(model)
            rows += r
            prs[model] = pr
            print(model, round(pr, 4), flush=True)
        df = pd.DataFrame(rows)
        df["abs_error"] = (df.pred_spearman - df.obs_spearman_sample).abs()
        df.to_csv(RES_DIR / "x3_theory_vs_observed.csv", index=False)
        pd.DataFrame({"model": list(prs), "participation_ratio": list(prs.values())}).to_csv(pr_path, index=False)
    plot_theory(df)
    e1 = pd.read_csv(RES_DIR / "e1_retrieval_summary.csv")
    rel50 = e1[(e1.method == "random") & np.isclose(e1.frac, 0.5)].set_index("model").rel_mean
    conc = pd.DataFrame({"model": list(prs), "participation_ratio": list(prs.values())})
    at50 = df[np.isclose(df.frac, 0.5)].set_index("model")
    conc["pred_spearman50"] = conc.model.map(at50.pred_spearman)
    conc["obs_spearman50"] = conc.model.map(at50.obs_spearman_sample)
    conc["rel50_random"] = conc.model.map(rel50)
    aniso_path = RES_DIR / "e10_anisotropy.csv"
    if aniso_path.exists():
        an = pd.read_csv(aniso_path).set_index("model")
        for col in ["isoscore", "uniformity", "corr_mean"]:
            conc[col] = conc.model.map(an[col]) if col in an else np.nan
    conc.to_csv(RES_DIR / "x3_concentration.csv", index=False)
    corr = correlations(conc)
    corr.to_csv(RES_DIR / "x3_correlations.csv", index=False)
    plot_concentration(conc, corr)
    print(df.groupby("model").abs_error.agg(["mean", "max"]).round(4).to_string())
    print(conc.round(4).to_string(index=False))
    print(corr.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
