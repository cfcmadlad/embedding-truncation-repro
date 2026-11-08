from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import core
from src.encode import MODELS
from src.paper import FIG1_RANDOM_AVG, PAPER_OF, TABLE2
from src.plotting import MODEL_COLORS, PAPER

st.set_page_config(page_title="Random truncation of text embeddings", layout="wide")


@st.cache_data(show_spinner=False)
def cached_eval(model: str, dataset: str, method: str, frac: float, seed: int) -> dict:
    return core.evaluate(model, dataset, method, frac, seed)


@st.cache_data(show_spinner=False)
def cached_top(model: str, dataset: str, method: str, frac: float, seed: int, query: str | None, qi: int | None):
    return core.top_docs(model, dataset, method, frac, seed, query, qi)


def curve(model: str, dataset: str, method: str, frac: float, current: float) -> plt.Figure:
    is_cls = dataset in core.CLASSIFICATION
    per = core.read_csv("e1_classification.csv" if is_cls else "e1_retrieval.csv")
    summary = core.read_csv("e1_classification_summary.csv" if is_cls else "e1_retrieval_summary.csv")
    col, dcol = ("accuracy", "task") if is_cls else ("ndcg", "dataset")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = np.arange(0, 100, 10)
    kind = "last" if method == "last-K" else "random"
    if per is not None:
        sub = per[(per.model == model) & (per[dcol] == dataset)]
        full = sub[(sub.method == "last") & (sub.frac == 0)][col]
        if len(full):
            g = sub[sub.method == kind].groupby("frac")[col]
            m, s = g.mean() / full.iloc[0] * 100, g.std().fillna(0) / full.iloc[0] * 100
            ax.errorbar(m.index * 100, m.values, yerr=s.values, color=MODEL_COLORS[model], marker="o", label=f"ours, {dataset}, {kind}-K")
    if summary is not None:
        sm = summary[(summary.model == model) & (summary.method == kind)].sort_values("frac")
        if len(sm):
            ax.plot(sm.frac * 100, sm.rel_mean, color=MODEL_COLORS[model], ls=":", label=f"ours, average, {kind}-K")
    ax.plot(xs, FIG1_RANDOM_AVG["classification" if is_cls else "retrieval"], color=PAPER, ls="--", label="paper Fig 1, average")
    paper_model = PAPER_OF.get(model)
    if paper_model:
        ax.scatter([50], [TABLE2[paper_model]["mteb_trunc" if is_cls else "beir_trunc"]], marker="s", s=90, color=PAPER, zorder=5, label=f"paper Table 2, {paper_model}")
    ax.scatter([frac * 100], [current], marker="*", s=350, color="gold", edgecolor="black", zorder=6, label=f"current: {method}")
    ax.set_xlabel("Dimensions removed (%)")
    ax.set_ylabel("Relative performance (%)")
    ax.set_xticks(xs)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10)
    fig.tight_layout()
    return fig


@st.cache_resource(show_spinner=False)
def encoder_warmup(model: str) -> threading.Thread:
    t = threading.Thread(target=core.warm_encoder, args=(model,), daemon=True)
    t.start()
    return t


def side(model: str, dataset: str, method: str, frac: float, seed: int) -> None:
    warm = encoder_warmup(model)
    ds = core.texts(dataset)
    st.subheader("Query panel")
    choice = st.selectbox("Query", range(len(ds.queries)), format_func=lambda i: ds.queries[i][:120])
    typed = st.text_input("Or type a new query (live encoder call on CPU)")
    query = typed.strip() or None
    if query and warm.is_alive():
        with st.spinner("Loading the encoder (first query only)"):
            warm.join()
    with st.spinner("Encoding query" if query else "Ranking"):
        full, trunc = cached_top(model, dataset, method, frac, seed, query, None if query else int(choice))
    left, right = st.columns(2)
    for col, rows, title, other, sign in ((left, full, "Full embedding", "truncated", 1), (right, trunc, "Truncated embedding", "full", -1)):
        col.markdown(f"**{title}**")
        for r, row in enumerate(rows, 1):
            delta = sign * (row["other_rank"] - r)
            mark = "unchanged" if delta == 0 else (f"down {delta} under truncation" if delta > 0 else f"up {-delta} under truncation")
            rel = " | relevant" if row["relevant"] else ""
            col.markdown(f"**{r}.** rank in {other}: {row['other_rank']} ({mark}){rel}  \n{row['doc']}")


def critique() -> None:
    items = [
        ("fig_leakage", "X1 selection leakage", "x1_leakage.csv"),
        ("fig_bootstrap", "X2 bootstrap significance", "x2_bootstrap.csv"),
        ("fig_theory_vs_observed", "X3 theory vs observed", "x3_theory_vs_observed.csv"),
        ("fig_concentration_vs_drop", "X3 concentration vs drop", "x3_concentration.csv"),
    ]
    for fig, title, csv in items:
        st.subheader(title)
        png = core.FIGURES / f"{fig}.png"
        if png.exists():
            st.image(str(png))
        else:
            st.info(f"figures/{fig}.png not produced yet")
        df = core.read_csv(csv)
        if df is not None:
            st.dataframe(df.head(20), width="stretch")


def main() -> None:
    st.sidebar.header("Setting")
    model = st.sidebar.selectbox("Model", list(MODELS), format_func=lambda k: f"{k}: {MODELS[k].hf_id}")
    dataset = st.sidebar.selectbox("Dataset", core.DATASETS)
    method = st.sidebar.selectbox("Truncation method", core.METHODS)
    seed = st.sidebar.number_input("Seed (random-K)", 0, 9, 0) if method == "random-K" else 0
    k = st.sidebar.slider("K: % of dimensions removed", 0, 90, 50, 10)
    frac = round(k / 100, 1)
    st.title("Randomly removing dimensions of text embeddings")
    st.caption("Takeshita et al., EMNLP 2025. Reproduction on cached embeddings, CPU only.")
    tab_main, tab_crit = st.tabs(["Truncation", "Critique: leakage and theory"])
    with tab_main:
        start = time.perf_counter()
        try:
            with st.spinner("Evaluating"):
                res = cached_eval(model, dataset, method, frac, int(seed))
        except FileNotFoundError as e:
            st.warning(f"Missing results file for this method: {e}")
            return
        c = st.columns(3)
        c[0].metric(f"Full {res['metric']}", f"{res['full']:.4f}")
        c[1].metric(f"Truncated {res['metric']}", f"{res['truncated']:.4f}", f"{res['truncated'] - res['full']:+.4f}")
        c[2].metric("Relative", f"{res['relative']:.1f}%")
        c = st.columns(3)
        c[0].metric("Mean Spearman, full vs truncated ranking", "n/a" if res["spearman"] is None else f"{res['spearman']:.3f}")
        c[1].metric("Dims removed", f"{res['removed']} / {MODELS[model].dim}")
        st.pyplot(curve(model, dataset, method, frac, res["relative"]), clear_figure=True)
        st.caption(f"Computed in {time.perf_counter() - start:.2f} s")
        if dataset not in core.CLASSIFICATION:
            side(model, dataset, method, frac, int(seed))
    with tab_crit:
        critique()


main()
