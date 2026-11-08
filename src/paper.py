from __future__ import annotations

PAPER_OF = {"minilm": "Para-MiniLM", "mpnet": "MPNet", "contriever": "Contriever", "e5": "E5-large"}

FIG1_RANDOM_AVG = {
    "retrieval": [100.0, 99.7, 99.0, 98.0, 97.5, 95.5, 94.0, 90.5, 83.0, 62.0],
    "classification": [100.0, 99.8, 99.4, 98.8, 98.3, 97.5, 96.5, 95.2, 92.3, 85.0],
}

TABLE2 = {
    "MPNet": {"beir_trunc": 97.7, "beir_pca": 99.4, "mteb_trunc": 96.6, "mteb_pca": 99.6},
    "Contriever": {"beir_trunc": 87.3, "beir_pca": 77.3, "mteb_trunc": 95.4, "mteb_pca": 98.5},
    "E5-large": {"beir_trunc": 95.9, "beir_pca": 90.6, "mteb_trunc": 97.7, "mteb_pca": 99.8},
    "E5-Mistral": {"beir_trunc": 99.6, "beir_pca": 100.6, "mteb_trunc": 98.2, "mteb_pca": 100.4},
    "Para-MiniLM": {"beir_trunc": 94.6, "beir_pca": 99.3, "mteb_trunc": 97.4, "mteb_pca": 99.5},
    "ST5": {"beir_trunc": 95.9, "beir_pca": 100.2, "mteb_trunc": 97.8, "mteb_pca": 99.9},
}

FIG8_SPEARMAN = {
    "MPNet": {50: 0.90, 90: 0.59},
    "Contriever": {50: 0.85, 90: 0.43},
    "E5-large": {50: 0.88, 90: 0.55},
    "E5-Mistral": {50: 0.95, 90: 0.74},
    "Para-MiniLM": {50: 0.88, 90: 0.56},
    "ST5": {50: 0.92, 90: 0.63},
}

TABLE1 = {
    "MPNet": {"n_outliers": 5, "without_outliers": 0.576, "random": 0.576, "random_std": 9e-4},
    "Contriever": {"n_outliers": 3, "without_outliers": 0.525, "random": 0.524, "random_std": 1e-3},
    "E5-large": {"n_outliers": 1, "without_outliers": 0.577, "random": 0.577, "random_std": 4e-4},
    "E5-Mistral": {"n_outliers": 32, "without_outliers": 0.651, "random": 0.626, "random_std": 6e-4},
    "Para-MiniLM": {"n_outliers": 2, "without_outliers": 0.483, "random": 0.483, "random_std": 6e-4},
    "ST5": {"n_outliers": 2, "without_outliers": 0.489, "random": 0.489, "random_std": 9e-4},
}

FIG4_COUNTS = {
    "MPNet": {"dims": 768, "nanobeir": (595, 169, 4), "mteb": (246, 522, 0)},
    "Contriever": {"dims": 768, "nanobeir": (498, 270, 0), "mteb": (398, 370, 0)},
    "E5-large": {"dims": 1024, "nanobeir": (308, 716, 0), "mteb": (430, 594, 0)},
    "E5-Mistral": {"dims": 4096, "nanobeir": (2652, 1419, 25), "mteb": (1932, 2164, 0)},
    "Para-MiniLM": {"dims": 384, "nanobeir": (117, 267, 0), "mteb": (182, 202, 0)},
    "ST5": {"dims": 768, "nanobeir": (204, 564, 0), "mteb": (277, 491, 0)},
}

FULL_NANOBEIR = {"MPNet": 0.5755, "Contriever": 0.524, "E5-large": 0.5765, "E5-Mistral": 0.6264, "Para-MiniLM": 0.4835, "ST5": 0.4895}

ANISOTROPY_PEARSON = {
    "T5": {"uniform_loss": 0.36, "isoscore": -0.47, "corr_mean": 0.45},
    "BERT": {"uniform_loss": -0.35, "isoscore": 0.39, "corr_mean": -0.30},
}

QWEN_TABLE3 = {
    "MMLU acc": {"full": 0.718, "first": 0.709, "last": 0.709},
    "GSM8K strict EM": {"full": 0.766, "first": 0.045, "last": 0.011},
    "HellaSwag acc": {"full": 0.718, "first": 0.709, "last": 0.709},
}
