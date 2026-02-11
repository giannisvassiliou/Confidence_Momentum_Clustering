#!/usr/bin/env python3
"""


Multi-run version of FULL13_clean_with_OSKM_ablations.py (20 Newsgroups).

Adds:
- --runs flag (independent runs; run_seed = base_seed + run_id)
- Aggregated statistics (mean, std, median, min, max, 95% CI half-width) for NMI/ARI/Silhouette
- CSV outputs:
    - results_long.csv   (one row per run × algorithm × mode)
    - results_summary.csv (aggregated per algorithm × mode)
- Optional best-effort single-thread mode (--single_thread) to reduce BLAS/OpenMP nondeterminism
- Optional per-run SVD randomness (--svd_seed_per_run)

Usage:
  python FULL13_clean_with_OSKM_ablations_multirun.py --runs 10
  python FULL13_clean_with_OSKM_ablations_multirun.py --runs 20 --single_thread --output_dir out_full13
  python FULL13_clean_with_OSKM_ablations_multirun.py --runs 10 --svd_seed_per_run

Notes:
- Online (streaming) clustering is path-dependent; multiple runs are the correct way to report results.
- This script keeps preprocessing consistent by default (TF-IDF and SVD fitted once). Use --svd_seed_per_run
  if you explicitly want representation randomness across runs (slower).

"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from collections import deque, defaultdict

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize


# =========================
# Defaults / Config (mirrors original)
# =========================
SEED = 42

# Dataset
SUBSET = "train"  # "train" or "test" or "all"
REMOVE_META = ("headers", "footers", "quotes")

# Preprocessing
TFIDF_MAX_FEATURES = 20000
TFIDF_STOP_WORDS = "english"
TFIDF_MAX_DF = 0.8
TFIDF_MIN_DF = 1
TFIDF_NGRAM_RANGE = (1, 1)

SVD_COMPONENTS = 150  # set to 0 to disable SVD

# Evaluation sizes
SAMPLE_SIZE = 2000
TRAIN_SIZE = 2000
EVAL_SIZE = 2000

# Algorithms
K = 20  # 20 Newsgroups has 20 classes
LEARNING_RATE = 0.1
ALPHA = 0.05
MOMENTUM = 0.9


# =========================
# Utilities
# =========================
def set_global_seed(seed: int) -> None:
    np.random.seed(seed)


def dict_from_row(X: np.ndarray, idx: int) -> Dict[str, float]:
    row = X[idx]
    return {f"f{j}": float(row[j]) for j in range(row.shape[0])}


def normalize_vec(z: np.ndarray) -> np.ndarray:
    return normalize(z.reshape(1, -1))[0]


def get_initial_centroids(k: int, n_features: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centroids = rng.normal(0.0, 0.1, size=(k, n_features))
    return normalize(centroids)


def mean_ci95(x: np.ndarray) -> Tuple[float, float]:
    """
    95% CI for the mean using normal approximation: mean ± 1.96 * (std/sqrt(n)).
    """
    x = np.asarray(x, dtype=float)
    n = int(len(x))
    if n <= 1:
        return float(x.mean()), 0.0
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    half = 1.96 * (sd / np.sqrt(n))
    return mu, float(half)


# =========================
# Algorithms (as in original)
# =========================
class OSKMWithAdamOnSphere:
    def __init__(
        self,
        k: int,
        learning_rate: float,
        initial_centroids: np.ndarray,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        use_time_decay: bool = True,
    ):
        self.k = k
        self.learning_rate = learning_rate
        self.centroids = initial_centroids.copy()

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.use_time_decay = use_time_decay

        self.m = np.zeros_like(self.centroids)
        self.v = np.zeros_like(self.centroids)
        self.t = np.zeros((k,), dtype=np.int64)
        self.n_samples_seen = 0

    def learn_one(self, x_dict: Dict[str, float]):
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        sims = self.centroids @ z
        i = int(np.argmax(sims))

        mu = self.centroids[i]
        dot = float(np.dot(mu, z))
        g = z - dot * mu

        lr = self.learning_rate / (1.0 + 0.0005 * self.n_samples_seen) if self.use_time_decay else self.learning_rate

        self.t[i] += 1
        ti = int(self.t[i])

        self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
        self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)

        mhat = self.m[i] / (1.0 - (self.beta1 ** ti))
        vhat = self.v[i] / (1.0 - (self.beta2 ** ti))

        mu_new = mu + lr * (mhat / (np.sqrt(vhat) + self.eps))
        self.centroids[i] = normalize_vec(mu_new)

        self.n_samples_seen += 1
        return self

    def predict_one(self, x_dict: Dict[str, float]) -> int:
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        sims = self.centroids @ z
        return int(np.argmax(sims))


class OnlineSphericalKMeans_Fair:
    def __init__(self, k: int, learning_rate: float = 0.05, initial_centroids: Optional[np.ndarray] = None, seed: int = 42):
        self.k = k
        self.learning_rate = learning_rate
        self.initial_centroids = initial_centroids
        self.seed = seed
        self.centroids: Optional[np.ndarray] = None
        self.n_samples_seen = 0

    def _ensure_centroids(self, dim: int) -> None:
        if self.centroids is not None:
            return
        if self.initial_centroids is not None:
            self.centroids = self.initial_centroids.copy()
        else:
            rng = np.random.default_rng(self.seed)
            c = rng.normal(0.0, 0.1, size=(self.k, dim))
            self.centroids = normalize(c)

    def learn_one(self, x_dict: Dict[str, float]):
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        self._ensure_centroids(len(z))
        assert self.centroids is not None

        sims = self.centroids @ z
        i = int(np.argmax(sims))

        lr = self.learning_rate / (1.0 + 0.0005 * self.n_samples_seen)
        self.centroids[i] = normalize_vec(self.centroids[i] + lr * (z - self.centroids[i]))
        self.n_samples_seen += 1
        return self

    def predict_one(self, x_dict: Dict[str, float]) -> Optional[int]:
        if self.centroids is None:
            return None
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        sims = self.centroids @ z
        return int(np.argmax(sims))


class OSKMWithMomentum:
    def __init__(self, k: int, learning_rate: float = 0.05, momentum: float = 0.9,
                 initial_centroids: Optional[np.ndarray] = None, seed: int = 42):
        self.k = k
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.initial_centroids = initial_centroids
        self.seed = seed
        self.centroids: Optional[np.ndarray] = None
        self.velocity: Optional[np.ndarray] = None
        self.n_samples_seen = 0

    def _ensure_centroids(self, dim: int) -> None:
        if self.centroids is not None:
            return
        if self.initial_centroids is not None:
            self.centroids = self.initial_centroids.copy()
        else:
            rng = np.random.default_rng(self.seed)
            c = rng.normal(0.0, 0.1, size=(self.k, dim))
            self.centroids = normalize(c)
        self.velocity = np.zeros_like(self.centroids)

    def learn_one(self, x_dict: Dict[str, float]):
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        self._ensure_centroids(len(z))
        assert self.centroids is not None and self.velocity is not None

        sims = self.centroids @ z
        i = int(np.argmax(sims))

        lr = self.learning_rate / (1.0 + 0.0005 * self.n_samples_seen)
        grad = z - self.centroids[i]

        self.velocity[i] = self.momentum * self.velocity[i] + grad
        self.centroids[i] = normalize_vec(self.centroids[i] + lr * self.velocity[i])

        self.n_samples_seen += 1
        return self

    def predict_one(self, x_dict: Dict[str, float]) -> Optional[int]:
        if self.centroids is None:
            return None
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        sims = self.centroids @ z
        return int(np.argmax(sims))


class OSKMWithConfidenceWeighting:
    def __init__(self, k: int, learning_rate: float = 0.05,
                 initial_centroids: Optional[np.ndarray] = None, seed: int = 42,
                 conf_min: float = 0.1):
        self.k = k
        self.learning_rate = learning_rate
        self.initial_centroids = initial_centroids
        self.seed = seed
        self.conf_min = conf_min
        self.centroids: Optional[np.ndarray] = None
        self.n_samples_seen = 0

    def _ensure_centroids(self, dim: int) -> None:
        if self.centroids is not None:
            return
        if self.initial_centroids is not None:
            self.centroids = self.initial_centroids.copy()
        else:
            rng = np.random.default_rng(self.seed)
            c = rng.normal(0.0, 0.1, size=(self.k, dim))
            self.centroids = normalize(c)

    def learn_one(self, x_dict: Dict[str, float]):
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        self._ensure_centroids(len(z))
        assert self.centroids is not None

        sims = self.centroids @ z
        i = int(np.argmax(sims))
        conf = float(sims[i])

        lr = self.learning_rate / (1.0 + 0.0005 * self.n_samples_seen)
        conf_clipped = float(np.clip(conf, self.conf_min, 1.0))
        effective_lr = lr * conf_clipped

        self.centroids[i] = normalize_vec(self.centroids[i] + effective_lr * (z - self.centroids[i]))
        self.n_samples_seen += 1
        return self

    def predict_one(self, x_dict: Dict[str, float]) -> Optional[int]:
        if self.centroids is None:
            return None
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        sims = self.centroids @ z
        return int(np.argmax(sims))


class SAOKM_Fair:
    def __init__(self, k: int, alpha: float, initial_centroids: Optional[np.ndarray] = None, seed: int = 42):
        self.k = k
        self.alpha = alpha
        self.initial_centroids = initial_centroids
        self.seed = seed
        self.centroids: Optional[np.ndarray] = None
        self.n_samples_seen = 0

    def _ensure_centroids(self, dim: int) -> None:
        if self.centroids is not None:
            return
        if self.initial_centroids is not None:
            self.centroids = self.initial_centroids.copy()
        else:
            rng = np.random.default_rng(self.seed)
            c = rng.normal(0.0, 0.1, size=(self.k, dim))
            self.centroids = normalize(c)

    def learn_one(self, x_dict: Dict[str, float]):
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        self._ensure_centroids(len(z))
        assert self.centroids is not None

        s = self.centroids @ z
        i = int(np.argmax(s))
        w = float(self.alpha * (s[i] ** 3))
        self.centroids[i] = normalize_vec(w * z + (1.0 - w) * self.centroids[i])

        self.n_samples_seen += 1
        return self

    def predict_one(self, x_dict: Dict[str, float]) -> Optional[int]:
        if self.centroids is None:
            return None
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        s = self.centroids @ z
        return int(np.argmax(s))


class ConfidenceGatedSAOKM:
    def __init__(self, k: int, learning_rate: float = 0.05, initial_centroids: Optional[np.ndarray] = None, seed: int = 42):
        self.k = k
        self.learning_rate = learning_rate
        self.initial_centroids = initial_centroids
        self.seed = seed
        self.centroids: Optional[np.ndarray] = None
        self.n_samples_seen = 0

    def _ensure_centroids(self, dim: int) -> None:
        if self.centroids is not None:
            return
        if self.initial_centroids is not None:
            self.centroids = self.initial_centroids.copy()
        else:
            rng = np.random.default_rng(self.seed)
            c = rng.normal(0.0, 0.1, size=(self.k, dim))
            self.centroids = normalize(c)

    def learn_one(self, x_dict: Dict[str, float]):
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        self._ensure_centroids(len(z))
        assert self.centroids is not None

        s = self.centroids @ z
        i = int(np.argmax(s))
        confidence = float(s[i])

        lr = self.learning_rate / (1.0 + 0.0005 * self.n_samples_seen)
        if confidence > 0.7:
            effective_lr = lr
        elif confidence > 0.3:
            effective_lr = lr * confidence
        else:
            effective_lr = lr * 0.1

        self.centroids[i] = normalize_vec(self.centroids[i] + effective_lr * (z - self.centroids[i]))
        self.n_samples_seen += 1
        return self

    def predict_one(self, x_dict: Dict[str, float]) -> Optional[int]:
        if self.centroids is None:
            return None
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        s = self.centroids @ z
        return int(np.argmax(s))


class AdaptiveGradientSAOKM:
    def __init__(self, k: int, learning_rate: float = 0.05, initial_centroids: Optional[np.ndarray] = None, seed: int = 42):
        self.k = k
        self.learning_rate = learning_rate
        self.initial_centroids = initial_centroids
        self.seed = seed
        self.centroids: Optional[np.ndarray] = None
        self.n_samples_seen = 0

    def _ensure_centroids(self, dim: int) -> None:
        if self.centroids is not None:
            return
        if self.initial_centroids is not None:
            self.centroids = self.initial_centroids.copy()
        else:
            rng = np.random.default_rng(self.seed)
            c = rng.normal(0.0, 0.1, size=(self.k, dim))
            self.centroids = normalize(c)

    def learn_one(self, x_dict: Dict[str, float]):
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        self._ensure_centroids(len(z))
        assert self.centroids is not None

        s = self.centroids @ z
        i = int(np.argmax(s))

        lr = self.learning_rate / (1.0 + 0.0005 * self.n_samples_seen)
        similarity_boost = float(np.clip(s[i] ** 2, 0.1, 1.0))
        effective_lr = lr * similarity_boost

        self.centroids[i] = normalize_vec(self.centroids[i] + effective_lr * (z - self.centroids[i]))
        self.n_samples_seen += 1
        return self

    def predict_one(self, x_dict: Dict[str, float]) -> Optional[int]:
        if self.centroids is None:
            return None
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        s = self.centroids @ z
        return int(np.argmax(s))


class DualModeSAOKM:
    def __init__(self, k: int, learning_rate: float = 0.05, initial_centroids: Optional[np.ndarray] = None, seed: int = 42):
        self.k = k
        self.learning_rate = learning_rate
        self.initial_centroids = initial_centroids
        self.seed = seed
        self.centroids: Optional[np.ndarray] = None
        self.n_samples_seen = 0
        self.exploration_phase = True
        self.phase_switch_threshold = 800
        self.recent_similarities = deque(maxlen=100)

    def _ensure_centroids(self, dim: int) -> None:
        if self.centroids is not None:
            return
        if self.initial_centroids is not None:
            self.centroids = self.initial_centroids.copy()
        else:
            rng = np.random.default_rng(self.seed)
            c = rng.normal(0.0, 0.1, size=(self.k, dim))
            self.centroids = normalize(c)

    def learn_one(self, x_dict: Dict[str, float]):
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        self._ensure_centroids(len(z))
        assert self.centroids is not None

        s = self.centroids @ z
        i = int(np.argmax(s))
        max_sim = float(s[i])
        self.recent_similarities.append(max_sim)

        if (
            self.n_samples_seen > self.phase_switch_threshold
            and len(self.recent_similarities) > 50
            and float(np.mean(self.recent_similarities)) > 0.6
        ):
            self.exploration_phase = False

        if self.exploration_phase:
            w = self.learning_rate * max_sim
            self.centroids[i] = normalize_vec(w * z + (1.0 - w) * self.centroids[i])
        else:
            lr = self.learning_rate / (1.0 + 0.001 * (self.n_samples_seen - self.phase_switch_threshold))
            self.centroids[i] = normalize_vec(self.centroids[i] + lr * (z - self.centroids[i]))

        self.n_samples_seen += 1
        return self

    def predict_one(self, x_dict: Dict[str, float]) -> Optional[int]:
        if self.centroids is None:
            return None
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        s = self.centroids @ z
        return int(np.argmax(s))


class MomentumHybridSAOKM:
    def __init__(self, k: int, learning_rate: float = 0.05, momentum: float = 0.9,
                 initial_centroids: Optional[np.ndarray] = None, seed: int = 42):
        self.k = k
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.initial_centroids = initial_centroids
        self.seed = seed
        self.centroids: Optional[np.ndarray] = None
        self.momentum_vectors: Optional[np.ndarray] = None
        self.n_samples_seen = 0

    def _ensure_centroids(self, dim: int) -> None:
        if self.centroids is not None:
            return
        if self.initial_centroids is not None:
            self.centroids = self.initial_centroids.copy()
        else:
            rng = np.random.default_rng(self.seed)
            c = rng.normal(0.0, 0.1, size=(self.k, dim))
            self.centroids = normalize(c)
        self.momentum_vectors = np.zeros_like(self.centroids)

    def learn_one(self, x_dict: Dict[str, float]):
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        self._ensure_centroids(len(z))
        assert self.centroids is not None and self.momentum_vectors is not None

        s = self.centroids @ z
        i = int(np.argmax(s))
        confidence = float(s[i])

        lr = self.learning_rate / (1.0 + 0.0005 * self.n_samples_seen)
        gradient = z - self.centroids[i]

        effective_momentum = float(self.momentum * np.clip(confidence, 0.5, 1.0))
        self.momentum_vectors[i] = effective_momentum * self.momentum_vectors[i] + lr * gradient
        self.centroids[i] = normalize_vec(self.centroids[i] + self.momentum_vectors[i])

        self.n_samples_seen += 1
        return self

    def predict_one(self, x_dict: Dict[str, float]) -> Optional[int]:
        if self.centroids is None:
            return None
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        s = self.centroids @ z
        return int(np.argmax(s))


class ConfidenceGradientSAOKM:
    def __init__(self, k: int, learning_rate: float = 0.05,
                 initial_centroids: Optional[np.ndarray] = None, seed: int = 42):
        self.k = k
        self.learning_rate = learning_rate
        self.initial_centroids = initial_centroids
        self.seed = seed
        self.centroids: Optional[np.ndarray] = None
        self.n_samples_seen = 0

    def _ensure_centroids(self, dim: int) -> None:
        if self.centroids is not None:
            return
        if self.initial_centroids is not None:
            self.centroids = self.initial_centroids.copy()
        else:
            rng = np.random.default_rng(self.seed)
            c = rng.normal(0.0, 0.1, size=(self.k, dim))
            self.centroids = normalize(c)

    def learn_one(self, x_dict: Dict[str, float]):
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        self._ensure_centroids(len(z))
        assert self.centroids is not None

        s = self.centroids @ z
        i = int(np.argmax(s))
        confidence = float(s[i])

        base_lr = self.learning_rate / (1.0 + 0.0005 * self.n_samples_seen)
        confidence_factor = float(np.clip(confidence, 0.1, 1.0))
        effective_lr = base_lr * confidence_factor

        self.centroids[i] = normalize_vec(self.centroids[i] + effective_lr * (z - self.centroids[i]))
        self.n_samples_seen += 1
        return self

    def predict_one(self, x_dict: Dict[str, float]) -> Optional[int]:
        if self.centroids is None:
            return None
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        s = self.centroids @ z
        return int(np.argmax(s))






# =========================
# ESA-Stream (L2-faithful baseline with macro-clustering to K)
# =========================
# GridID type for this baseline
GridID = Tuple[int, ...]


class ESAStreamL2MacroBaseline:
    """
    Fair competitor version of ESA-Stream-style grid + exponential decay on this script's
    existing L2-normalized embedding stream, PLUS a lightweight macro-clustering step to output
    exactly K clusters (like other competitors).

    Why this is needed:
      A pure "grid-id = label" approach can create hundreds/thousands of micro-clusters in high-dim space,
      which makes ARI unfairly low and 'k' explode. ESA-Stream's intent is to summarize into grids and
      then cluster/merge them. This class follows that spirit while staying faithful to the L2 pipeline.

    Mechanics:
      - Maintain decayed grid densities and grid centroids on the unit sphere.
      - Periodically (recluster_every) run weighted spherical k-means over active grid centroids to produce
        macro_k centroids (default: global K).
      - Predictions use the macro centroids (cosine argmax), so reported clusters remain ~K.
      - Learning updates grids only; macro step is just a periodic readout.

    Hyperparameters:
      grid_len: quantization step size in feature space
      lam: exponential decay factor for grid density
      prune_min_density: remove grids with very low decayed density
      recluster_every: how often to recompute macro centroids
      max_grids_for_macro: cap number of grids used for macro step (take densest)
      macro_iters: Lloyd iterations for spherical k-means
    """

    def __init__(
        self,
        dim: int,
        macro_k: int,
        grid_len: float = 0.5,
        lam: float = 0.998,
        prune_min_density: float = 1e-3,
        recluster_every: int = 500,
        max_grids_for_macro: int = 2000,
        macro_iters: int = 15,
        seed: int = 42,
    ):
        self.dim = int(dim)
        self.macro_k = int(macro_k)
        self.grid_len = float(grid_len)
        self.lam = float(lam)
        self.prune_min_density = float(prune_min_density)
        self.recluster_every = int(recluster_every)
        self.max_grids_for_macro = int(max_grids_for_macro)
        self.macro_iters = int(macro_iters)
        self.rng = np.random.default_rng(int(seed))

        self._density: Dict[GridID, float] = {}
        self._numerator: Dict[GridID, np.ndarray] = {}
        self._centroid: Dict[GridID, np.ndarray] = {}
        self._last_update: Dict[GridID, int] = {}

        self._macro_centroids: Optional[np.ndarray] = None  # shape (macro_k, d)
        self._t = 0

    def _grid_id(self, z: np.ndarray) -> GridID:
        return tuple(np.floor(z / self.grid_len).astype(int).tolist())

    def _decay_to(self, gid: GridID, t: int) -> None:
        tu = self._last_update[gid]
        if t <= tu:
            return
        dt = t - tu
        factor = self.lam ** dt
        self._density[gid] *= factor
        self._numerator[gid] *= factor
        if self._density[gid] > 0:
            c = self._numerator[gid] / self._density[gid]
            self._centroid[gid] = normalize_vec(c)
        self._last_update[gid] = t

    def prune(self) -> None:
        to_delete = []
        for gid in list(self._density.keys()):
            self._decay_to(gid, self._t)
            if self._density[gid] < self.prune_min_density:
                to_delete.append(gid)
        for gid in to_delete:
            self._density.pop(gid, None)
            self._numerator.pop(gid, None)
            self._centroid.pop(gid, None)
            self._last_update.pop(gid, None)

    def _weighted_spherical_kmeans(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Weighted spherical k-means (cosine) with Lloyd updates.
        X: (m,d) unit vectors, w: (m,) nonnegative weights
        Returns: C (k,d) unit centroids
        """
        m, d = X.shape
        k = min(self.macro_k, m)
        # init: sample proportional to weights (fallback uniform)
        prob = w / w.sum() if w.sum() > 0 else np.full(m, 1.0 / m)
        init_idx = self.rng.choice(m, size=k, replace=False, p=prob)
        C = X[init_idx].copy()

        for _ in range(self.macro_iters):
            # assign by cosine
            sims = X @ C.T  # (m,k)
            a = np.argmax(sims, axis=1)  # (m,)

            # update centroids with weighted mean then normalize
            newC = np.zeros_like(C)
            for j in range(k):
                mask = (a == j)
                if not np.any(mask):
                    # re-seed empty cluster
                    ridx = int(self.rng.choice(m, p=prob))
                    newC[j] = X[ridx]
                    continue
                ww = w[mask][:, None]
                mu = (ww * X[mask]).sum(axis=0)
                newC[j] = normalize_vec(mu)
            C = newC

        # if m < macro_k, pad by duplicating to keep shape (macro_k,d)
        if k < self.macro_k:
            pad = np.vstack([C, C[self.rng.integers(0, k, size=self.macro_k - k)]])
            return pad
        return C

    def _maybe_recluster(self) -> None:
        if self.recluster_every <= 0:
            return
        if (self._t % self.recluster_every) != 0:
            return
        if not self._centroid:
            return

        # collect active grids, decay to now
        gids = list(self._centroid.keys())
        dens = []
        cents = []
        for gid in gids:
            self._decay_to(gid, self._t)
            if self._density[gid] <= 0:
                continue
            dens.append(self._density[gid])
            cents.append(self._centroid[gid])
        if not cents:
            return

        dens = np.array(dens, dtype=float)
        cents = np.vstack(cents)  # (m,d)

        # keep top densest grids if too many
        if len(dens) > self.max_grids_for_macro:
            top = np.argsort(dens)[-self.max_grids_for_macro:]
            dens = dens[top]
            cents = cents[top]

        # ensure unit
        cents = np.array([normalize_vec(v) for v in cents])

        self._macro_centroids = self._weighted_spherical_kmeans(cents, dens)

    def learn_one(self, x_dict: Dict[str, float]):
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        self._t += 1

        gid = self._grid_id(z)
        if gid not in self._density:
            self._density[gid] = 1.0
            self._numerator[gid] = z.copy()
            self._centroid[gid] = z.copy()
            self._last_update[gid] = self._t
        else:
            self._decay_to(gid, self._t)
            self._density[gid] += 1.0
            self._numerator[gid] += z
            c = self._numerator[gid] / self._density[gid]
            self._centroid[gid] = normalize_vec(c)

        if (self._t % 500) == 0:
            self.prune()

        self._maybe_recluster()
        return self

    def predict_one(self, x_dict: Dict[str, float]) -> Optional[int]:
        if self._macro_centroids is None:
            # not enough history yet; return None so STREAM eval ignores early points
            return None
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        sims = self._macro_centroids @ z
        return int(np.argmax(sims))


# =========================
# Evaluation
# =========================
@dataclass
class EvalResult:
    run: int
    seed: int
    mode: str  # STREAM or FINAL
    name: str
    n_clusters: int
    nmi: float
    ari: float
    silhouette: Optional[float]
    total_time_s: float
    learn_time_s: float
    predict_time_s: float


def evaluate_streaming(
    algorithm,
    name: str,
    X: np.ndarray,
    y_true: np.ndarray,
    sample_size: int,
    seed: int,
) -> Tuple[int, float, float, Optional[float], float, float, float]:
    n = X.shape[0]
    sample_size = min(sample_size, n)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=sample_size, replace=False)

    preds: List[Optional[int]] = []
    total_start = time.time()
    learn_time = 0.0
    pred_time = 0.0

    for idx in indices:
        x_dict = dict_from_row(X, int(idx))

        t0 = time.time()
        p = algorithm.predict_one(x_dict)
        t1 = time.time()
        pred_time += (t1 - t0)
        preds.append(p)

        t2 = time.time()
        algorithm.learn_one(x_dict)
        t3 = time.time()
        learn_time += (t3 - t2)

    total_time = time.time() - total_start

    valid = [i for i, p in enumerate(preds) if p is not None]
    if not valid:
        return 0, 0.0, 0.0, None, total_time, learn_time, pred_time

    y_eval = y_true[indices[valid]]
    y_pred = np.array([preds[i] for i in valid], dtype=int)

    n_clusters = int(len(np.unique(y_pred)))
    nmi = float(normalized_mutual_info_score(y_eval, y_pred))
    ari = float(adjusted_rand_score(y_eval, y_pred))

    sil: Optional[float] = None
    if n_clusters >= 2:
        try:
            sil = float(silhouette_score(X[indices[valid]], y_pred))
        except Exception:
            sil = None

    return n_clusters, nmi, ari, sil, total_time, learn_time, pred_time


def evaluate_final_model(
    algorithm,
    name: str,
    X: np.ndarray,
    y_true: np.ndarray,
    train_size: int,
    eval_size: int,
    seed: int,
) -> Tuple[int, float, float, Optional[float], float, float, float]:
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)

    train_size = min(train_size, n)
    train_idx = rng.choice(all_idx, size=train_size, replace=False)

    remaining = np.setdiff1d(all_idx, train_idx, assume_unique=False)
    if len(remaining) >= eval_size:
        eval_idx = rng.choice(remaining, size=eval_size, replace=False)
    else:
        eval_size = min(eval_size, n)
        eval_idx = rng.choice(all_idx, size=eval_size, replace=False)

    total_start = time.time()
    learn_time = 0.0
    pred_time = 0.0

    for idx in train_idx:
        x_dict = dict_from_row(X, int(idx))
        t0 = time.time()
        algorithm.learn_one(x_dict)
        t1 = time.time()
        learn_time += (t1 - t0)

    y_pred: List[int] = []
    for idx in eval_idx:
        x_dict = dict_from_row(X, int(idx))
        t0 = time.time()
        p = algorithm.predict_one(x_dict)
        t1 = time.time()
        pred_time += (t1 - t0)
        y_pred.append(p if p is not None else -1)

    total_time = time.time() - total_start

    y_pred_arr = np.array(y_pred, dtype=int)
    y_eval = y_true[eval_idx]

    valid = y_pred_arr != -1
    y_pred_arr = y_pred_arr[valid]
    y_eval = y_eval[valid]
    eval_X = X[eval_idx][valid]

    if len(y_pred_arr) == 0:
        return 0, 0.0, 0.0, None, total_time, learn_time, pred_time

    n_clusters = int(len(np.unique(y_pred_arr)))
    nmi = float(normalized_mutual_info_score(y_eval, y_pred_arr))
    ari = float(adjusted_rand_score(y_eval, y_pred_arr))

    sil: Optional[float] = None
    if n_clusters >= 2 and len(y_pred_arr) >= 2:
        try:
            sil = float(silhouette_score(eval_X, y_pred_arr))
        except Exception:
            sil = None

    return n_clusters, nmi, ari, sil, total_time, learn_time, pred_time


def make_algorithms(initial_centroids: np.ndarray, seed: int) -> List[Tuple[str, object]]:
    # Keep names identical to original where possible
    return [
        ("0b. OSKM + Adam-on-the-sphere (Baseline)",
         OSKMWithAdamOnSphere(
             k=K,
             learning_rate=LEARNING_RATE,
             initial_centroids=initial_centroids,
             beta1=0.9,
             beta2=0.999,
             eps=1e-8,
             use_time_decay=True,
         )),
        ("Online Spherical K-Means (Baseline)",
         OnlineSphericalKMeans_Fair(k=K, learning_rate=LEARNING_RATE, initial_centroids=initial_centroids, seed=seed)),
        ("ESA-Stream (Grid+Decay → Macro-K) on L2 (Baseline)",
         ESAStreamL2MacroBaseline(dim=initial_centroids.shape[1], macro_k=K, grid_len=0.5, lam=0.998, prune_min_density=1e-3,
                                  recluster_every=500, max_grids_for_macro=2000, macro_iters=15, seed=seed)),
        ("OSKM + Momentum (Ablation)",
         OSKMWithMomentum(k=K, learning_rate=LEARNING_RATE, momentum=MOMENTUM, initial_centroids=initial_centroids, seed=seed)),
        ("OSKM + Confidence Weighting (Ablation)",
         OSKMWithConfidenceWeighting(k=K, learning_rate=LEARNING_RATE, initial_centroids=initial_centroids, seed=seed, conf_min=0.1)),
        ("SAOKM Original (Baseline)",
         SAOKM_Fair(k=K, alpha=0.7, initial_centroids=initial_centroids, seed=seed)),
        # ("Confidence-Gated SAOKM",
        #  ConfidenceGatedSAOKM(k=K, learning_rate=LEARNING_RATE, initial_centroids=initial_centroids, seed=seed)),
        # ("Adaptive Gradient SAOKM",
        #  AdaptiveGradientSAOKM(k=K, learning_rate=LEARNING_RATE, initial_centroids=initial_centroids, seed=seed)),
        # ("Dual-Mode SAOKM",
        #  DualModeSAOKM(k=K, learning_rate=LEARNING_RATE, initial_centroids=initial_centroids, seed=seed)),
        ("Momentum Hybrid SAOKM",
         MomentumHybridSAOKM(k=K, learning_rate=LEARNING_RATE, momentum=MOMENTUM, initial_centroids=initial_centroids, seed=seed)),
        # ("Confidence-Gradient SAOKM",
        #  ConfidenceGradientSAOKM(k=K, learning_rate=LEARNING_RATE, initial_centroids=initial_centroids, seed=seed)),
    ]


def aggregate_results(results: List[EvalResult]) -> List[dict]:
    by_key: Dict[Tuple[str, str], List[EvalResult]] = defaultdict(list)
    for r in results:
        by_key[(r.mode, r.name)].append(r)

    summary_rows: List[dict] = []
    for (mode, name), rows in by_key.items():
        nmi = np.array([x.nmi for x in rows], dtype=float)
        ari = np.array([x.ari for x in rows], dtype=float)
        sil = np.array([x.silhouette if x.silhouette is not None else np.nan for x in rows], dtype=float)
        total_time = np.array([x.total_time_s for x in rows], dtype=float)

        sil_valid = sil[~np.isnan(sil)]
        sil_mean = float(np.mean(sil_valid)) if len(sil_valid) else float("nan")
        sil_std = float(np.std(sil_valid, ddof=1)) if len(sil_valid) > 1 else 0.0

        _, nmi_ci = mean_ci95(nmi)
        _, ari_ci = mean_ci95(ari)

        summary_rows.append({
            "mode": mode,
            "name": name,
            "runs": len(rows),

            "nmi_mean": float(np.mean(nmi)),
            "nmi_std": float(np.std(nmi, ddof=1)) if len(nmi) > 1 else 0.0,
            "nmi_median": float(np.median(nmi)),
            "nmi_min": float(np.min(nmi)),
            "nmi_max": float(np.max(nmi)),
            "nmi_ci95_halfwidth": float(nmi_ci),

            "ari_mean": float(np.mean(ari)),
            "ari_std": float(np.std(ari, ddof=1)) if len(ari) > 1 else 0.0,
            "ari_median": float(np.median(ari)),
            "ari_min": float(np.min(ari)),
            "ari_max": float(np.max(ari)),
            "ari_ci95_halfwidth": float(ari_ci),

            "sil_mean": sil_mean,
            "sil_std": sil_std,
            "sil_median": float(np.nanmedian(sil)) if len(sil_valid) else float("nan"),
            "sil_min": float(np.nanmin(sil)) if len(sil_valid) else float("nan"),
            "sil_max": float(np.nanmax(sil)) if len(sil_valid) else float("nan"),

            "total_time_mean": float(np.mean(total_time)),
            "total_time_std": float(np.std(total_time, ddof=1)) if len(total_time) > 1 else 0.0,
        })

    return summary_rows


def print_summary_table(title: str, summary_rows: List[dict], sort_key: str = "nmi_mean") -> None:
    rows = sorted(summary_rows, key=lambda r: (r.get(sort_key, float("-inf"))), reverse=True)
    print("\n" + "=" * 130)
    print(title)
    print("=" * 130)
    header = (
        "Algorithm                              | Runs | NMI mean±std     | ARI mean±std     | Sil mean±std     | "
        "NMI CI95(±) | ARI CI95(±) | Time mean(s)"
    )
    print(header)
    print("-" * 130)
    for r in rows:
        name = r["name"][:37]
        runs = r["runs"]
        nmi_m, nmi_s = r["nmi_mean"], r["nmi_std"]
        ari_m, ari_s = r["ari_mean"], r["ari_std"]
        sil_m, sil_s = r["sil_mean"], r["sil_std"]
        nmi_ci = r["nmi_ci95_halfwidth"]
        ari_ci = r["ari_ci95_halfwidth"]
        t_m = r["total_time_mean"]
        print(
            f"{name:37s} | {runs:4d} | {nmi_m:6.3f}±{nmi_s:6.3f} | {ari_m:6.3f}±{ari_s:6.3f} | "
            f"{sil_m:6.3f}±{sil_s:6.3f} | {nmi_ci:9.3f} | {ari_ci:9.3f} | {t_m:10.2f}"
        )


def write_csv_long(path: str, results: List[EvalResult]) -> None:
    fields = [
        "run", "seed", "mode", "name",
        "n_clusters", "nmi", "ari", "silhouette",
        "total_time_s", "learn_time_s", "predict_time_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "run": r.run,
                "seed": r.seed,
                "mode": r.mode,
                "name": r.name,
                "n_clusters": r.n_clusters,
                "nmi": r.nmi,
                "ari": r.ari,
                "silhouette": ("" if r.silhouette is None else r.silhouette),
                "total_time_s": r.total_time_s,
                "learn_time_s": r.learn_time_s,
                "predict_time_s": r.predict_time_s,
            })


def write_csv_summary(path: str, summary_rows: List[dict]) -> None:
    fields = [
        "mode", "name", "runs",
        "nmi_mean", "nmi_std", "nmi_median", "nmi_min", "nmi_max", "nmi_ci95_halfwidth",
        "ari_mean", "ari_std", "ari_median", "ari_min", "ari_max", "ari_ci95_halfwidth",
        "sil_mean", "sil_std", "sil_median", "sil_min", "sil_max",
        "total_time_mean", "total_time_std",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(summary_rows, key=lambda d: (d["mode"], d["name"])):
            w.writerow({k: r.get(k, "") for k in fields})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=10, help="Number of independent runs.")
    p.add_argument("--seed", type=int, default=SEED, help="Base seed; run seed = seed + run_id.")
    p.add_argument("--subset", type=str, default=SUBSET, choices=["train", "test", "all"], help="20NG subset.")
    p.add_argument("--sample_size", type=int, default=SAMPLE_SIZE, help="Streaming sample size.")
    p.add_argument("--train_size", type=int, default=TRAIN_SIZE, help="Final-model train size.")
    p.add_argument("--eval_size", type=int, default=EVAL_SIZE, help="Final-model eval size.")
    p.add_argument("--svd_components", type=int, default=SVD_COMPONENTS, help="SVD components (0 disables).")
    p.add_argument("--svd_seed_per_run", action="store_true", help="Use per-run seed for TruncatedSVD random_state (slower).")
    p.add_argument("--single_thread", action="store_true", help="Set OMP/MKL/OPENBLAS threads to 1 (best-effort).")
    p.add_argument("--output_dir", type=str, default=".", help="Directory to write CSV outputs.")
    p.add_argument("--quiet", action="store_true", help="Less printing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.single_thread:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    set_global_seed(args.seed)

    if not args.quiet:
        print("=" * 110)
        print("HYBRID OSKM-SAOKM ALGORITHMS COMPARISON (20 NEWGROUPS) — MULTI-RUN")
        print("=" * 110)
        print(f"Runs: {args.runs} | Base seed: {args.seed} | Subset: {args.subset}")
        print(f"Stream sample: {args.sample_size} | Final train/eval: {args.train_size}/{args.eval_size}")
        print(f"SVD components: {args.svd_components} | svd_seed_per_run: {bool(args.svd_seed_per_run)} | single_thread: {bool(args.single_thread)}")
        print("=" * 110)

    # -------- Load dataset once --------
    if not args.quiet:
        print("Loading 20 Newsgroups dataset...")
    subset_arg = args.subset if args.subset != "all" else None
    newsgroups = fetch_20newsgroups(subset=subset_arg, remove=REMOVE_META)
    y_true = np.array(newsgroups.target, dtype=int)
    if not args.quiet:
        print(f"Dataset loaded: {len(newsgroups.data)} documents, {len(newsgroups.target_names)} categories")

    # -------- Preprocess once --------
    if not args.quiet:
        print("\nPreprocessing: TF-IDF -> SVD -> L2 normalize")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        stop_words=TFIDF_STOP_WORDS,
        max_df=TFIDF_MAX_DF,
        min_df=TFIDF_MIN_DF,
        ngram_range=TFIDF_NGRAM_RANGE,
    )
    X_tfidf = vectorizer.fit_transform(newsgroups.data)
    if not args.quiet:
        print(f"TF-IDF shape: {X_tfidf.shape}")

    # Default: fixed representation (one SVD fit) for fairness across runs
    if args.svd_components and args.svd_components > 0:
        svd = TruncatedSVD(n_components=args.svd_components, random_state=args.seed)
        X_base = svd.fit_transform(X_tfidf)
        if not args.quiet:
            print(f"SVD shape: {X_base.shape}")
            try:
                explained = float(svd.explained_variance_ratio_.sum())
                print(f"Variance explained: {explained:.3f}")
            except Exception:
                pass
    else:
        X_base = X_tfidf.toarray()
        if not args.quiet:
            print(f"Dense shape (no SVD): {X_base.shape}")

    X_base = normalize(X_base)

    # -------- Multi-run loop --------
    all_results: List[EvalResult] = []

    for run_id in range(1, args.runs + 1):
        run_seed = int(args.seed + run_id)

        # Optional: representation randomness per run (slower)
        if args.svd_seed_per_run and args.svd_components and args.svd_components > 0:
            svd = TruncatedSVD(n_components=args.svd_components, random_state=run_seed)
            X = normalize(svd.fit_transform(X_tfidf))
        else:
            X = X_base

        initial_centroids = get_initial_centroids(k=K, n_features=X.shape[1], seed=run_seed)

        if not args.quiet:
            print("\n" + "-" * 110)
            print(f"RUN {run_id}/{args.runs} (seed={run_seed})")
            print("-" * 110)

        # STREAM
        algos_stream = make_algorithms(initial_centroids, seed=run_seed)
        for name, algo in algos_stream:
            n_clusters, nmi, ari, sil, total_t, learn_t, pred_t = evaluate_streaming(
                algo, name, X, y_true, sample_size=args.sample_size, seed=run_seed
            )
            all_results.append(EvalResult(
                run=run_id, seed=run_seed, mode="STREAM", name=name,
                n_clusters=n_clusters, nmi=nmi, ari=ari, silhouette=sil,
                total_time_s=total_t, learn_time_s=learn_t, predict_time_s=pred_t
            ))
            if not args.quiet:
                print(f"[RUN {run_id}] STREAM finished: {name} | k={n_clusters} | NMI={nmi:.4f} | ARI={ari:.4f} | Sil={'N/A' if sil is None else f'{sil:.4f}'} | total={total_t:.2f}s (learn={learn_t:.2f}s, pred={pred_t:.2f}s)")
            if not args.quiet:
                sil_str = "N/A" if sil is None else f"{sil:.3f}"
                print(f"  [STREAM] {name:35s} | NMI {nmi:.3f} | ARI {ari:.3f} | Sil {sil_str}")

        # FINAL (fresh instances)
        algo_map = dict(make_algorithms(initial_centroids, seed=run_seed))
        for name, _ in algos_stream:
            fresh = algo_map[name]
            n_clusters, nmi, ari, sil, total_t, learn_t, pred_t = evaluate_final_model(
                fresh, name, X, y_true, train_size=args.train_size, eval_size=args.eval_size, seed=run_seed
            )
            all_results.append(EvalResult(
                run=run_id, seed=run_seed, mode="FINAL", name=name,
                n_clusters=n_clusters, nmi=nmi, ari=ari, silhouette=sil,
                total_time_s=total_t, learn_time_s=learn_t, predict_time_s=pred_t
            ))
            if not args.quiet:
                print(f"[RUN {run_id}] FINAL  finished: {name} | k={n_clusters} | NMI={nmi:.4f} | ARI={ari:.4f} | Sil={'N/A' if sil is None else f'{sil:.4f}'} | total={total_t:.2f}s (learn={learn_t:.2f}s, pred={pred_t:.2f}s)")
            if not args.quiet:
                sil_str = "N/A" if sil is None else f"{sil:.3f}"
                print(f"  [FINAL ] {name:35s} | NMI {nmi:.3f} | ARI {ari:.3f} | Sil {sil_str}")

    # -------- Aggregate & print --------
    summary_rows = aggregate_results(all_results)
    stream_summary = [r for r in summary_rows if r["mode"] == "STREAM"]
    final_summary = [r for r in summary_rows if r["mode"] == "FINAL"]

    print_summary_table("20 NEWSGROUPS — STREAMING (predict-before-learn) — AGGREGATED OVER RUNS", stream_summary, sort_key="nmi_mean")
    print_summary_table("20 NEWSGROUPS — FINAL-MODEL (train then assign) — AGGREGATED OVER RUNS", final_summary, sort_key="nmi_mean")

    # -------- Write outputs --------
    os.makedirs(args.output_dir, exist_ok=True)
    long_path = os.path.join(args.output_dir, "results_long.csv")
    summary_path = os.path.join(args.output_dir, "results_summary.csv")

    write_csv_long(long_path, all_results)
    write_csv_summary(summary_path, summary_rows)

    print("\nWrote:")
    print(f"  - {long_path}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()
