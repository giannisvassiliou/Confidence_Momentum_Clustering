
#--data-root c:\imagenet-10 --max-classes 10   --images-per-class 800   --resize 224   --sample-size 5000  --train-size 3000   --eval-size 5000   --noise-sigma 0.12  --feature-dropout 0.2  --balance-lambda 0.15 --runs 2 --wilcoxon
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Full, clean, syntax-safe script:
# - ImageNet-style subset loader (root/<class>/*.jpg)
# - ResNet-18 feature extraction + caching
# - k-means++ initialization on extracted (L2-normalized) features
# - Balanced assignment penalty to prevent collapse (unsupervised)
# - Optional feature perturbations: Gaussian noise + feature dropout (to stress robustness)
# - Streaming + Final-model evaluation with NMI/ARI (+ optional silhouette)
#
# Windows path tip: use forward slashes like D:/vision_datasets/imagenet_subset

from __future__ import annotations

import argparse
import hashlib
import shutil
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize


# =========================
# Defaults
# =========================
SEED = 42
LEARNING_RATE = 0.1
ALPHA = 0.08
MOMENTUM = 0.9

SAMPLE_SIZE = 5000
TRAIN_SIZE = 10000
EVAL_SIZE = 5000

RESNET_IMAGE_SIZE = 224


# =========================
# Utilities
# =========================
def set_global_seed(seed: int) -> None:
    np.random.seed(seed)


def normalize_vec(z: np.ndarray) -> np.ndarray:
    return normalize(z.reshape(1, -1))[0]


def balanced_argmax(sims: np.ndarray, counts: np.ndarray, balance_lambda: float) -> int:
    # score_j = sim_j - λ log(count_j + 1)
    if balance_lambda <= 0.0:
        return int(np.argmax(sims))
    scores = sims - float(balance_lambda) * np.log(counts + 1.0)
    return int(np.argmax(scores))


def dict_from_row(X: np.ndarray, idx: int) -> Dict[str, float]:
    row = X[idx]
    return {f"f{j}": float(row[j]) for j in range(row.shape[0])}


def apply_feature_perturbations(
    X: np.ndarray,
    seed: int,
    noise_sigma: float = 0.0,
    dropout_p: float = 0.0,
) -> np.ndarray:
    # Controlled perturbations on already-normalized features.
    # - Gaussian: X <- normalize(X + sigma * N(0,1))
    # - Dropout: randomly zero a fraction of dims per sample, then renormalize
    if noise_sigma <= 0.0 and dropout_p <= 0.0:
        return X

    rng = np.random.default_rng(seed)
    Xp = X.astype(np.float32, copy=True)

    if noise_sigma > 0.0:
        Xp = Xp + float(noise_sigma) * rng.standard_normal(size=Xp.shape).astype(np.float32)

    if dropout_p > 0.0:
        p = float(dropout_p)
        if not (0.0 <= p < 1.0):
            raise ValueError("--feature-dropout must be in [0,1).")
        mask = (rng.random(size=Xp.shape) >= p).astype(np.float32)
        Xp *= mask

    norms = np.linalg.norm(Xp, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return Xp / norms


def get_initial_centroids_kmeanspp(
    X: np.ndarray,
    k: int,
    seed: int,
    pool_size: int = 50000,
) -> np.ndarray:
    # k-means++ on L2-normalized features (spherical-friendly).
    # For unit vectors, ||x-c||^2 = 2 - 2*(x·c).
    assert X.ndim == 2 and X.shape[0] >= k, "Need X with at least k rows"
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    if pool_size and 0 < pool_size < n:
        pool_idx = rng.choice(n, size=int(pool_size), replace=False)
        P = X[pool_idx]
    else:
        P = X

    m = P.shape[0]
    first = int(rng.integers(low=0, high=m))
    centroids = [P[first]]

    sims = P @ centroids[0]
    D2 = 2.0 - 2.0 * sims
    D2 = np.clip(D2, 0.0, None)

    for _ in range(1, k):
        total = float(D2.sum())
        if not np.isfinite(total) or total <= 1e-12:
            idx = int(rng.integers(low=0, high=m))
        else:
            probs = D2 / total
            idx = int(rng.choice(m, p=probs))
        c = P[idx]
        centroids.append(c)

        sims_new = P @ c
        d2_new = 2.0 - 2.0 * sims_new
        d2_new = np.clip(d2_new, 0.0, None)
        D2 = np.minimum(D2, d2_new)

    return normalize(np.stack(centroids, axis=0))


def make_feature_cache_path(
    cache_dir: Path,
    data_root: str,
    max_classes: int,
    images_per_class: int,
    max_images: int,
    seed: int,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = (
        f"imagenet_subset|root={Path(data_root).resolve()}|classes={max_classes}"
        f"|ipc={images_per_class}|max={max_images}|seed={seed}|feat=resnet18"
    )
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"imagenet_subset_resnet18_c{max_classes}_ipc{images_per_class}_max{max_images}_seed{seed}_{h}.npz"


def list_image_files(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _is_imagenet_folder(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if not subdirs:
        return False
    for sd in subdirs[:20]:
        for fp in sd.iterdir():
            if fp.is_file() and fp.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                return True
    return False


def _pick_extracted_root(extract_dir: Path) -> Path:
    children = [p for p in extract_dir.iterdir()]
    dirs = [p for p in children if p.is_dir()]
    if len(dirs) == 1 and _is_imagenet_folder(dirs[0]):
        return dirs[0]
    return extract_dir


def download_and_extract_subset(
    url: str,
    dataset_cache_dir: Path,
    dataset_name: str,
    force: bool = False,
) -> Path:
    # Download+extract a public subset archive to dataset_cache_dir/dataset_name.
    try:
        import requests
        from tqdm import tqdm
    except Exception as e:
        print(e)

    dataset_cache_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = dataset_cache_dir / dataset_name
    marker = extract_dir / ".EXTRACTED_OK"

    if extract_dir.exists() and marker.exists() and not force:
        print(f"[download] Using existing extracted dataset: {extract_dir}")
        return _pick_extracted_root(extract_dir)

    if force and extract_dir.exists():
        print(f"[download] Removing existing dataset because --force-download was set: {extract_dir}")
        shutil.rmtree(extract_dir, ignore_errors=True)

    extract_dir.mkdir(parents=True, exist_ok=True)

    filename = url.split("?")[0].split("/")[-1] or "dataset_archive"
    archive_path = dataset_cache_dir / filename

    print(f"[download] Downloading: {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(archive_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    print(f"[download] Extracting: {archive_path} -> {extract_dir}")
    suffix = archive_path.suffix.lower()
    name_lower = archive_path.name.lower()

    if suffix == ".zip":
        import zipfile
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    elif suffix == ".tar" or name_lower.endswith(".tar.gz") or name_lower.endswith(".tgz"):
        import tarfile
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_dir)
    else:
        raise RuntimeError(f"Unsupported archive type: {archive_path.name} (expected .zip, .tar, .tar.gz, .tgz)")

    marker.write_text("ok", encoding="utf-8")
    chosen = _pick_extracted_root(extract_dir)
    if not _is_imagenet_folder(chosen):
        print("[download] WARNING: Extracted data root does not look like ImageNet folder structure.")
        print("Expected: <root>/<class_folder>/*.jpg")
    else:
        print(f"[download] Dataset ready at: {chosen}")
    return chosen


# =========================
# Algorithms
# =========================
class OnlineSphericalKMeans_Fair:
    def __init__(self, k: int, learning_rate: float, initial_centroids: np.ndarray, balance_lambda: float):
        self.k = k
        self.learning_rate = learning_rate
        self.centroids = initial_centroids.copy()
        self.n_samples_seen = 0
        self.balance_lambda = float(balance_lambda)
        self.counts = np.zeros(self.k, dtype=np.int64)

    def learn_one(self, x_dict: Dict[str, float]):
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        sims = self.centroids @ z
        i = balanced_argmax(sims, self.counts, self.balance_lambda)
        self.counts[i] += 1

        lr = self.learning_rate / (1.0 + 0.0005 * self.n_samples_seen)
        self.centroids[i] = normalize_vec(self.centroids[i] + lr * (z - self.centroids[i]))
        self.n_samples_seen += 1
        return self

    def predict_one(self, x_dict: Dict[str, float]) -> int:
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        sims = self.centroids @ z
        return balanced_argmax(sims, self.counts, self.balance_lambda)


class SAOKM_Fair_org:
    def __init__(self, k: int, alpha: float, initial_centroids: np.ndarray, balance_lambda: float):
        self.k = k
        self.alpha = alpha
        self.centroids = initial_centroids.copy()
        self.balance_lambda = float(balance_lambda)
        self.counts = np.zeros(self.k, dtype=np.int64)

    def learn_one(self, x_dict: Dict[str, float]):
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        s = self.centroids @ z
        i = balanced_argmax(s, self.counts, self.balance_lambda)
        self.counts[i] += 1

        w = float(self.alpha * (float(s[i]) ** 3))
        self.centroids[i] = normalize_vec(w * z + (1.0 - w) * self.centroids[i])
        return self

    def predict_one(self, x_dict: Dict[str, float]) -> int:
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        s = self.centroids @ z
        return balanced_argmax(s, self.counts, self.balance_lambda)


class OSKMWithMomentum:
    def __init__(self, k: int, learning_rate: float, momentum: float, initial_centroids: np.ndarray, balance_lambda: float):
        self.k = k
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.centroids = initial_centroids.copy()
        self.velocity = np.zeros_like(self.centroids)
        self.n_samples_seen = 0
        self.balance_lambda = float(balance_lambda)
        self.counts = np.zeros(self.k, dtype=np.int64)

    def learn_one(self, x_dict: Dict[str, float]):
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        sims = self.centroids @ z
        i = balanced_argmax(sims, self.counts, self.balance_lambda)
        self.counts[i] += 1

        lr = self.learning_rate / (1.0 + 0.0005 * self.n_samples_seen)
        grad = z - self.centroids[i]
        self.velocity[i] = self.momentum * self.velocity[i] + lr * grad
        self.centroids[i] = normalize_vec(self.centroids[i] + self.velocity[i])
        self.n_samples_seen += 1
        return self

    def predict_one(self, x_dict: Dict[str, float]) -> int:
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        sims = self.centroids @ z
        return balanced_argmax(sims, self.counts, self.balance_lambda)


class OSKMWithConfidenceWeighting:
    def __init__(self, k: int, learning_rate: float, initial_centroids: np.ndarray, balance_lambda: float):
        self.k = k
        self.learning_rate = learning_rate
        self.centroids = initial_centroids.copy()
        self.n_samples_seen = 0
        self.balance_lambda = float(balance_lambda)
        self.counts = np.zeros(self.k, dtype=np.int64)

    def learn_one(self, x_dict: Dict[str, float]):
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        sims = self.centroids @ z
        i = balanced_argmax(sims, self.counts, self.balance_lambda)
        self.counts[i] += 1

        conf = float(sims[i])
        lr = self.learning_rate / (1.0 + 0.0005 * self.n_samples_seen)
        conf_clipped = float(np.clip(conf, 0.5, 1.0))
        effective_lr = lr * conf_clipped
        self.centroids[i] = normalize_vec(self.centroids[i] + effective_lr * (z - self.centroids[i]))
        self.n_samples_seen += 1
        return self

    def predict_one(self, x_dict: Dict[str, float]) -> int:
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        sims = self.centroids @ z
        return balanced_argmax(sims, self.counts, self.balance_lambda)







class MomentumHybridSAOKM:
    # This is the "Hybrid SAOKM" (momentum + confidence-scaled updates)
    def __init__(self, k: int, learning_rate: float, momentum: float, initial_centroids: np.ndarray, balance_lambda: float):
        self.k = k
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.centroids = initial_centroids.copy()
        self.momentum_vectors = np.zeros_like(self.centroids)
        self.n_samples_seen = 0
        self.balance_lambda = float(balance_lambda)
        self.counts = np.zeros(self.k, dtype=np.int64)

    def learn_one(self, x_dict: Dict[str, float]):
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        sims = self.centroids @ z
        i = balanced_argmax(sims, self.counts, self.balance_lambda)
        self.counts[i] += 1

        conf = float(sims[i])
        lr = self.learning_rate / (1.0 + 0.0005 * self.n_samples_seen)
        gradient = z - self.centroids[i]
        effective_momentum = float(self.momentum * np.clip(conf, 0.5, 1.0))
        self.momentum_vectors[i] = effective_momentum * self.momentum_vectors[i] + lr * gradient
        self.centroids[i] = normalize_vec(self.centroids[i] + self.momentum_vectors[i])
        self.n_samples_seen += 1
        return self

    def predict_one(self, x_dict: Dict[str, float]) -> int:
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        sims = self.centroids @ z
        return balanced_argmax(sims, self.counts, self.balance_lambda)






class OSKMWithAdamOnSphereBaseline:
    """
    Adam-on-the-sphere baseline (NO confidence weighting), with optional balanced assignment.

    - Assignment uses: score_j = sim_j - λ log(count_j + 1)
      to discourage cluster collapse (same spirit as the other Fair baselines).
    - Update uses Adam on the sphere-projected ascent direction and then renormalizes.
    """
    def __init__(
        self,
        k: int,
        learning_rate: float,
        initial_centroids: np.ndarray,
        balance_lambda: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.k = k
        self.learning_rate = learning_rate
        self.centroids = initial_centroids.copy()

        self.balance_lambda = float(balance_lambda)
        self.counts = np.zeros(self.k, dtype=np.int64)

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = np.zeros_like(self.centroids)
        self.v = np.zeros_like(self.centroids)
        self.t = np.zeros((k,), dtype=np.int64)
        self.n_samples_seen = 0

    def learn_one(self, x_dict):
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        sims = self.centroids @ z
        i = balanced_argmax(sims, self.counts, self.balance_lambda)
        self.counts[i] += 1

        mu = self.centroids[i]
        dot = float(np.dot(mu, z))
        g = z - dot * mu  # tangent (sphere-projected) ascent direction

        lr = self.learning_rate / (1.0 + 0.0005 * self.n_samples_seen)

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

    def predict_one(self, x_dict):
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        sims = self.centroids @ z
        return balanced_argmax(sims, self.counts, self.balance_lambda)



# =========================
# ESA-Stream (L2-faithful baseline with macro-clustering to K)
# =========================
# GridID type for this baseline
GridID = Tuple[int, ...]


class ESAStreamL2MacroBaseline:
    """
    Fair ESA-Stream-style competitor for this script's L2-normalized ResNet features.

    - Learns a grid-based microstructure with exponential decay (densities + centroids).
    - Periodically performs weighted spherical k-means over active grid centroids to produce
      exactly K macro-centroids (so cluster count is comparable to other methods).
    - Uses the same balanced assignment penalty (score = sim - λ log(count+1)) to avoid collapse.
    - predict_one NEVER returns None (this script expects int predictions).

    This is "fair" vs K-way methods: it outputs ~K clusters and uses the same L2/cosine geometry.
    """

    def __init__(
        self,
        dim: int,
        macro_k: int,
        balance_lambda: float = 0.0,
        grid_len: float = 0.5,
        lam: float = 0.998,
        prune_min_density: float = 1e-3,
        recluster_every: int = 500,
        max_grids_for_macro: int = 2000,
        macro_iters: int = 15,
        seed: int = 0,
    ):
        self.dim = int(dim)
        self.macro_k = int(macro_k)
        self.balance_lambda = float(balance_lambda)

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

        self._macro_centroids: Optional[np.ndarray] = None  # (macro_k, d)
        self._t = 0

        # Balanced assignment counts on macro clusters (same idea as other baselines here)
        self.counts = np.zeros(self.macro_k, dtype=np.int64)

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
        Weighted spherical k-means with Lloyd updates.
        X: (m,d) unit vectors, w: (m,) nonnegative weights
        Returns: C (macro_k,d) unit centroids
        """
        m, d = X.shape
        k = min(self.macro_k, m)
        prob = w / w.sum() if w.sum() > 0 else np.full(m, 1.0 / m)

        init_idx = self.rng.choice(m, size=k, replace=False, p=prob)
        C = X[init_idx].copy()

        for _ in range(self.macro_iters):
            sims = X @ C.T  # (m,k)
            a = np.argmax(sims, axis=1)

            newC = np.zeros_like(C)
            for j in range(k):
                mask = (a == j)
                if not np.any(mask):
                    ridx = int(self.rng.choice(m, p=prob))
                    newC[j] = X[ridx]
                    continue
                ww = w[mask][:, None]
                mu = (ww * X[mask]).sum(axis=0)
                newC[j] = normalize_vec(mu)
            C = newC

        if k < self.macro_k:
            pad = np.vstack([C, C[self.rng.integers(0, k, size=self.macro_k - k)]])
            return pad
        return C

    def _recluster_now(self) -> None:
        if not self._centroid:
            return

        gids = list(self._centroid.keys())
        dens = []
        cents = []
        for gid in gids:
            self._decay_to(gid, self._t)
            if self._density.get(gid, 0.0) <= 0:
                continue
            dens.append(self._density[gid])
            cents.append(self._centroid[gid])

        if not cents:
            return

        dens = np.array(dens, dtype=float)
        cents = np.vstack(cents)

        if len(dens) > self.max_grids_for_macro:
            top = np.argsort(dens)[-self.max_grids_for_macro:]
            dens = dens[top]
            cents = cents[top]

        cents = np.array([normalize_vec(v) for v in cents])
        self._macro_centroids = self._weighted_spherical_kmeans(cents, dens)

        # If macro_k changed (shouldn't), reinit counts safely
        if self.counts.shape[0] != self.macro_k:
            self.counts = np.zeros(self.macro_k, dtype=np.int64)

    def _maybe_recluster(self) -> None:
        if self.recluster_every <= 0:
            return
        if (self._t % self.recluster_every) != 0:
            return
        self._recluster_now()

    def _macro_assign(self, z: np.ndarray) -> int:
        sims = self._macro_centroids @ z
        return balanced_argmax(sims, self.counts, self.balance_lambda)

    def learn_one(self, x_dict: Dict[str, float]):
        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        self._t += 1

        # update grid
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

        # periodic cleanup + recluster
        if (self._t % 500) == 0:
            self.prune()
        self._maybe_recluster()

        # update macro counts for balanced assignment (only when macro model exists)
        if self._macro_centroids is not None:
            j = self._macro_assign(z)
            self.counts[j] += 1

        return self

    def predict_one(self, x_dict: Dict[str, float]) -> int:
        # Never return None: build macro centroids ASAP once we have some grids.
        if self._macro_centroids is None:
            if self._centroid and len(self._centroid) >= max(2, min(self.macro_k, 10)):
                self._recluster_now()
            if self._macro_centroids is None:
                return 0

        z = normalize_vec(np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=float))
        j = self._macro_assign(z)
        return int(j)


# =========================
# Evaluation
# =========================
@dataclass
class EvalResult:
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
    compute_silhouette: bool,
    print_every: int = 500,
) -> EvalResult:
    n = X.shape[0]
    sample_size = min(sample_size, n)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=sample_size, replace=False)

    preds: List[int] = []
    total_start = time.time()
    learn_time = 0.0
    pred_time = 0.0

    print(f"\n--- {name} [STREAM] ---")
    for i, idx in enumerate(indices, start=1):
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

        if print_every and i % print_every == 0:
            n_clusters_so_far = len(set(preds))
            print(f"  Processed {i:5d} | Clusters: {n_clusters_so_far:3d} | Time: {time.time()-total_start:.2f}s")

    total_time = time.time() - total_start

    y_eval = y_true[indices]
    y_pred = np.array(preds, dtype=int)

    n_clusters = int(len(np.unique(y_pred)))
    nmi = float(normalized_mutual_info_score(y_eval, y_pred))
    ari = float(adjusted_rand_score(y_eval, y_pred))

    sil: Optional[float] = None
    if compute_silhouette and n_clusters >= 2:
        try:
            sil = float(silhouette_score(X[indices], y_pred))
        except Exception:
            sil = None

    sil_str = f"{sil:.3f}" if sil is not None else "N/A"
    print(f"  Results: Clusters={n_clusters}, NMI={nmi:.3f}, ARI={ari:.3f}, Sil={sil_str}")
    print(f"  Timing: Total={total_time:.2f}s, Samples/sec={sample_size/total_time:.1f}")
    print(f"  Per-sample: Learn={(learn_time/sample_size)*1000:.2f}ms, Predict={(pred_time/sample_size)*1000:.2f}ms")

    return EvalResult(name, n_clusters, nmi, ari, sil, total_time, learn_time, pred_time)


def evaluate_final_model(
    algorithm,
    name: str,
    X: np.ndarray,
    y_true: np.ndarray,
    train_size: int,
    eval_size: int,
    seed: int,
    compute_silhouette: bool,
) -> EvalResult:
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)

    # Ensure a non-empty eval split
    eval_size = int(min(eval_size, n))
    if eval_size <= 0:
        eval_size = 1
    if train_size + eval_size > n:
        train_size = max(1, n - eval_size)
    train_size = min(train_size, n)

    train_idx = rng.choice(all_idx, size=train_size, replace=False)
    remaining = np.setdiff1d(all_idx, train_idx, assume_unique=False)

    if remaining.shape[0] == 0:
        eval_idx = rng.choice(train_idx, size=min(eval_size, train_idx.shape[0]), replace=False)
    else:
        eval_size = min(eval_size, remaining.shape[0])
        eval_idx = rng.choice(remaining, size=eval_size, replace=False)

    total_start = time.time()
    learn_time = 0.0
    pred_time = 0.0

    print(f"\n--- {name} [FINAL] ---")
    for i, idx in enumerate(train_idx, start=1):
        x_dict = dict_from_row(X, int(idx))
        t0 = time.time()
        algorithm.learn_one(x_dict)
        t1 = time.time()
        learn_time += (t1 - t0)
        if i % 1000 == 0:
            print(f"  Trained on {i}/{train_size} samples...")

    preds: List[int] = []
    t_pred0 = time.time()
    for idx in eval_idx:
        x_dict = dict_from_row(X, int(idx))
        preds.append(algorithm.predict_one(x_dict))
    t_pred1 = time.time()
    pred_time += (t_pred1 - t_pred0)

    total_time = time.time() - total_start

    y_eval = y_true[eval_idx]
    y_pred = np.array(preds, dtype=int)

    if y_pred.size == 0:
        n_clusters = 0
        nmi = 0.0
        ari = 0.0
    else:
        n_clusters = int(len(np.unique(y_pred)))
        nmi = float(normalized_mutual_info_score(y_eval, y_pred))
        ari = float(adjusted_rand_score(y_eval, y_pred))

    sil: Optional[float] = None
    if compute_silhouette and n_clusters >= 2:
        try:
            sil = float(silhouette_score(X[eval_idx], y_pred))
        except Exception:
            sil = None

    sil_str = f"{sil:.3f}" if sil is not None else "N/A"
    print(f"  Final Results: Clusters={n_clusters}, NMI={nmi:.3f}, ARI={ari:.3f}, Sil={sil_str}")
    print(f"  Timing: Total={total_time:.2f}s")
    print(f"  Per-sample: Train(Learn)={(learn_time/max(1,train_size))*1000:.2f}ms, Eval(Predict)={(pred_time/max(1,len(eval_idx)))*1000:.2f}ms")

    return EvalResult(name, n_clusters, nmi, ari, sil, total_time, learn_time, pred_time)


def print_ranking(title: str, rows: Sequence[EvalResult]) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    print("Algorithm                              | Clusters | NMI   | ARI   | Silhouette | Total(s) | Learn(s) | Predict(s)")
    print("-" * 120)
    for r in rows:
        sil_str = f"{r.silhouette:.3f}" if r.silhouette is not None else "N/A"
        print(
            f"{r.name:37s} | {r.n_clusters:8d} | {r.nmi:5.3f} | {r.ari:5.3f} | {sil_str:>10s} | "
            f"{r.total_time_s:8.2f} | {r.learn_time_s:8.2f} | {r.predict_time_s:9.2f}"
        )


# =========================
# Data loading + features
# =========================
def load_imagenet_subset_as_numpy(
    data_root: str,
    seed: int,
    max_classes: int,
    images_per_class: int,
    max_images: int,
    resize: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    from PIL import Image

    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"--data-root not found: {root}")

    all_class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not all_class_dirs:
        raise RuntimeError(f"No class folders found under: {root}")

    class_dirs = all_class_dirs[:max_classes] if max_classes > 0 else all_class_dirs
    class_names = [p.name for p in class_dirs]
    k = len(class_names)
    print(f"Found {len(all_class_dirs)} class folders; using K={k} classes.")

    rng = np.random.default_rng(seed)
    images_list: List[np.ndarray] = []
    labels_list: List[int] = []

    t0 = time.time()
    for ci, cdir in enumerate(class_dirs):
        files = list_image_files(cdir)
        if not files:
            print(f"  WARNING: no images in {cdir.name}, skipping.")
            continue

        if images_per_class > 0 and len(files) > images_per_class:
            pick = rng.choice(len(files), size=images_per_class, replace=False)
            files = [files[i] for i in pick]

        for fp in files:
            try:
                img = Image.open(fp).convert("RGB")
                if resize and resize > 0:
                    img = img.resize((resize, resize))
                images_list.append(np.array(img, dtype=np.uint8))
                labels_list.append(ci)
            except Exception:
                continue

        if (ci + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  Loaded class {ci+1}/{k} | images so far: {len(images_list)} | elapsed {elapsed:.1f}s")

    if not labels_list:
        raise RuntimeError("No images loaded. Check your subset folders and image formats.")

    images = np.stack(images_list, axis=0)
    labels = np.array(labels_list, dtype=int)

    idx = rng.permutation(len(labels))
    images = images[idx]
    labels = labels[idx]

    if max_images > 0 and len(labels) > max_images:
        images = images[:max_images]
        labels = labels[:max_images]

    print(f"Loaded subset: images={images.shape}, labels={labels.shape}, K={k}")
    return images, labels, class_names


def featurize_resnet18(images_uint8: np.ndarray, batch_size: int, device: str, seed: int) -> np.ndarray:
    try:
        import torch
        import torchvision
    except Exception as e:
        print(e)
    torch.manual_seed(seed)
    np.random.seed(seed)

    weights = getattr(torchvision.models, "ResNet18_Weights", None)
    if weights is not None:
        model = torchvision.models.resnet18(weights=weights.DEFAULT)
        preprocess = weights.DEFAULT.transforms()
        print("Loaded ResNet-18 weights: DEFAULT (torchvision)")
    else:
        model = torchvision.models.resnet18(pretrained=True)
        preprocess = None
        print("Loaded ResNet-18 weights: pretrained=True (legacy torchvision)")

    model.eval()
    model.to(device)

    feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))

    feats = []
    n = images_uint8.shape[0]
    t0 = time.time()
    print(f"ResNet featurizing {n} images on {device} (batch_size={batch_size}) ...")
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = images_uint8[start:end]
        x = torch.from_numpy(batch).to(device=device, dtype=torch.float32) / 255.0
        x = x.permute(0, 3, 1, 2)

        if preprocess is not None:
            x = preprocess(x)

        with torch.no_grad():
            y = feature_extractor(x).flatten(1)
        feats.append(y.detach().cpu().numpy())

        if start == 0 or end == n or (start // batch_size) % 20 == 0:
            elapsed = time.time() - t0
            rate = end / max(elapsed, 1e-9)
            print(f"  {end:6d}/{n} | elapsed {elapsed:.1f}s | {rate:.1f} img/s")

    X = np.concatenate(feats, axis=0)
    return normalize(X)


def maybe_load_cached_features(cache_path: Path, use_cache: bool) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    if not use_cache:
        return None
    if cache_path.exists():
        print(f"Loading cached features from: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        class_names = list(data["class_names"])
        print(f"Loaded cache: X={X.shape}, y={y.shape}, K={len(set(y))}")
        return X, y, class_names
    return None


def save_cached_features(cache_path: Path, X: np.ndarray, y: np.ndarray, class_names: List[str]) -> None:
    tmp = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, X=X, y=y, class_names=np.array(class_names, dtype=object))
    tmp.replace(cache_path)
    print(f"Saved features cache to: {cache_path}")


# =========================
# Main
# =========================
def make_algorithms(initial_centroids: np.ndarray, k: int, balance_lambda: float) -> List[Tuple[str, object]]:
    return [
        ("0. Adam-on-Sphere Baseline",
         OSKMWithAdamOnSphereBaseline(k=k, learning_rate=LEARNING_RATE, initial_centroids=initial_centroids, balance_lambda=balance_lambda,)),
        ("1. OSKM Baseline",
         OnlineSphericalKMeans_Fair(k=k, learning_rate=LEARNING_RATE, initial_centroids=initial_centroids, balance_lambda=balance_lambda)),
        ("ESA-Stream (Grid+Decay → Macro-K) on L2 (Baseline)",
         ESAStreamL2MacroBaseline(dim=initial_centroids.shape[1], macro_k=k, balance_lambda=balance_lambda,
                                  grid_len=0.5, lam=0.998, prune_min_density=1e-3,
                                  recluster_every=500, max_grids_for_macro=2000, macro_iters=15, seed=0)),
        ("2. OSKM + Momentum (Ablation)",
         OSKMWithMomentum(k=k, learning_rate=LEARNING_RATE, momentum=MOMENTUM, initial_centroids=initial_centroids, balance_lambda=balance_lambda)),
        ("3. OSKM + Confidence Weighting (Ablation)",
         OSKMWithConfidenceWeighting(k=k, learning_rate=LEARNING_RATE, initial_centroids=initial_centroids, balance_lambda=balance_lambda)),
        ("4. Momentum-Enhanced Hybrid SAOKM (Full)",
         MomentumHybridSAOKM(k=k, learning_rate=LEARNING_RATE, momentum=MOMENTUM, initial_centroids=initial_centroids, balance_lambda=balance_lambda)),
        ("5. SAOKM Original",
         SAOKM_Fair_org(k=k, alpha=ALPHA, initial_centroids=initial_centroids, balance_lambda=balance_lambda)),
        
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ImageNet subset online spherical clustering benchmark (kmeans++ init + balanced assignment + optional noise).")

    p.add_argument("--data-root", type=str, default="",
                   help="Path to ImageNet-style subset folder: root/<class>/*.jpg")
    p.add_argument("--download-url", type=str, default="",
                   help="Direct URL to a public subset archive (.zip, .tar, .tar.gz, .tgz). If set, script downloads+extracts.")
    p.add_argument("--dataset-name", type=str, default="imagenet_subset",
                   help="Name for the downloaded dataset folder under --dataset-cache-dir.")
    p.add_argument("--dataset-cache-dir", type=str, default="./datasets_cache",
                   help="Where to store downloaded+extracted datasets.")
    p.add_argument("--force-download", action="store_true",
                   help="Re-download and re-extract even if dataset exists.")

    p.add_argument("--max-classes", type=int, default=10,
                   help="Use first N class folders (sorted by name). Set 0 for all.")
    p.add_argument("--images-per-class", type=int, default=800,
                   help="Randomly sample up to N images per class (0 for all).")
    p.add_argument("--max-images", type=int, default=0,
                   help="Optional global cap on total images after sampling (0 disables).")
    p.add_argument("--resize", type=int, default=RESNET_IMAGE_SIZE,
                   help="Resize images to this square size before featurizing (default 224).")

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--seed", type=int, default=SEED)


    # Repeated runs for statistics (evaluation seeds vary per run)
    p.add_argument("--runs", type=int, default=1,
                   help="Repeat evaluation this many times (features + init centroids are computed once).")
    p.add_argument("--eval-seed-step", type=int, default=1000,
                   help="Per-run seed increment: eval_seed = seed + run_idx * eval_seed_step.")

    # Wilcoxon signed-rank test (paired across runs)
    p.add_argument("--wilcoxon", action="store_true",
                   help="Run paired Wilcoxon signed-rank tests vs a baseline method (requires --runs >= 2).")
    p.add_argument("--wilcoxon-baseline", type=str, default="",
                   help="Baseline method name. If empty, uses the first method in the algorithm list.")
    p.add_argument("--wilcoxon-metric", type=str, default="nmi", choices=["nmi", "ari"],
                   help="Metric to use for Wilcoxon tests.")
    p.add_argument("--balance-lambda", type=float, default=0.15,
                   help="Popularity penalty strength λ in score=sim-λ*log(count+1). Set 0 to disable.")

    p.add_argument("--noise-sigma", type=float, default=0.0,
                   help="Gaussian noise stddev added to features (then renormalized). 0 disables.")
    p.add_argument("--feature-dropout", type=float, default=0.0,
                   help="Randomly zero out this fraction of feature dims per sample (then renormalize). 0 disables.")

    p.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    p.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    p.add_argument("--eval-size", type=int, default=EVAL_SIZE)
    p.add_argument("--silhouette", action="store_true", help="Compute silhouette score (can be slow).")

    p.add_argument("--cache-dir", type=str, default="./cache_imagenet_subset")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--kpp-pool", type=int, default=50000,
                   help="Candidate pool size for k-means++ seeding (0 uses all points).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    data_root = args.data_root.strip()
    if args.download_url.strip():
        data_root_path = download_and_extract_subset(
            url=args.download_url.strip(),
            dataset_cache_dir=Path(args.dataset_cache_dir),
            dataset_name=args.dataset_name.strip() or "imagenet_subset",
            force=args.force_download,
        )
        data_root = str(data_root_path)

    if not data_root:
        raise SystemExit(
            "You must provide either --data-root <local-subset> or --download-url <public-archive-url>."
        )

    feat_cache_path = make_feature_cache_path(
        cache_dir=Path(args.cache_dir),
        data_root=data_root,
        max_classes=args.max_classes,
        images_per_class=args.images_per_class,
        max_images=args.max_images,
        seed=args.seed,
    )

    cached = maybe_load_cached_features(feat_cache_path, use_cache=not args.no_cache)
    if cached is not None:
        X, y_true, class_names = cached
    else:
        images, y_true, class_names = load_imagenet_subset_as_numpy(
            data_root=data_root,
            seed=args.seed,
            max_classes=args.max_classes,
            images_per_class=args.images_per_class,
            max_images=args.max_images,
            resize=args.resize,
        )
        print(f"Class names (label order): {class_names[:min(10, len(class_names))]}{' ...' if len(class_names) > 10 else ''}")

        print("\nFeaturizing with: resnet18")
        t0 = time.time()
        X = featurize_resnet18(images, batch_size=args.batch_size, device=args.device, seed=args.seed)
        t1 = time.time()
        print(f"Feature matrix: {X.shape} | featurization time: {t1-t0:.2f}s")

        if not args.no_cache:
            save_cached_features(feat_cache_path, X, y_true, class_names)

    # Optional robustness perturbations (do not touch cache on disk)
    if args.noise_sigma > 0.0 or args.feature_dropout > 0.0:
        print(f"Applying feature perturbations: noise_sigma={args.noise_sigma}, feature_dropout={args.feature_dropout}")
        X = apply_feature_perturbations(X, seed=args.seed, noise_sigma=args.noise_sigma, dropout_p=args.feature_dropout)
        print("Feature perturbations applied.")




    k = int(np.unique(y_true).size)

    if k < 2:
        raise RuntimeError("Need at least 2 classes with images to run clustering.")
    print(f"\nUsing K={k} clusters/classes for evaluation.")
    
    # Compute shared initial centroids once (keeps comparisons fair across runs)
    initial_centroids = get_initial_centroids_kmeanspp(X=X, k=k, seed=args.seed, pool_size=args.kpp_pool)
    print("Generating shared initial centroids (k-means++ on cosine/spherical)... done.")
    
    runs = max(1, int(args.runs))
    all_stream: Dict[str, List[EvalResult]] = {}
    all_final: Dict[str, List[EvalResult]] = {}
    algo_names: Optional[List[str]] = None
    
    for run_idx in range(runs):
        eval_seed = int(args.seed) + run_idx * int(args.eval_seed_step)
        print("\n" + "#" * 90)
        print(f"RUN {run_idx + 1}/{runs}  (eval seed={eval_seed})")
        print("#" * 90)
    
        print("\n" + "=" * 90)
        print("STREAMING EVALUATION (predict-before-learn)")
        print("=" * 90)
    
        algos_stream = make_algorithms(initial_centroids, k=k, balance_lambda=args.balance_lambda)
        if algo_names is None:
            algo_names = [name for name, _ in algos_stream]
    
        stream_results: List[EvalResult] = []
        for name, algo in algos_stream:
            res = evaluate_streaming(
                algo, name, X, y_true,
                sample_size=args.sample_size,
                seed=eval_seed,
                compute_silhouette=args.silhouette,
            )
            stream_results.append(res)
            all_stream.setdefault(name, []).append(res)
            if True:
                silv = 'N/A' if res.silhouette is None else f"{res.silhouette:.4f}"
                print(f"[RUN {run_idx + 1}] STREAM finished: {name} | k={res.n_clusters} | NMI={res.nmi:.4f} | ARI={res.ari:.4f} | Sil={silv} | total={res.total_time_s:.2f}s (learn={res.learn_time_s:.2f}s, pred={res.predict_time_s:.2f}s)")
    
        print("\n" + "=" * 90)
        print("FINAL-MODEL EVALUATION (train first, then assign)")
        print("=" * 90)
    
        final_results: List[EvalResult] = []
        algos_final = make_algorithms(initial_centroids, k=k, balance_lambda=args.balance_lambda)
        algo_map = {name: algo for name, algo in algos_final}
        for name in algo_names:
            fresh = algo_map[name]
            res = evaluate_final_model(
                fresh, name, X, y_true,
                train_size=args.train_size,
                eval_size=args.eval_size,
                seed=eval_seed,
                compute_silhouette=args.silhouette,
            )
            final_results.append(res)
            all_final.setdefault(name, []).append(res)
            if True:
                silv = 'N/A' if res.silhouette is None else f"{res.silhouette:.4f}"
                print(f"[RUN {run_idx + 1}] FINAL  finished: {name} | k={res.n_clusters} | NMI={res.nmi:.4f} | ARI={res.ari:.4f} | Sil={silv} | total={res.total_time_s:.2f}s (learn={res.learn_time_s:.2f}s, pred={res.predict_time_s:.2f}s)")
    
        # Preserve original single-run ranking behavior
        if runs == 1:
            stream_sorted = sorted(stream_results, key=lambda r: r.nmi, reverse=True)
            final_sorted = sorted(final_results, key=lambda r: r.nmi, reverse=True)
            print_ranking("FINAL RANKING - STREAMING EVALUATION (ImageNet subset)", stream_sorted)
            print_ranking("FINAL RANKING - FINAL-MODEL EVALUATION (ImageNet subset)", final_sorted)
    
    # Aggregate statistics across runs
    if runs > 1:
        def _summarize(metric_name: str, values: List[float]) -> str:
            arr = np.array(values, dtype=float)
            return (f"mean={arr.mean():.4f} ± {arr.std(ddof=1):.4f} | "
                    f"median={np.median(arr):.4f} | min={arr.min():.4f} | max={arr.max():.4f}")
    
        def _print_stats(title: str, results_dict: Dict[str, List[EvalResult]]) -> None:
            print("\n" + "=" * 90)
            print(title)
            print("=" * 90)
    
            # Sort by mean NMI descending
            items = []
            for name, rs in results_dict.items():
                nmis = [r.nmi for r in rs]
                items.append((float(np.mean(nmis)), name))
            items.sort(reverse=True)
    
            for rank, (_, name) in enumerate(items, start=1):
                rs = results_dict[name]
                nmis = [r.nmi for r in rs]
                aris = [r.ari for r in rs]
                print(f"\n{rank:2d}. {name}")
                print(f"    NMI: {_summarize('nmi', nmis)}")
                print(f"    ARI: {_summarize('ari', aris)}")
                if rs[0].silhouette is not None:
                    sils = [r.silhouette for r in rs if r.silhouette is not None]
                    if len(sils) == len(rs):
                        print(f"    Silhouette: {_summarize('silhouette', sils)}")
    
        _print_stats("MULTI-RUN STATS - STREAMING EVALUATION (ImageNet subset)", all_stream)
        _print_stats("MULTI-RUN STATS - FINAL-MODEL EVALUATION (ImageNet subset)", all_final)
    
        # Optional Wilcoxon signed-rank tests (paired across runs)
        if args.wilcoxon:
            if runs < 2:
                print("\n[Wilcoxon] Need --runs >= 2 to run paired tests.")
            else:
                try:
                    from scipy.stats import wilcoxon
                except Exception as e:
                    print(f"\n[Wilcoxon] scipy not available ({e}). Install scipy to enable Wilcoxon tests.")
                else:
                    baseline = (args.wilcoxon_baseline.strip() if args.wilcoxon_baseline else "")
                    if not baseline:
                        baseline = algo_names[0] if algo_names else ""
                    metric = args.wilcoxon_metric
    
                    def _scores(d: Dict[str, List[EvalResult]], name: str) -> np.ndarray:
                        if metric == "nmi":
                            return np.array([r.nmi for r in d[name]], dtype=float)
                        return np.array([r.ari for r in d[name]], dtype=float)
    
                    if baseline not in all_stream or baseline not in all_final:
                        print(f"\n[Wilcoxon] Baseline '{baseline}' not found. Available methods: {list(all_stream.keys())}")
                    else:
                        def _wilcoxon_table(title: str, d: Dict[str, List[EvalResult]]) -> None:
                            print("\n" + "-" * 90)
                            print(title)
                            print("-" * 90)
                            base = _scores(d, baseline)
                            print(f"Baseline: {baseline} | metric={metric} | runs={runs}")
                            for name in algo_names or list(d.keys()):
                                if name == baseline:
                                    continue
                                x = _scores(d, name)
                                try:
                                    stat, p = wilcoxon(x, base, zero_method="wilcox", correction=False, alternative="two-sided")
                                except ValueError:
                                    # e.g. all differences are zero
                                    stat, p = float("nan"), 1.0
                                delta = float(np.median(x - base))
                                print(f"{name:45s} medianΔ={delta:+.4f}   W={stat}   p={p:.6g}")
    
                        _wilcoxon_table("WILCOXON (STREAMING)", all_stream)
                        _wilcoxon_table("WILCOXON (FINAL-MODEL)", all_final)
    
    
if __name__ == "__main__":
    main()
