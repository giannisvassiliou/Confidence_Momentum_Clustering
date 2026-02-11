#!/usr/bin/env python3
"""
TWITTER_news_stream_clustering_enhanced.py

Enhanced version with comprehensive collapse resistance metrics for Section 7 of the paper.

NEW METRICS ADDED:
1. Assignment Entropy H(t) - measures cluster diversity over time
2. Gini Coefficient - measures assignment concentration
3. Centroid Angular Diversity - measures centroid separation
4. Temporal Stability - measures cluster volatility
5. Enhanced visualization and reporting

Original functionality preserved, metrics computed in sliding window.
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Iterable
from collections import deque, Counter
from pathlib import Path

import numpy as np

# Optional deps
try:
    from sklearn.preprocessing import normalize as sk_normalize
    from sklearn.metrics import silhouette_score
except Exception:
    sk_normalize = None
    silhouette_score = None

# Sentence embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# For plotting (optional)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False


# =========================
# Config defaults
# =========================
SEED = 42

# Algorithm hyperparams
LEARNING_RATE = 0.08
ALPHA = 0.08
MOMENTUM = 0.9

# Streaming evaluation window
WINDOW_SIZE = 2000
SIL_SAMPLE_SIZE = 800

# Confidence gating thresholds
CONF_CLIP_MIN = 0.1
CONF_CLIP_MAX = 1.0


# =========================
# NEW: Collapse Resistance Metrics
# =========================

def compute_assignment_entropy(assignments: Sequence[int], k: int) -> float:
    """
    Compute Shannon entropy of cluster assignments.
    
    H(t) = -Σ p_i log(p_i)
    
    Returns:
        0.0: complete collapse (all assignments to 1 cluster)
        log(k): perfect balance across k clusters
    """
    if not assignments:
        return 0.0
    
    counts = Counter(assignments)
    total = len(assignments)
    
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log(p + 1e-12)
    
    return float(entropy)


def compute_gini_coefficient(assignments: Sequence[int]) -> float:
    """
    Compute Gini coefficient of cluster size distribution.
    
    Returns:
        0.0: perfect equality (all clusters same size)
        ~1.0: perfect inequality (one cluster dominates)
    """
    if not assignments:
        return 0.0
    
    counts = list(Counter(assignments).values())
    if len(counts) == 1:
        return 1.0  # Complete concentration
    
    counts = sorted(counts)
    n = len(counts)
    total = sum(counts)
    
    if total == 0:
        return 0.0
    
    # Gini coefficient formula
    cumsum = np.cumsum(counts)
    gini = (2 * np.sum((np.arange(1, n + 1)) * counts) - (n + 1) * total) / (n * total)
    
    return float(gini)


def compute_centroid_angular_diversity(centroids: np.ndarray) -> Dict[str, float]:
    """
    Compute pairwise angular diversity metrics for centroids.
    
    Returns dict with:
        mean_similarity: mean cosine similarity (→1.0 indicates collapse)
        min_similarity: minimum pairwise similarity
        max_similarity: maximum pairwise similarity
        
    Lower mean_similarity indicates better separation/diversity.
    """
    k = centroids.shape[0]
    
    if k <= 1:
        return {
            "mean_similarity": 1.0,
            "min_similarity": 1.0,
            "max_similarity": 1.0,
        }
    
    # Compute pairwise cosine similarities
    sim_matrix = centroids @ centroids.T
    
    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones((k, k), dtype=bool), k=1)
    similarities = sim_matrix[mask]
    
    return {
        "mean_similarity": float(np.mean(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
    }


def compute_temporal_stability(current_assignments: Sequence[int], 
                               previous_assignments: Sequence[int]) -> float:
    """
    Compute fraction of points that changed cluster assignment.
    
    Returns:
        0.0: perfect stability (no changes)
        1.0: complete volatility (all points changed)
    """
    if not current_assignments or not previous_assignments:
        return 0.0
    
    n = min(len(current_assignments), len(previous_assignments))
    if n == 0:
        return 0.0
    
    changes = sum(1 for i in range(n) if current_assignments[i] != previous_assignments[i])
    return float(changes) / n


# =========================
# Enhanced Stats Dataclass
# =========================

@dataclass
class StreamingStats:
    """Enhanced with collapse resistance metrics."""
    t_seen: int = 0
    active_clusters: int = 0
    mean_conf: float = 0.0
    p90_conf: float = 0.0
    silhouette: Optional[float] = None
    churn_new_clusters: int = 0
    top_clusters: List[Tuple[int, int]] = None
    
    # NEW: Collapse resistance metrics
    entropy: float = 0.0
    max_entropy: float = 0.0  # log(k) for reference
    normalized_entropy: float = 0.0  # entropy / max_entropy
    gini: float = 0.0
    centroid_mean_sim: float = 0.0
    centroid_min_sim: float = 0.0
    centroid_max_sim: float = 0.0
    temporal_stability: float = 0.0  # fraction unchanged from previous window
    
    def __post_init__(self):
        if self.top_clusters is None:
            self.top_clusters = []


def compute_window_stats(
    preds: Sequence[int],
    confs: Sequence[float],
    window_embs: Optional[np.ndarray] = None,
    sil_sample_size: int = 800,
    centroids: Optional[np.ndarray] = None,
    previous_preds: Optional[Sequence[int]] = None,
    k: int = 80,
) -> StreamingStats:
    """Enhanced with collapse resistance metrics."""
    
    stats = StreamingStats()
    
    if not preds:
        return stats
    
    # Original metrics
    counts = Counter(preds)
    stats.active_clusters = len(counts)
    stats.churn_new_clusters = sum(1 for c in counts.values() if c == 1)
    stats.top_clusters = counts.most_common(5)
    
    if confs:
        stats.mean_conf = float(np.mean(confs))
        stats.p90_conf = float(np.percentile(confs, 90))
    
    if window_embs is not None and len(window_embs) >= 10 and silhouette_score is not None:
        if len(set(preds)) > 1:
            sample_size = min(len(window_embs), sil_sample_size)
            idx = np.random.choice(len(window_embs), size=sample_size, replace=False)
            try:
                stats.silhouette = silhouette_score(
                    window_embs[idx],
                    np.array(preds)[idx],
                    metric="cosine"
                )
            except:
                stats.silhouette = None
    
    # NEW: Collapse resistance metrics
    
    # 1. Assignment Entropy
    stats.entropy = compute_assignment_entropy(preds, k)
    stats.max_entropy = np.log(k)
    stats.normalized_entropy = stats.entropy / stats.max_entropy if stats.max_entropy > 0 else 0.0
    
    # 2. Gini Coefficient
    stats.gini = compute_gini_coefficient(preds)
    
    # 3. Centroid Angular Diversity
    if centroids is not None:
        diversity = compute_centroid_angular_diversity(centroids)
        stats.centroid_mean_sim = diversity["mean_similarity"]
        stats.centroid_min_sim = diversity["min_similarity"]
        stats.centroid_max_sim = diversity["max_similarity"]
    
    # 4. Temporal Stability
    if previous_preds is not None:
        stats.temporal_stability = 1.0 - compute_temporal_stability(preds, previous_preds)
    
    return stats


# =========================
# Results Tracking for Plotting
# =========================

class MetricsTracker:
    """Track metrics over time for visualization."""
    
    def __init__(self, algo_name: str):
        self.algo_name = algo_name
        self.timestamps = []
        self.active_clusters = []
        self.entropy = []
        self.normalized_entropy = []
        self.gini = []
        self.centroid_mean_sim = []
        self.mean_conf = []
        self.silhouette = []
    
    def add(self, t: int, stats: StreamingStats):
        self.timestamps.append(t)
        self.active_clusters.append(stats.active_clusters)
        self.entropy.append(stats.entropy)
        self.normalized_entropy.append(stats.normalized_entropy)
        self.gini.append(stats.gini)
        self.centroid_mean_sim.append(stats.centroid_mean_sim)
        self.mean_conf.append(stats.mean_conf)
        if stats.silhouette is not None:
            self.silhouette.append(stats.silhouette)
    
    def save_plots(self, output_dir: str):
        """Generate comparison plots."""
        if not PLOTTING_AVAILABLE:
            print("Matplotlib not available, skipping plots.")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create multi-panel figure
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f'Collapse Resistance Metrics: {self.algo_name}', fontsize=14, fontweight='bold')
        
        # Plot 1: Active Clusters
        ax = axes[0, 0]
        ax.plot(self.timestamps, self.active_clusters, 'b-', linewidth=2)
        ax.set_ylabel('Active Clusters', fontsize=10)
        ax.set_xlabel('Samples Seen', fontsize=10)
        ax.set_title('(a) Active Cluster Count Over Time', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Normalized Entropy
        ax = axes[0, 1]
        ax.plot(self.timestamps, self.normalized_entropy, 'g-', linewidth=2)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect Balance')
        ax.axhline(y=0.0, color='k', linestyle='--', alpha=0.5, label='Complete Collapse')
        ax.set_ylabel('Normalized Entropy', fontsize=10)
        ax.set_xlabel('Samples Seen', fontsize=10)
        ax.set_title('(b) Assignment Entropy (normalized by log k)', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Gini Coefficient
        ax = axes[1, 0]
        ax.plot(self.timestamps, self.gini, 'r-', linewidth=2)
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Complete Concentration')
        ax.axhline(y=0.0, color='g', linestyle='--', alpha=0.5, label='Perfect Equality')
        ax.set_ylabel('Gini Coefficient', fontsize=10)
        ax.set_xlabel('Samples Seen', fontsize=10)
        ax.set_title('(c) Assignment Concentration (Gini)', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Centroid Similarity
        ax = axes[1, 1]
        ax.plot(self.timestamps, self.centroid_mean_sim, 'm-', linewidth=2)
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Collapsed (identical)')
        ax.set_ylabel('Mean Pairwise Similarity', fontsize=10)
        ax.set_xlabel('Samples Seen', fontsize=10)
        ax.set_title('(d) Centroid Angular Diversity', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Mean Confidence
        ax = axes[2, 0]
        ax.plot(self.timestamps, self.mean_conf, 'c-', linewidth=2)
        ax.set_ylabel('Mean Confidence', fontsize=10)
        ax.set_xlabel('Samples Seen', fontsize=10)
        ax.set_title('(e) Assignment Confidence', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Raw Entropy
        ax = axes[2, 1]
        ax.plot(self.timestamps, self.entropy, 'orange', linewidth=2, label='Observed')
        if self.timestamps:
            max_ent = np.log(80)  # Assuming k=80
            ax.axhline(y=max_ent, color='g', linestyle='--', alpha=0.5, label=f'Max (log {80})')
        ax.set_ylabel('Entropy (nats)', fontsize=10)
        ax.set_xlabel('Samples Seen', fontsize=10)
        ax.set_title('(f) Raw Assignment Entropy', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = self.algo_name.replace(" ", "_").replace(".", "").replace("+", "plus")
        filepath = output_path / f"{filename}_metrics.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  → Saved plot: {filepath}")


# =========================
# Utilities
# =========================
def set_global_seed(seed: int) -> None:
    np.random.seed(seed)


def l2_normalize_vec(z: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(z) + 1e-12)
    return (z / denom).astype(np.float32, copy=False)


def normalize_mat(X: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return (X / denom).astype(np.float32, copy=False)


def dict_from_embedding(z: np.ndarray) -> Dict[str, float]:
    return {f"f{i}": float(z[i]) for i in range(z.shape[0])}


def embed_texts(encoder, texts: List[str], batch_size: int = 32) -> np.ndarray:
    Z = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    Z = np.asarray(Z, dtype=np.float32)
    return normalize_mat(Z)


def get_initial_centroids(k: int, n_features: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centroids = rng.normal(0.0, 0.1, size=(k, n_features)).astype(np.float32)
    return normalize_mat(centroids)


def vector_from_xdict(x_dict: Dict[str, float]) -> np.ndarray:
    return np.array([x_dict[k] for k in sorted(x_dict.keys())], dtype=np.float32)


# =========================
# Algorithms (same as original)
# =========================
class OnlineSphericalKMeans_Fair:
    def __init__(self, k: int, learning_rate: float, initial_centroids: np.ndarray):
        self.k = k
        self.learning_rate = float(learning_rate)
        self.centroids = initial_centroids.copy()
        self.n_samples_seen = 0

    def learn_one(self, x_dict: Dict[str, float]):
        z = l2_normalize_vec(vector_from_xdict(x_dict))
        sims = self.centroids @ z
        i = int(np.argmax(sims))

        lr = self.learning_rate / (1.0 + 0.0005 * self.n_samples_seen)
        self.centroids[i] = l2_normalize_vec(self.centroids[i] + lr * (z - self.centroids[i]))

        self.n_samples_seen += 1
        return self

    def predict_one(self, x_dict: Dict[str, float]) -> int:
        z = l2_normalize_vec(vector_from_xdict(x_dict))
        sims = self.centroids @ z
        return int(np.argmax(sims))

    def predict_with_conf(self, x_dict: Dict[str, float]) -> Tuple[int, float]:
        z = l2_normalize_vec(vector_from_xdict(x_dict))
        sims = self.centroids @ z
        i = int(np.argmax(sims))
        return i, float(sims[i])


class OSKMWithMomentum:
    def __init__(self, k: int, learning_rate: float, momentum: float, initial_centroids: np.ndarray):
        self.k = k
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.centroids = initial_centroids.copy()
        self.velocity = np.zeros_like(self.centroids)
        self.n_samples_seen = 0

    def learn_one(self, x_dict: Dict[str, float]):
        z = l2_normalize_vec(vector_from_xdict(x_dict))
        s = self.centroids @ z
        i = int(np.argmax(s))

        lr = self.learning_rate / (1.0 + 0.0005 * self.n_samples_seen)
        grad = z - self.centroids[i]

        self.velocity[i] = self.momentum * self.velocity[i] + grad
        self.centroids[i] = l2_normalize_vec(self.centroids[i] + lr * self.velocity[i])

        self.n_samples_seen += 1
        return self

    def predict_one(self, x_dict: Dict[str, float]) -> int:
        z = l2_normalize_vec(vector_from_xdict(x_dict))
        s = self.centroids @ z
        return int(np.argmax(s))

    def predict_with_conf(self, x_dict: Dict[str, float]) -> Tuple[int, float]:
        z = l2_normalize_vec(vector_from_xdict(x_dict))
        s = self.centroids @ z
        i = int(np.argmax(s))
        return i, float(s[i])


class OSKMWithConfidenceWeighting:
    def __init__(self, k: int, learning_rate: float, initial_centroids: np.ndarray):
        self.k = k
        self.learning_rate = float(learning_rate)
        self.centroids = initial_centroids.copy()
        self.n_samples_seen = 0

    def learn_one(self, x_dict: Dict[str, float]):
        z = l2_normalize_vec(vector_from_xdict(x_dict))
        s = self.centroids @ z
        i = int(np.argmax(s))
        conf = float(s[i])

        # Decayed base LR + clipped confidence step (prevents over-updating early winners)
        lr = self.learning_rate / (1.0 + 0.0005 * self.n_samples_seen)
        conf_step = float(np.clip(conf, 0.5, 1))
        effective_lr = lr * conf_step

        self.centroids[i] = l2_normalize_vec(self.centroids[i] + effective_lr * (z - self.centroids[i]))

        self.n_samples_seen += 1
        return self

    def predict_one(self, x_dict: Dict[str, float]) -> int:
        z = l2_normalize_vec(vector_from_xdict(x_dict))
        s = self.centroids @ z
        return int(np.argmax(s))

    def predict_with_conf(self, x_dict: Dict[str, float]) -> Tuple[int, float]:
        z = l2_normalize_vec(vector_from_xdict(x_dict))
        s = self.centroids @ z
        i = int(np.argmax(s))
        return i, float(s[i])


class MomentumHybridSAOKM:
    def __init__(self, k: int, learning_rate: float, momentum: float, initial_centroids: np.ndarray):
        self.k = k
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.centroids = initial_centroids.copy()
        self.velocity = np.zeros_like(self.centroids)
        self.n_samples_seen = 0

    def learn_one(self, x_dict: Dict[str, float]):
        z = l2_normalize_vec(vector_from_xdict(x_dict))
        s = self.centroids @ z
        i = int(np.argmax(s))
        conf = float(s[i])

        lr = self.learning_rate / (1.0 + 0.0005 * self.n_samples_seen)
        grad = z - self.centroids[i]

        # Stable hybrid SAOKM update:
        # - keep momentum fixed (do NOT multiply momentum by confidence)
        # - modulate the step size with clipped confidence
        conf_step = float(np.clip(conf, 0.5, 1))
        self.velocity[i] = self.momentum * self.velocity[i] + (lr * conf_step) * grad

        # Safety: cap the velocity norm to avoid runaway winner-takes-all collapse.
        vnorm = float(np.linalg.norm(self.velocity[i]))
        if vnorm > 1.0:
            self.velocity[i] *= (1.0 / (vnorm + 1e-12))

        self.centroids[i] = l2_normalize_vec(self.centroids[i] + self.velocity[i])

        self.n_samples_seen += 1
        return self

    def predict_one(self, x_dict: Dict[str, float]) -> int:
        z = l2_normalize_vec(vector_from_xdict(x_dict))
        s = self.centroids @ z
        return int(np.argmax(s))

    def predict_with_conf(self, x_dict: Dict[str, float]) -> Tuple[int, float]:
        z = l2_normalize_vec(vector_from_xdict(x_dict))
        s = self.centroids @ z
        i = int(np.argmax(s))
        return i, float(s[i])


def make_algorithms(initial_centroids: np.ndarray, k: int) -> List[Tuple[str, object]]:
    return [
        ("1. OSKM Baseline", OnlineSphericalKMeans_Fair(k, LEARNING_RATE, initial_centroids.copy())),
        ("2. OSKM + Momentum", OSKMWithMomentum(k, LEARNING_RATE, MOMENTUM, initial_centroids.copy())),
        ("3. OSKM + Confidence", OSKMWithConfidenceWeighting(k, LEARNING_RATE, initial_centroids.copy())),
        ("4. Momentum Hybrid SAOKM (ours)", MomentumHybridSAOKM(k, LEARNING_RATE, MOMENTUM, initial_centroids.copy())),
    ]


# =========================
# Tweet Sources (unchanged)
# =========================
def tweet_source_replay(jsonl_path: str) -> Iterable[Dict]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jsonl_path = os.path.join(script_dir, jsonl_path)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def tweet_source_live(bearer_token: str, rules: List[str]) -> Iterable[Dict]:
    try:
        import tweepy
    except ImportError:
        print("ERROR: tweepy is required for live mode. Install: pip install tweepy", file=sys.stderr)
        sys.exit(2)

    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

    try:
        existing = client.get_rules()
        if existing.data:
            ids = [r.id for r in existing.data]
            client.delete_rules(ids)
    except Exception as e:
        print(f"Warning clearing rules: {e}")

    for rule_text in rules:
        client.add_rules(tweepy.StreamRule(rule_text))

    class Listener(tweepy.StreamingClient):
        def on_tweet(self, tweet):
            yield {
                "id": str(tweet.id),
                "text": tweet.text,
                "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
            }

    stream = Listener(bearer_token=bearer_token, wait_on_rate_limit=True)
    stream.filter(tweet_fields=["created_at"])


# =========================
# ENHANCED Streaming Runner
# =========================
def run_streaming(
    tweets: Iterable[Dict],
    encoder,
    algo,
    algo_name: str,
    window_size: int,
    report_every: int,
    max_tweets: Optional[int],
    compute_silhouette: bool,
    k: int,
    output_dir: Optional[str] = None,
):
    """Enhanced with collapse resistance metrics and tracking."""
    
    t0 = time.time()
    seen = 0
    pred_time = 0.0
    learn_time = 0.0

    preds_window = deque(maxlen=window_size)
    confs_window = deque(maxlen=window_size)
    texts_window = deque(maxlen=window_size)
    
    # Track previous window for stability
    previous_preds = None
    
    # For silhouette
    if compute_silhouette:
        embs_window = deque(maxlen=window_size)
    else:
        embs_window = None

    # Metrics tracking for plots
    tracker = MetricsTracker(algo_name)

    batch_texts = []
    batch_meta = []
    batch_size = 32

    def _flush_batch():
        nonlocal seen, pred_time, learn_time, previous_preds
        
        if not batch_texts:
            return

        # Batch embed
        Z = embed_texts(encoder, batch_texts, batch_size=batch_size)

        for z, meta in zip(Z, batch_meta):
            if max_tweets is not None and seen >= max_tweets:
                break

            x_dict = dict_from_embedding(z)

            tp0 = time.time()
            p, conf = algo.predict_with_conf(x_dict)
            tp1 = time.time()
            pred_time += (tp1 - tp0)

            tl0 = time.time()
            algo.learn_one(x_dict)
            tl1 = time.time()
            learn_time += (tl1 - tl0)

            preds_window.append(int(p))
            confs_window.append(float(conf))
            texts_window.append(meta.get("text", ""))

            if embs_window is not None:
                embs_window.append(z)

            seen += 1

            if report_every and seen % report_every == 0:
                elapsed = time.time() - t0
                window_embs = None
                if embs_window is not None and len(embs_window) >= 10:
                    window_embs = np.asarray(list(embs_window), dtype=np.float32)

                # Get centroids for diversity metrics
                centroids = algo.centroids if hasattr(algo, 'centroids') else None

                stats = compute_window_stats(
                    preds_window,
                    confs_window,
                    window_embs=window_embs,
                    sil_sample_size=SIL_SAMPLE_SIZE,
                    centroids=centroids,
                    previous_preds=previous_preds,
                    k=k,
                )
                stats.t_seen = seen

                # Track for plotting
                tracker.add(seen, stats)

                # Enhanced reporting
                sil_str = f"{stats.silhouette:.3f}" if stats.silhouette is not None else "N/A"
                
                print("\n" + "=" * 120)
                print(f"[{algo_name}] seen={seen} | elapsed={elapsed:.1f}s | tweets/sec={seen/max(1e-6,elapsed):.1f}")
                print(f"  Window Size: {len(preds_window)}")
                print("-" * 120)
                print("ORIGINAL METRICS:")
                print(f"  active_clusters={stats.active_clusters} | mean_conf={stats.mean_conf:.3f} | "
                      f"p90_conf={stats.p90_conf:.3f} | sil={sil_str}")
                print(f"  churn(singletons)={stats.churn_new_clusters} | top_clusters: {stats.top_clusters[:3]}")
                print("-" * 120)
                print("COLLAPSE RESISTANCE METRICS:")
                print(f"  Entropy: {stats.entropy:.3f} (max={stats.max_entropy:.3f}, normalized={stats.normalized_entropy:.3f})")
                print(f"    → Interpretation: 0.0=complete collapse, 1.0=perfect balance")
                print(f"  Gini Coefficient: {stats.gini:.3f}")
                print(f"    → Interpretation: 0.0=equal distribution, 1.0=complete concentration")
                print(f"  Centroid Angular Diversity:")
                print(f"    mean_similarity={stats.centroid_mean_sim:.3f} | "
                      f"min={stats.centroid_min_sim:.3f} | max={stats.centroid_max_sim:.3f}")
                print(f"    → Interpretation: →1.0 indicates centroids collapsing together")
                if stats.temporal_stability > 0:
                    print(f"  Temporal Stability: {stats.temporal_stability:.3f} (fraction unchanged from prev window)")
                print("=" * 120)
                
                # Show example texts
                if stats.top_clusters:
                    top_cluster = stats.top_clusters[0][0]
                    examples = [t for t, p in zip(list(texts_window)[-300:], list(preds_window)[-300:]) if p == top_cluster]
                    examples = [e.replace("\n", " ").strip() for e in examples if e.strip()]
                    examples = examples[:3]
                    if examples:
                        print("  Examples from top cluster:")
                        for ex in examples:
                            print(f"    - {ex[:180]}{'...' if len(ex)>180 else ''}")
                
                # Update previous window for next stability calculation
                previous_preds = list(preds_window)

        batch_texts.clear()
        batch_meta.clear()

    for tw in tweets:
        if max_tweets is not None and seen >= max_tweets:
            break

        text = (tw.get("text") or "").strip()
        if not text:
            continue

        batch_texts.append(text)
        batch_meta.append(tw)

        if len(batch_texts) >= batch_size:
            _flush_batch()

    _flush_batch()

    total = time.time() - t0
    if seen == 0:
        print("No tweets processed.")
        return

    print("\n" + "=" * 120)
    print(f"FINAL SUMMARY [{algo_name}]")
    print("=" * 120)
    print(f"Processed: {seen} tweets | Total time: {total:.1f}s | Rate: {seen/total:.1f} tweets/sec")
    print(f"Timing breakdown:")
    print(f"  - Learning: {learn_time:.2f}s ({(learn_time/seen)*1000:.2f}ms/tweet)")
    print(f"  - Prediction: {pred_time:.2f}s ({(pred_time/seen)*1000:.2f}ms/tweet)")
    
    # Final window stats
    centroids = algo.centroids if hasattr(algo, 'centroids') else None
    final_stats = compute_window_stats(
        preds_window,
        confs_window,
        centroids=centroids,
        k=k,
    )
    
    print("-" * 120)
    print("FINAL WINDOW COLLAPSE METRICS:")
    print(f"  Active Clusters: {final_stats.active_clusters} / {k}")
    print(f"  Normalized Entropy: {final_stats.normalized_entropy:.3f}")
    print(f"  Gini Coefficient: {final_stats.gini:.3f}")
    print(f"  Centroid Mean Similarity: {final_stats.centroid_mean_sim:.3f}")
    print("=" * 120)
    
    # Save plots
    if output_dir and PLOTTING_AVAILABLE:
        tracker.save_plots(output_dir)


# =========================
# CLI
# =========================
def load_encoder(name: str, device: Optional[str] = None):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers required: pip install -U sentence-transformers")

    name = name.lower().strip()
    if name in ("bert", "bert-base", "bert-base-uncased"):
        model_id = "bert-base-uncased"
    elif name in ("roberta", "roberta-base"):
        model_id = "roberta-base"
    elif name in ("mpnet", "all-mpnet-base-v2", "mpnet-base"):
        model_id = "all-mpnet-base-v2"
    else:
        model_id = name

    if device:
        enc = SentenceTransformer(model_id, device=device)
    else:
        enc = SentenceTransformer(model_id)
    return enc, model_id


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Enhanced tweet clustering with collapse resistance metrics.")
    ap.add_argument("--mode", choices=["live", "replay"], default="replay")
    ap.add_argument("--input", default="", help="JSONL file for replay mode")
    ap.add_argument("--encoder", default="mpnet", help="bert | roberta | mpnet")
    ap.add_argument("--device", default="", help="cpu | cuda | mps")
    ap.add_argument("--k", type=int, default=80, help="Number of clusters")
    ap.add_argument("--window", type=int, default=WINDOW_SIZE)
    ap.add_argument("--report_every", type=int, default=500)
    ap.add_argument("--max_tweets", type=int, default=5000, help="0 for unlimited")
    ap.add_argument("--silhouette", action="store_true")
    ap.add_argument("--output_dir", default="./collapse_metrics_output", help="Directory for plots")
    ap.add_argument("--rules", nargs="*", default=[])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(SEED)

    max_tweets = None if (args.max_tweets == 0) else int(args.max_tweets)

    enc, model_id = load_encoder(args.encoder, device=args.device or None)

    probe = embed_texts(enc, ["probe"], batch_size=1)
    dim = int(probe.shape[1])

    initial_centroids = get_initial_centroids(k=int(args.k), n_features=dim, seed=SEED)

    algos = make_algorithms(initial_centroids, k=int(args.k))

    if args.mode == "replay":
        if not args.input:
            print("ERROR: --input required for replay mode", file=sys.stderr)
            sys.exit(2)
        tweet_iter = tweet_source_replay(args.input)
    else:
        bearer = os.environ.get("X_BEARER_TOKEN") or os.environ.get("TWITTER_BEARER_TOKEN")
        if not bearer:
            print("ERROR: Live mode requires X_BEARER_TOKEN", file=sys.stderr)
            sys.exit(2)
        rules = args.rules or [
            '(breaking OR "reported" OR "according to" OR Reuters OR "AP News" OR BBC OR CNN) lang:en -is:retweet'
        ]
        tweet_iter = tweet_source_live(bearer, rules)

    print("=" * 120)
    print("ENHANCED TWEET CLUSTERING WITH COLLAPSE RESISTANCE METRICS")
    print("=" * 120)
    print(f"mode={args.mode} | encoder={model_id} | k={args.k} | window={args.window}")
    print(f"report_every={args.report_every} | max_tweets={'∞' if max_tweets is None else max_tweets}")
    print(f"silhouette={'on' if args.silhouette else 'off'} | output_dir={args.output_dir}")
    print("=" * 120)

    for name, algo in algos:
        fresh_init = initial_centroids.copy()
        
        if name.startswith("1. OSKM"):
            fresh = OnlineSphericalKMeans_Fair(k=int(args.k), learning_rate=LEARNING_RATE, initial_centroids=fresh_init)
        elif name.startswith("2. OSKM + Momentum"):
            fresh = OSKMWithMomentum(k=int(args.k), learning_rate=LEARNING_RATE, momentum=MOMENTUM, initial_centroids=fresh_init)
        elif name.startswith("3. OSKM + Confidence"):
            fresh = OSKMWithConfidenceWeighting(k=int(args.k), learning_rate=LEARNING_RATE, initial_centroids=fresh_init)
        else:
            fresh = MomentumHybridSAOKM(k=int(args.k), learning_rate=LEARNING_RATE, momentum=MOMENTUM, initial_centroids=fresh_init)

        if args.mode == "live" and not name.startswith("4."):
            continue

        if args.mode == "replay":
            tweet_iter_local = tweet_source_replay(args.input)
        else:
            tweet_iter_local = tweet_iter

        run_streaming(
            tweet_iter_local,
            encoder=enc,
            algo=fresh,
            algo_name=name,
            window_size=int(args.window),
            report_every=int(args.report_every),
            max_tweets=max_tweets,
            compute_silhouette=bool(args.silhouette),
            k=int(args.k),
            output_dir=args.output_dir,
        )

    print("\n" + "=" * 120)
    print("All algorithms completed. Check output directory for visualizations.")
    print("=" * 120)


if __name__ == "__main__":
    main()
