# Hybrid SAOKM: Confidence-Modulated Momentum for Online Spherical Clustering

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **Hybrid SAOKM** (Semantic Adaptive Online K-Means), a family of confidence-aware online spherical clustering algorithms for high-dimensional semantic data streams.

## 📄 Paper

**Title:** Hybrid SAOKM: Confidence-Modulated Momentum for Online Spherical Clustering

**Abstract:** Clustering high-dimensional semantic data streams requires online methods that adapt rapidly while remaining robust to noise, overlap, and weak separability. We propose a family of confidence-aware online spherical clustering algorithms that adapt updates based on assignment quality. Our Hybrid SAOKM scales both learning rates and momentum by cosine similarity, treating similarity as a confidence signal that amplifies reliable updates while suppressing noise. This confidence-gated momentum yields smoother centroid trajectories and improved stability under strict single-pass constraints.

---

## 🌟 Key Features

- **Confidence-Aware Learning**: Uses cosine similarity as a principled confidence signal to modulate update strength
- **Momentum Modulation**: Combines time-decayed learning rates with confidence-gated momentum for improved stability
- **Strictly Online**: Single-pass, bounded memory O(kd) per update
- **Robust**: Handles noisy embeddings, weak separability, and adversarial overlap
- **Scalable**: Efficient implementation suitable for large-scale streaming applications

---


# Initialize the algorithm
clusterer = HybridSAOKM(
    k=4,                    # number of clusters
    d=150,                  # embedding dimension
    alpha=0.1,              # base learning rate
    beta=0.0001,            # learning rate decay
    gamma=0.9,              # momentum coefficient
    c_fl=0.5,               # confidence lower bound
    c_fu=1.0,               # confidence upper bound
    seed=42
)


---

## 📊 Experiments

We provide complete experimental scripts for all datasets reported in the paper:

### Datasets

1. **AG News** (`AGNewsS.py`) - 4-way news topic classification
2. **Reuters RCV1** (`ReutersS.py`) - Multi-topic newswire corpus
3. **20 Newsgroups** (`20NewsgroupsS.py`) - 20-topic discussion benchmark
4. **ImageNet** (`imagenetS.py`) - 10-class ImageNet with ResNet-18 embeddings
5. **Twitter Stream** (`TwitterS.py`) - Synthetic short-text adversarial stream

### Running Experiments

Each experiment script supports multi-run evaluation with configurable parameters:

```bash
# Run AG News experiments (30 runs)
python AGNewsS.py --runs 30 --output_dir results/agnews

# Run with custom hyperparameters
python AGNewsS.py --runs 30 --learning_rate 0.1 --momentum 0.9 --c_fl 0.5

# Single-threaded for reproducibility
python AGNewsS.py --runs 30 --single-thread

# Quick test (fewer runs)
python AGNewsS.py --runs 5
```

### Common Arguments

All experiment scripts support the following arguments:

- `--runs`: Number of independent runs (default: 30)
- `--output_dir`: Directory for results (default: `results/`)
- `--single-thread`: Force single-threaded execution for determinism
- `--learning_rate`: Base learning rate α (default: 0.08)
- `--momentum`: Momentum coefficient γ (default: 0.9)
- `--c_fl`: Confidence lower bound (default: 0.5)
- `--seed`: Base random seed (default: 42)

### Output Format

Each experiment produces two CSV files:

1. **`results_long.csv`**: One row per run × algorithm × evaluation mode
   - Columns: `run`, `algorithm`, `mode`, `NMI`, `ARI`, `Silhouette`, `runtime`

2. **`results_summary.csv`**: Aggregated statistics per algorithm
   - Columns: `algorithm`, `mode`, `mean_NMI`, `std_NMI`, `mean_ARI`, `std_ARI`, etc.

---

## 🧪 Algorithms Implemented

The repository includes implementations of all methods compared in the paper:

### Main Methods
- **OSKM**: Online Spherical K-Means (baseline)
- **SAOKM Original**: Confidence-weighted updates without momentum
- **Hybrid SAOKM**: Full method with confidence-modulated momentum (proposed)

### Ablations
- **OSKM + Momentum**: Classical momentum without confidence gating
- **OSKM + Confidence Weighting**: Confidence weighting without momentum
- **OSKM + Adam**: Adam-style adaptive optimization on the sphere

### Comparison Baseline
- **ESA-Stream**: Self-adaptive streaming clustering (not strictly single-pass)

---

## 📈 Reproducing Paper Results

To reproduce all results from the paper:

```bash
# Create results directory
mkdir -p paper_results

# Run all experiments (takes several hours)
python AGNewsS.py --runs 30 --output_dir paper_results/agnews
python ReutersS.py --runs 30 --output_dir paper_results/reuters
python 20NewsgroupsS.py --runs 30 --output_dir paper_results/newsgroups
python imagenetS.py --runs 30 --data-root c:\imagenet-10 --max-classes 10   --images-per-class 800   --resize 224   --sample-size 5000  --train-size 3000   --eval-size 5000   --noise-sigma 0.12  --feature-dropout 0.2  --balance-lambda 0.15 --runs 2 --wilcoxon
 --output_dir paper_results/imagenet


# Statistical significance tests are computed automatically
# Results will match Tables 2-10 in the paper
```



## 🔧 Hyperparameter Tuning

Best-performing hyperparameters (from Appendix 8):

```python
# Recommended settings for Hybrid SAOKM
config = {
    'alpha': 0.1,       # Higher learning rate for aggressive adaptation
    'gamma': 0.9,       # Strong momentum for stability
    'c_fl': 0.5,        # Moderate confidence floor
    'c_fu': 1.0,        # Upper bound (less critical)
    'beta': 0.0001      # Learning rate decay coefficient
}
```

**Key findings from ablation studies:**
- Learning rate (α) is the dominant factor affecting quality
- Higher momentum (γ=0.9) consistently outperforms lower values
- Confidence floor (c_fl=0.5) prevents momentum collapse
- These settings are stable across datasets and evaluation modes

---


## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{hybrid-saokm-2026,
  title={Hybrid SAOKM: Confidence-Modulated Momentum for Online Spherical Clustering},
  author={Anonymous},
  journal={Under Review},
  year={2026}
}

## 📋 Requirements

```
numpy>=1.21.0
scikit-learn>=1.0.0
pandas>=1.3.0
scipy>=1.7.0
```

Python 3.8 or higher is required.

---

## 🗂️ Repository Structure

```
hybrid-saokm/
├── README.md                      # This file
│
├── experiments/                   # Experiment scripts
│   ├── AGNewsS.py                # AG News experiments
│   ├── ReutersS.py               # Reuters RCV1 experiments
│   ├── 20NewsgroupsS.py          # 20 Newsgroups experiments
│   ├── imagenetS.py              # ImageNet experiments
│   └── TwitterS.py               # Synthetic Twitter stream



 **L2 Normalization**
   - All vectors normalized to unit length
   - Data lies on unit hypersphere S^(d-1)

### Evaluation Modes

**STREAM Mode (Predict-Before-Learn)**
- Process each point sequentially
- Make prediction with current centroids
- Update centroids after prediction
- Reports online clustering quality

**FINAL Mode (Train-Then-Assign)**
- Train on prefix of stream
- Evaluate on held-out set
- No updates during evaluation
- Reports converged model quality



## ❓ FAQ

**Q: Why are silhouette scores negative or near zero?**

A: In high-dimensional cosine-normalized spaces with overlapping clusters, silhouette scores are often uninformative. We report them for completeness but rely primarily on NMI/ARI for quality assessment.

**Q: How do I adapt this to my own embeddings?**

A: Ensure your data is L2-normalized, then use the basic API shown in Quick Start. The key requirement is that all vectors lie on the unit hypersphere.

**Q: What if I have concept drift?**

A: The current implementation assumes stationary distributions. Future work will address non-stationary streams with explicit drift detection and adaptation mechanisms.

**Q: Can I use this for offline clustering?**

A: While the algorithms are designed for online settings, you can process a static dataset in arbitrary order. However, offline k-means or spherical k-means may be more appropriate.

**Q: How sensitive is the method to initialization?**

A: Our ablation studies (Appendix 8) show that with proper hyperparameters, the method is relatively robust to initialization. Random and k-means++ initialization produce similar results once dynamics are tuned.


---

## 🙏 Acknowledgments

We thank the authors of the baseline methods (OSKM, ESA-Stream) and the creators of the benchmark datasets (AG News, Reuters RCV1, 20 Newsgroups, ImageNet) for making their work publicly available.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔄 Version History

- **v1.0.0** (2026-02) - Initial release with paper submission
  - Core algorithms: OSKM, SAOKM, Hybrid SAOKM
  - All experiments from paper
  - Complete ablation studies
  - Statistical significance testing

---

**Note**: This is an anonymized repository for peer review. Full author information and institutional affiliations will be added upon acceptance.
