# Optimizer & LR Schedule Interactions for Genomic Classification 🧬

> **ASDS 6304 — Optimization Methods · Group 5 · University of Texas at Arlington · Spring 2026**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-ee4c2c.svg)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-f89939.svg)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Dataset-TCGA%20Pan--Cancer-9b59b6.svg)](https://xenabrowser.net)
[![Best Acc](https://img.shields.io/badge/Best%20Test%20Acc-92.9%25-brightgreen.svg)]()
[![Models](https://img.shields.io/badge/Model%20Fits-144-informational.svg)]()

A structured **6 × 4 × 2 factorial benchmark** of optimizers, learning-rate schedules, and regularizers on multinomial logistic regression for 33-class TCGA Pan-Cancer classification — asking whether the optimizer or the schedule drives generalization in high-dimensional tabular genomics.

**Headline finding:** The optimizer matters ~20× more than the schedule. L-BFGS wins. Adam is suboptimal here.

---

## 📋 Table of Contents

- [Background](#background)
- [Dataset](#dataset)
- [Experimental Design](#experimental-design)
- [Results](#results)
- [Gene Budget Extension](#gene-budget-extension)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Conclusions](#conclusions)

---

## Background

Applied genomic studies rarely justify their optimizer choice — Adam is used reflexively, cosine schedules are copied from deep-learning pipelines, and L-BFGS is overlooked. This project asks two questions simultaneously:

1. **Does optimizer choice matter** for regularized logistic regression in 20,000-dimensional gene-expression space?
2. **Do learning-rate schedules** (cosine, exponential, warm restarts) translate from deep networks to shallow tabular models?

The answer: yes to (1), largely no to (2).

---

## Dataset

**TCGA Pan-Cancer Atlas** via [UCSC Xena Toil Recompute](https://xenabrowser.net):

| Property | Value |
|---|---|
| Full cohort | 10,535 tumour samples, 58,581 genes |
| Cancer types | 33 |
| Working subset | 495 samples (15 per type, stratified, seed 42) |
| Variance filter | Top 20,000 genes (removes 8,556 constant transcripts) |
| Split | 60 / 20 / 20 → 297 / 99 / 99 samples |
| Normalization | StandardScaler fit on train only |

Gene expression is RSEM log₂(count + 1), harmonized across cancer types via the Toil uniform pipeline.

---

## Experimental Design

A fully-crossed **6 × 4 × 2 × 3 = 144 model fits** factorial with architecture and loss held fixed throughout. Any performance difference is attributable to the optimization regime alone.

### Optimizers

| Optimizer | Update Rule | Family |
|---|---|---|
| Full-batch GD | `w ← w − η ∇L(w)` | Deterministic first-order |
| SGD (b=1) | `w ← w − η ∇Lᵢ(w)` | Stochastic first-order |
| Mini-batch SGD | `w ← w − η ∇L_B(w)`, \|B\|=32 | Stochastic first-order |
| SGD + Momentum | `v ← βv + ∇L; w ← w − ηv`, β=0.9 | Stochastic w/ momentum |
| Adam | Adaptive moment estimates | Adaptive first-order |
| L-BFGS | `w ← w − ηH⁻¹∇L` | Quasi-Newton |

### Learning-Rate Schedules

| Schedule | Formula |
|---|---|
| Constant | `η_t = η₀` |
| Cosine decay | `η_t = η₀ · ½(1 + cos(πt/T))`, T=100 |
| Exponential decay | `η_t = η₀ · γᵗ`, γ=0.97 |
| Warm restarts | Cosine with period T₀=25 epochs |

### Model

Multinomial logistic regression as a single PyTorch linear layer (20,000 → 33), trained with cross-entropy + L1 or L2 regularization (λ = 10⁻⁴), 100 epochs, 3 random seeds (42, 123, 2026).

---

## Results

### Optimizer Benchmark

| Optimizer | Val Acc | Test Acc | Test F1 | Time (s) |
|---|---|---|---|---|
| **L-BFGS** | **0.937** | **0.914** | **0.907** | 13.19 |
| Mini-batch SGD | 0.936 | 0.903 | 0.896 | 1.32 |
| GD (full batch) | 0.921 | 0.887 | 0.878 | 0.43 |
| SGD + Momentum | 0.882 | 0.878 | 0.875 | 1.39 |
| Adam | 0.878 | 0.861 | 0.851 | 1.55 |
| SGD (batch=1) | 0.844 | 0.822 | 0.808 | 31.10 |

- **Peak individual test accuracy:** 0.919 (macro-F1 0.913), achieved by all 12 L-BFGS+L1 runs (std = 0.000 across all seeds — deterministic convex convergence) and 3 Mini-batch SGD runs.
- **Best practical choice:** Mini-batch SGD with Constant + L1 reaches 0.919 in **1.3 s** vs ~22–27 s per L-BFGS+L1 run.
- **Adam underperforms** and shows late-training oscillation — its per-parameter adaptive rates damp the steps the L1 penalty is trying to drive to zero.

### Schedule Effects

| Schedule | Mean Val Acc | Std |
|---|---|---|
| Cosine decay | 0.9021 | 0.0415 |
| Exponential decay | 0.9001 | 0.0398 |
| Warm restarts | 0.8990 | 0.0408 |
| Constant | 0.8973 | 0.0440 |

**Total schedule swing: 0.48 pp. Optimizer swing: 9.3 pp. The optimizer matters ~20× more.**

Within-optimizer schedule sensitivity: Mini-batch SGD = 0.0 pp, GD = 0.3 pp, L-BFGS = 0.5 pp — essentially zero. Only Adam shows a meaningful schedule effect (4.2 pp). Schedule tuning is not a productive use of effort on shallow tabular logistic regression.

### Regularizer Effects

L1 and L2 are essentially equivalent in generalization (mean val acc: L1 0.9019, L2 0.8973; difference within seed noise). **L1 is preferred** for clinical genomics because it produces an interpretable sparse gene panel (1,000–2,800 genes at optimal C) without any accuracy cost.

---

## Gene Budget Extension

Gene selection under a clinical measurement budget, comparing a **linear-program (LP)** formulation against **Lasso top-k** ranking at matched cardinality.

**LP formulation:** maximize Σ Fⱼ xⱼ s.t. Σ xⱼ ≤ k, 0 ≤ xⱼ ≤ 1, where Fⱼ is the one-way ANOVA Fisher F-statistic. Solved via `scipy.optimize.linprog` (HiGHS). The constraint matrix is totally unimodular — LP relaxation returns integer-optimal solutions exactly.

| k | LP Test Acc | Lasso Top-k Test Acc | Lasso Top-k F1 | Gene Overlap |
|---|---|---|---|---|
| 10 | 0.354 | 0.293 | 0.252 | 0/10 |
| 25 | 0.596 | 0.636 | 0.594 | 0/25 |
| 50 | 0.717 | 0.818 | 0.815 | 0/50 |
| 100 | 0.798 | 0.859 | 0.852 | 4/100 |
| **200** | **0.848** | **0.929** | **0.918** | **8/200** |

**Key findings:**
- Lasso top-k beats LP at **every budget tested** — by up to 10 pp at k=50.
- The two methods share **zero genes** at k ≤ 50. They are optimizing nearly orthogonal objectives: LP picks the individually-highest-F genes (redundancy-blind); Lasso picks one representative per correlated module.
- **Practical recommendation:** Rank by Lasso coefficient magnitude, not univariate F-score. A **200-gene panel reaches 0.929 test accuracy** — matching the full 2,820-gene model — with two orders of magnitude fewer measurements.

---

## Getting Started

### Prerequisites

```bash
pip install torch torchvision scikit-learn scipy numpy pandas matplotlib seaborn
```

### Data

Download the TCGA Pan-Cancer expression matrix from [UCSC Xena](https://xenabrowser.net/datapages/?cohort=TCGA%20Pan-Cancer%20(PANCAN)). The notebook caches parsed CSVs after the first download (~700 MB); subsequent runs skip it and complete in roughly half the time.

### Run

```bash
jupyter notebook code.ipynb
```

The notebook runs end-to-end on a Google Colab Pro A100 instance in ~90 minutes for a fresh run (144-model factorial alone: ~20 min). Set one Boolean flag to switch between the 495-sample subset and the full 10,535-sample cohort.

**Notebook sections in order:**
1. Data download & preprocessing
2. Exploratory analysis (variance distribution, correlation matrix, PCA)
3. Lasso regularization path (25 values of C, 0.01–100)
4. Learning-rate sensitivity sweep (SGD b=1, 6 log-spaced rates)
5. 144-model factorial benchmark (6 × 4 × 2 × 3)
6. Schedule and regularizer effect analysis
7. Confusion matrix for best model
8. LP gene-budget extension

---

## Repository Structure

```
.
├── code.ipynb       # Full experimental notebook (data → results → figures)
├── paper.pdf        # Full written report
├── ppt.pptx         # Presentation slides
└── README.md
```

---

## Conclusions

| Finding | Detail |
|---|---|
| **Best optimizer** | L-BFGS — mean test acc 0.914; deterministic at 0.919 with L1 across all seeds |
| **Best practical optimizer** | Mini-batch SGD — 0.903 mean test acc in 1.3 s per fit |
| **Schedule effect** | 0.48 pp total swing — do not over-tune schedules on shallow tabular classifiers |
| **Regularizer** | L1 ≈ L2 in accuracy; L1 preferred for sparse interpretable gene panels |
| **Gene budget** | 200 Lasso-ranked genes → 0.929 test acc (matches full 2,820-gene model) |
| **LP vs Lasso** | Lasso top-k dominates at every budget; LP and Lasso criteria are nearly orthogonal |

The broader implication: **optimizer choice on convex tabular models is substantive**, not an afterthought. The reflexive default of Adam is suboptimal in this setting. L-BFGS or Mini-batch SGD should be reported as baselines in future Lasso-on-genomics studies.

---

## Citation

```bibtex
@misc{group5_optim_genomics_2026,
  title={Optimizer and Learning-Rate Schedule Interactions for High-Dimensional Genomic Classification},
  author={Group 5, ASDS 6304},
  institution={University of Texas at Arlington},
  year={2026},
  note={Course project, Prof. Ce Bian}
}
```

---

## References

- Crawford & Greene (2024). Optimizer's dilemma. *Bioinformatics Advances*, 4(1).
- Kingma & Ba (2015). Adam. *ICLR*.
- Liu & Nocedal (1989). L-BFGS. *Mathematical Programming*, 45(1).
- Loshchilov & Hutter (2017). SGDR. *ICLR*.
- Tibshirani (1996). Lasso. *JRSS-B*, 58(1).
- TCGA Research Network (2013). *Nature Genetics*, 45(10).

---

<div align="center">
  <sub>ASDS 6304 · Group 5 · University of Texas at Arlington · Spring 2026 · Instructor: Prof. Ce Bian</sub>
</div>
