# Dealing with Position Bias for Tap Propensity

![banner image](./img/banner.jpeg)

Most click models are wrong in the same way: they treat a click at position 1 and
a click at position 10 as equal signal. They are not. Position bias is one of the
most common and least corrected sources of noise in industrial ranking systems.

This repo is a deep dive into how to fix that — from the theory to a
production-ready implementation. It is structured as an annotated learning
resource: every design decision is explained, every tradeoff is made explicit, and
the code is written the way it would be written on a real ML platform team. The
`unbiastap` package is the artifact; understanding why it is built this way is the
point.

**Part 1 —** [Your CTR Metric Is Lying to You
](https://medium.com/@chesar/your-ctr-metric-is-lying-to-you-54baf1a041fc)

**Part 2 —** [Debiasing Clicks Is Step One. Making Them Usable Is Step Two](https://medium.com/@chesar/debiasing-clicks-is-step-one-making-them-usable-is-step-two-813b7ad7adfa)

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Local Setup](#local-setup)

---

## Overview

Position bias does not announce itself. It shows up silently: your model learns
that top-ranked items get clicked more, then ranks them even higher, then they get
clicked even more. The feedback loop compounds. The items you never show get no
signal. The items you always show look artificially good.

Fixing it requires knowing how much of each click was preference versus exposure —
and then making the training signal reflect preference, not just opportunity. This
repo builds that correction from scratch and makes it operationally stable:

- **`IPSXGBoostAdapter`** — XGBoost classifier trained with Inverse Propensity
  Scoring sample weights, so clicks at underexposed positions are up-weighted to
  compensate for their lower reach probability
- **`ClipSelector`** — data-driven selection of the IPS weight clip threshold using
  Kish Effective Sample Size; finds the most permissive clip that keeps the
  weighted dataset above a target statistical power floor, no manual tuning needed
- **`ShrinkageSelector`** — cross-validated selection of the Bayesian shrinkage
  prior strength; blends empirical reach rates with a parametric decay model so
  sparse tail positions get structure instead of noise

---

## Repository Structure

```
unbiastap/          core package
  data/             data loading and dataset builders (swipe-next, carousel, grid)
  models/
    pointwise_clf/  IPSXGBoostAdapter and its config
    custom/         ClipSelector, ShrinkageSelector and their configs
  evaluation/       ranking metrics (NDCG, debiased NDCG) and diagnostic plots
  utils/            config loader
notebooks/
  00_eda.ipynb      exploratory data analysis on the Expedia dataset
  01_swipe_next_bias.ipynb  full debiasing walkthrough (naive → IPS → clip → parametric → blend)
config/
  constants.yaml    shared constants (features, max position, dropout rate, seed)
tests/              pytest suite for all custom classes
```

---

## Local Setup

**Requirements**

- Python 3.11 or higher
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management

**Step by step**

1. Clone the repository:

```bash
git clone <repo-url>
cd unbiastap
```

2. Install dependencies:

```bash
make install
```

This runs `poetry install` and creates a virtual environment with all
dependencies pinned in `poetry.lock`.

3. Download the Expedia dataset from
   [Kaggle — Expedia Hotel Recommendations](https://www.kaggle.com/c/expedia-hotel-recommendations/data)
   and place `train.csv` in the `data/` directory:

```
data/
  train.csv
```
