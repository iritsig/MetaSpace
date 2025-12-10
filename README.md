

# MetaSpace — Alignment of Compliant Tabular Data Using Meta-Space

[![version](https://img.shields.io/github/v/release/iritsig/MetaSpace?style=for-the-badge&logo=github)](https://github.com/iritsig/MetaSpace)
[![lint](https://img.shields.io/github/actions/workflow/status/iritsig/MetaSpace/lint.yml?label=lint&style=for-the-badge&logo=github)](https://github.com/iritsig/MetaSpace/actions)
[![build](https://img.shields.io/github/actions/workflow/status/iritsig/MetaSpace/build.yml?label=build&style=for-the-badge&logo=github)](https://github.com/iritsig/MetaSpace/actions)
[![test](https://img.shields.io/github/actions/workflow/status/iritsig/MetaSpace/test.yml?label=test&style=for-the-badge&logo=github)](https://github.com/iritsig/MetaSpace/actions)
[![codecov](https://img.shields.io/codecov/c/github/iritsig/MetaSpace?style=for-the-badge&logo=codecov)](#)
[![conventional commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg?style=for-the-badge&logo=conventionalcommits)](https://www.conventionalcommits.org)
[![semantic-release](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--release-e10079.svg?style=for-the-badge)](https://github.com/semantic-release/semantic-release)
[![contributor covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg?style=for-the-badge)](https://www.contributor-covenant.org/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg?style=for-the-badge)](https://opensource.org/licenses/BSD-3-Clause)

## Purpose & Philosophy

This repository hosts the code and **reproducible pipeline** for the paper:

> **Alignment of Compliant Tabular Data Using Meta-Space**  
> N. E. Kired, F. Ravat, J. Song, O. Teste  


We consider three **compliant tabular scenarios**:

1. **Complete–to–Complete**: schema + instances on both sides  
2. **Schema-only–to–Schema-only**: only attribute names on both sides  
3. **Instance-only–to–Instance-only**: only instance values on both sides  

The core idea is to *not* compute similarity by a single score (e.g. cosine) but to embed each candidate pair of attributes (or columns) into a **rich meta-space** of heterogeneous meta-features:

- **Classical** meta-features   
  - String similarities on names (Levenshtein, Damerau–OSA, LCS, prefix/suffix ratios, Jaccard/Dice, Jaro/Jaro–Winkler, n-grams, …)  
  - Vector similarities/distances on embeddings (cosine, Euclidean, Minkowski, Chebyshev, Canberra, Pearson, Spearman)

- **Spectral** meta-features  on embedding-derived matrices  
  - Stable Rank, RankMe, Alpha-ReQ, NESum, SelfCluster  
  - Spectral and Frobenius norms, normalized spectral contributions  

- **Topological** meta-features (Topological Data Analysis)  
  - Persistence entropy, lifetimes, and counts in homology dimensions \(H_0\) and \(H_1\)  
  - Bottleneck and Wasserstein distances between persistence diagrams  

These features are computed from:

- Column **names**  
- Column **value embeddings**  
- **Sliding-window point clouds** built from column embedding vectors (to expose geometric structure)  
- **Global embedding matrices** of each tabular source  

A **supervised classifier** (the *Meta-Learner*) is trained in this meta-space to predict whether a pair of attributes is a semantic match.

The pipeline is organized into two main commands:

1. **`MetaSpace`** — builds the **meta-features** for all candidate attribute pairs between two tabular sources and attaches `true_match` labels from a **golden JSON**.
2. **`MetaLearn`** — trains and evaluates a classifier (e.g., **XGBoost**, **CatBoost**, **RandomForest**) on this meta-space and outputs metrics, predictions, and trained models.

---

## Abstract (paper summary)

Many real-world data integration tasks require aligning **compliant tabular data sources**, which may expose only schemas, only instances, or both. We propose a supervised framework, **Meta Space**, that uses a **meta-space of features** to address three alignment scenarios: complete–to–complete, schema-only–to–schema-only, and instance-only–to–instance-only.

Each tabular source is encoded using **pre-trained language models** (BERT, RoBERTa, DistilBERT, ALBERT, BART, MiniLM, …). From these dense vectors, we build a meta-space composed of **classical**, **spectral**, and **topological** meta-features capturing syntactic, semantic, geometric and structural properties of the embedding space. A supervised classifier trained on this meta-space predicts whether two sources (attributes) match.

On **551 tabular pairs** derived from the **Valentine** benchmark, our framework consistently outperforms strong baselines (including **Magneto**, **MagnetoGPT**, **MagnetoFTGPT**, **Coma++**, **Cupid**, **Similarity Flooding**, **DistributionBased**, and **ISResMat**) in all three compliant scenarios, with substantial gains in F1-score.

**Keywords:** Data Matching · Compliant Tabular Data · Machine Learning · Meta-Space · Topological Data Analysis

---

## Summary of the experimental evaluation

### Technical environment

- CPU-only experiments (Linux clusters and macOS).  
- Python 3.11 with scikit-learn; optional **XGBoost** and **CatBoost**.  
- Pre-trained text encoders: **MiniLM**, **BERT**, **RoBERTa**, **DistilBERT**, **ALBERT**, **BART**.  
- Large-scale runs on 24 CPU node partitions (Dell C6420, dual Xeon Gold 6136 @ 3 GHz, 192 GB RAM).

### Datasets

We evaluate **Meta Space** on five dataset collections from the **Valentine** benchmark:

- **TPC-DI** — moderately structured transactional data.  
- **ChEMBL** — domain-specific biomedical tables.  
- **WikiData** — structurally heterogeneous tables around music-related entities.  
- **OpenData** — government open data (Canada/USA/UK) with varying vocabularies and units.  
- **Magellan** — curated entity-matching tasks across e-commerce platforms (e.g., Amazon–Google, Walmart–Amazon).

Overall, the paper evaluates **551 pairs of tabular sources** across the three compliant scenarios:

1. Complete–to–Complete  
2. Schema-only–to–Schema-only  
3. Instance-only–to–Instance-only  

For each pair, we generate the meta-space of attribute pairs and train/test the meta-learner accordingly.

### Local organization of experimental artifacts

In the repository, the experiments used in the article are organized under the `Experiments/` directory:

```text
Experiments/
├── Datasets/                     # Raw and preprocessed datasets (Valentine subsets)
├── Results/                      # Aggregated metrics, plots, tables for the paper
├── RQ1_Effectiveness_Baseline/   # Scripts & outputs for RQ1 (baselines vs Meta Space)
├── RQ2_Effeciency/               # Scripts & outputs for RQ2 (runtime, embeddings, etc.)
├── RQ3_Feature_importance/       # Feature importance analyses (per family, per model)
└── RQ4_Feature_Selection/        # Feature-selection experiments & reduced meta-space
````

All the material needed to reproduce the experiments is kept **locally in this tree**.
There is no external Drive link required.

### Research questions (RQ1–RQ4)

The experimental study is structured around:

* **RQ1 — Effectiveness.**
  How well does Meta Space align heterogeneous tabular data in terms of micro-F1, precision, and recall?
  How stable is performance across runs, datasets, embedding models, and classifiers?
  How does it compare to baselines such as **Magneto**, **MagnetoFT**, **MagnetoGPT**, **MagnetoFTGPT**, **Coma/Coma++**, **Cupid**, **Similarity Flooding**, **DistributionBased**, and **ISResMat**?

* **RQ2 — Efficiency.**
  What are the computational costs of:

  * Embedding generation
  * Meta-feature computation
  * Meta-learner training and inference
    How does Meta Space runtime compare to the baselines, and how much does meta-feature computation dominate?

* **RQ3 — Ablation (feature families).**
  What happens if we remove entire families of meta-features (**Classical**, **Spectral**, **Topological**)?
  Which families are most important per scenario (complete, schema-only, instance-only)?

* **RQ4 — Feature selection.**
  Can we automatically select a compact subset of meta-features that preserves effectiveness while improving efficiency and stability?
  How stable are the selected subsets across runs and embedding models?

### Key results (high level)

**Effectiveness (RQ1).**

* Meta Space achieves **state-of-the-art F1** across all three compliant scenarios on Valentine:

  * **Complete–to–Complete:** F1 up to **≈ 0.97 ± 0.08**
  * **Schema-only–to–Schema-only:** F1 up to **≈ 0.98 ± 0.05**
  * **Instance-only–to–Instance-only:** F1 up to **≈ 0.85 ± 0.17**
* Tree-based models (**XGBoost**, **CatBoost**) consistently perform best; distance-based (**KNN**) and neural (**MLP**) are also strong. Simple linear models (Logistic Regression) show high recall but very low precision, confirming that the meta-space is **highly non-linear**.
* Meta Space significantly improves over the best competing baselines:

  * **+32% F1** (relative) in the **complete** scenario
  * **+46% F1** (relative) in the **schema-only** scenario
  * **+19% F1** (relative) in the **instance-only** scenario
    (vs. top Magneto/MagnetoGPT variants, depending on scenario)
* Performance is **stable** across all tested embedding models (ALBERT, BART, BERT, DistilBERT, MiniLM, RoBERTa), indicating that the meta-space representation **reduces dependence** on any particular encoder.

**Efficiency (RQ2).**

* **Meta-feature extraction** is the dominant cost; once features are computed, classifier training/inference are comparatively cheap (e.g. XGBoost training in ~tens of seconds, prediction in <1s per dataset).
* Using **MiniLM-L6-v2** as encoder gives the best trade-off between runtime and effectiveness, thanks to its compact architecture and 384-dim embeddings.
* In absolute terms, Meta Space is slower than very lightweight symbolic matchers (Coma, DistributionBased, SimilarityFlooding), but:

  * It remains competitive with, or **faster than**, several heavy learning-based baselines (Coma++, ISResMat, MagnetoGPT, MagnetoFTGPT).
  * Its higher cost is compensated by clearly superior F1.

**Feature families and ablation (RQ3).**

* **Classical meta-features** (string + embedding distances) provide the **strongest standalone performance**, especially in complete and schema-only scenarios (F1 ≈ 0.97–0.98 by themselves).
* **Spectral** and **Topological** meta-features are **crucial complements**:

  * They substantially help in **schema-only** and **instance-only** settings, where contextual information is weaker.
  * Spectral features (Stable Rank, RankMe, Alpha-ReQ, NESum, SelfCluster, …) encode global structure and anisotropy in embedding spaces.
  * Topological features (persistence entropies, lifetimes, Bottleneck/Wasserstein distances) capture robust geometric patterns that standard distances ignore.
* Combining **Classical + Spectral + Topological** yields a robust, geometry-aware representation that generalizes well across all scenarios.

**Feature selection and compact meta-space (RQ4).**

* Starting from **97 meta-features**, a **backward stepwise regression without correlation filtering (RWoC)** discovers compact, stable subsets:

  * **Complete scenario:** ~44–60 features kept (depending on encoder)
  * **Schema-only scenario:** ~46–59 features
  * **Instance-only scenario:** ~29–39 features
* These subsets **preserve almost all effectiveness**:

  * F1 still reaches **≈ 0.97** in complete and schema-only, and **≈ 0.80–0.85** in instance-only.
* At the same time, they significantly improve runtime:

  * **Median execution time reductions:**

    * **–31%** in the complete scenario
    * **–42%** in the schema-only scenario
    * **–53%** in the instance-only scenario
  * i.e., speedups between **1.4×** and **2.1×** without harming F1.

---

## System requirements

* **Python:** 3.11
* **OS:** macOS or Linux (CPU-only is fine)
* **Poetry** for packaging & virtualenvs
* Optional: **xgboost** and **catboost** (for the extra classifiers)

---

## Installation

```bash
git clone https://github.com/iritsig/MetaSpace.git
cd MetaSpace

poetry env use python3.11

# install project & dependencies
poetry install
```

If you later edit console scripts, re-run:

```bash
poetry install
```

---

## Repository layout

```text
MetaSpace/
├── pyproject.toml
├── src/
│   ├── meta_space/
│   │   ├── pipeline.py              # CLI: MetaSpace (build meta-features)
│   │   ├── embed_utils.py
│   │   ├── golden_tools.py
│   │   └── meta_features/
│   │       ├── features_list.pkl # the complete list of 97 meta-features to compute
│   │       └── meta_features.py # compute meta-features from the 3 families
│   └── Meta_learner/
│       └── train.py                 # CLI: MetaLearn (train/test classifiers)
├── tests/
│   ├── results_meta_space/          # default output dir for MetaSpace features
│   └── meta_learner/                # default output dir for models & metrics
└── Experiments/                     # datasets + article experiments (see above)
```

---

## Usage

### 1) Build the meta-space

You describe the matching task via a **golden JSON** and two CSVs (source/target).
Column names in the JSON must match the CSV headers **exactly**.

**Golden mapping (JSON)**

```json
{
  "dataset": "amazon_google_exp",
  "source": { "name": "S1", "csv": "path/to/source.csv" },
  "target": { "name": "S2", "csv": "path/to/target.csv" },
  "matches": [
    { "source_column": "title",        "target_column": "name" },
    { "source_column": "manufacturer", "target_column": "manufacturer" },
    { "source_column": "price",        "target_column": "price" }
  ]
}
```

Depending on the scenario, the CSVs may contain:

* **Complete:** headers + instance values (standard tables)
* **Schema-only:** only headers are meaningful (values may be empty or ignored)
* **Instance-only:** only values are meaningful (names may be generic or synthetic)

**CLI help**

```bash
poetry run MetaSpace --help
```

```text
Usage: MetaSpace [OPTIONS]

  Run MetaSpace: read CSVs, embed columns, build golden matrix, compute
  meta-features, save results.

Options:
  --dataset TEXT       Dataset name for bookkeeping.  [required]
  --source-csv FILE    Path to source CSV (S1).       [required]
  --target-csv FILE    Path to target CSV (S2).       [required]
  --golden-json FILE   Path to golden JSON.
  --model TEXT         Embedding model alias or HF checkpoint.
                       [default: all-MiniLM-L6-v2]
  --out-dir DIRECTORY  Output directory.
                       [default: tests/results_meta_space]
  --help               Show this message and exit.
```

**Example**

```bash
poetry run MetaSpace \
  --dataset amazon_google_exp \
  --source-csv "Experiments/Datasets/Magellan/Unionable/amazon_google_exp/amazon_google_exp_source.csv" \
  --target-csv "Experiments/Datasets/Magellan/Unionable/amazon_google_exp/amazon_google_exp_target.csv" \
  --golden-json "Experiments/Datasets/Magellan/Unionable/amazon_google_exp/amazon_google_exp_mapping.json" \
  --model all-MiniLM-L6-v2
```

**Outputs**

* `tests/results_meta_space/Meta_Space__{dataset}__{model}.csv`
  (all meta-features + `true_match` labels)
* Optional incremental `inter_*.csv` during long runs

---

### 2) Train & evaluate the meta-learner

**List available classifiers**

```bash
poetry run MetaLearn \
  --features-csv "MetaSpace/tests/results_meta_space/Meta_Space__amazon_google_exp__all-MiniLM-L6-v2.csv" \
  --list-classifiers
```

Example output:

```text
Available classifiers:
 - LogReg
 - RF
 - GBT
 - KNN
 - MLP
 - SVMlin
 - CatBoost
 - XGBoost
```

**CLI help**

```bash
poetry run MetaLearn --help
```

```text
Options:
  --features-csv FILE          Path to MetaSpace features CSV.  [required]
  --classifier TEXT            One or more classifiers (default: RF).
                               Choices printed by --list-classifiers.
  --split [random|by-dataset]  Train/test split strategy.  [default: by-dataset]
  --test-size FLOAT            Hold-out split size if --split=random.
                               [default: 0.2]
  --seed INTEGER               Random seed.  [default: 42]
  --out-dir DIRECTORY          Output directory.
                               [default: tests/meta_learner]
```

**Example**

```bash
poetry run MetaLearn \
  --features-csv "MetaSpace/tests/results_meta_space/Meta_Space__amazon_google_exp__all-MiniLM-L6-v2.csv" \
  --classifier XGBoost \
  --split by-dataset \
  --test-size 0.3 \
  --out-dir "./tests/meta_learner"
```

**Outputs**

* `tests/meta_learner/metrics_global.csv`
  (accuracy, precision, recall, F1, ROC-AUC if available)
* `tests/meta_learner/metrics_per_category.csv`
  (per dataset category when present)
* `tests/meta_learner/predictions__{Classifier}.csv`
  (`y_true`, `y_pred`, `y_prob`)
* `tests/meta_learner/model__{Classifier}.joblib`
  (preprocessor + classifier pipeline)

---

## What’s included

* [Poetry](https://python-poetry.org) for dependency management
* [Click](https://palletsprojects.com/p/click/) for CLIs
* [pytest](https://docs.pytest.org) for tests
* [flake8](https://flake8.pycqa.org) & [mypy](http://mypy-lang.org/) for code quality (if enabled)
* CI workflows (lint/test/build/release) are encouraged via the badges at the top

---

## Everyday activity

### Build / install

```bash
poetry install
```

### Lint & type check

```bash
poetry run flake8 --count --show-source --statistics
poetry run mypy .
```

### Unit tests

```bash
poetry run pytest -v
```

### Docker (optional)

```bash
# build
docker build -t MetaSpace .

# run
docker run --rm -ti MetaSpace
```

---

## Troubleshooting

* **Console scripts not found (`MetaSpace`, `MetaLearn`)**
  Ensure they’re declared under `[project.scripts]` in `pyproject.toml`, then reinstall:

  ```bash
  rm -f poetry.lock
  poetry lock
  poetry install
  ```

* **Poetry/TOML issues (e.g., “<empty>” version)**
  Keep a single `[project]` section and valid versions, then:

  ```bash
  rm -f poetry.lock
  poetry lock
  poetry install
  ```

* **Golden JSON errors**
  Structure must be:

  ```json
  { "matches": [ { "source_column": "...", "target_column": "..." } ] }
  ```

  and names **must match** CSV headers exactly.

---

## License

BSD 3-Clause — see `LICENSE`.

---

