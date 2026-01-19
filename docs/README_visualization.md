# Diachronic Visualization Pipeline – Technical and Practical Documentation

## Overview

This document describes **the full technical and practical pipeline implemented for the visualization and analysis of diachronic word embeddings**.
It is intended to serve as a **standalone README** for the visualization part of the project,n, explaining:

- What each component does
- How the pipeline is structured
- How to generate each type of output (PCA, t-SNE, diachronic associations, dispersion)
- How to interpret the resulting graphs
- What is required to run everything correctly

This documentation refers **only to a single-model setup** (no model comparison), which is the configuration actually used.

---

## Conceptual Goal

The goal of this part of the project is **not** to retrain embeddings, but to:

1. **Align embeddings from different decades into a common vector space**
2. **Visualize semantic evolution (drift) of words over time**
3. **Quantify stability and dispersion of semantic neighborhoods**
4. **Provide interpretable plots suitable for analysis and reporting**

---

## Directory Structure (Relevant Part)

```text
src/visualization/
├── drift_io.py
├── alignment_utils.py
├── neighbors.py
├── visualize_drift.py
├── plotting.py
├── diachronic_associations.py
├── plot_diachronic_associations_from_results.py
├── plot_stability_dispersion.py
```

Other required paths (defined in `src/config.py`):

```text
MODELS_DIR/           # contains emb_<decade>.pt
ALIGNMENT_TRANSFORMS_DIR/
VOCAB_FILE            # JSON word -> index
results/              # generated CSV outputs
plots/                # generated PNG plots
```

---

## Prerequisites

### Python Environment

Required packages:

```bash
pip install numpy torch scikit-learn matplotlib adjustText
```

### Required Data

You must already have:

- Trained embeddings: `emb_1900s.pt`, `emb_1910s.pt`, ...
- Alignment files: `alignment_1900s_to_1910s.pt`, ...
- Vocabulary file: JSON mapping words to indices

This pipeline **does not train embeddings**.

---

## Alignment Logic

### Why Alignment Is Required

Each decade embedding is trained independently.
Therefore, vectors from different decades live in **different coordinate systems**.

To compare them meaningfully, embeddings are aligned using **orthogonal Procrustes alignment**.

### How Alignment Works

Each alignment file encodes:

```
E_to = (E_from - mu_X) @ R + mu_Y
```

Where:
- `R` is an orthogonal rotation
- `mu_X`, `mu_Y` are centroids

### Bidirectional Alignment

The function `align_embeddings_to_ref()` supports:

- Forward alignment (older → newer)
- Backward alignment (newer → older, using inverse transform)

This allows **any decade to be used as reference**.

---

## Neighbor Extraction

### Purpose

For each target word and decade, we extract the **top-k nearest neighbors** using cosine similarity.

This is the basis for:
- Diachronic association plots
- Dispersion / stability metrics

### Output Format

Per-decade CSV files:

```csv
word,decade,rank,neighbor,cosine
computer,1950s,1,calculator,0.82
...
```

Generated automatically with:

```bash
--neighbors --topk K
```

---

## visualize_drift.py (Main Orchestrator)

This script coordinates the entire pipeline.

### What It Does

1. Loads embeddings for each decade
2. Aligns them to a reference decade
3. Optionally writes neighbors CSVs
4. Builds word trajectories
5. Produces PCA and/or t-SNE plots

### Basic Usage

```bash
python -m src.visualization.visualize_drift   --words dog   --ref 2010s   --pca --tsne   --connect   --background_n 2000
```

### Important Options

| Option | Meaning |
|------|--------|
| `--words` | Target words to track |
| `--ref` | Reference decade |
| `--pca` | Enable PCA |
| `--tsne` | Enable t-SNE |
| `--neighbors` | Write neighbors CSVs |
| `--background_n` | Background size for stable projections |
| `--connect` | Draw temporal trajectories |

---

## PCA Visualization

### What PCA Shows

- Linear projection of embeddings into 2D
- Preserves **global variance**
- Useful to visualize **overall semantic trajectory**

### What PCA Does NOT Show

- It does **not** prove semantic drift
- Low explained variance means interpretation must be cautious

### Why Background Is Used

PCA is fitted on a **large background vocabulary sample**, not only on trajectory points.
This stabilizes axes and avoids arbitrary rotations.

### Interpretation

- Smooth movement → gradual semantic shift
- Tight clusters → semantic stability
- Explained variance reported in title

---

## t-SNE Visualization

### What t-SNE Shows

- Local neighborhood structure
- Non-linear projection

### Important Caveat

t-SNE **distorts global distances**.
It is **illustrative only**, not quantitative.

### How It Is Used Here

- t-SNE is computed on (background + trajectory)
- Only trajectory points are plotted
- Perplexity is automatically constrained

---

## Diachronic Associations (Word-Level)

### Purpose

To answer:
> *Which words are most similar to the target word at each decade?*

### Script

```bash
python -m src.visualization.plot_diachronic_associations_from_results   --word computer   --topk 6   --from_decade 1950s   --to_decade 2010s
```

### Output

- Scatter plot:
  - X-axis: Year
  - Y-axis: Cosine similarity
  - Labels: Neighbor words

### Interpretation

- Changes in neighbors indicate **semantic shift**
- Early decades may be unreliable for rare words

This is the **most interpretable qualitative plot**.

---

## Stability and Dispersion Analysis

### Purpose

To quantify how **compact or dispersed** a word’s semantic neighborhood is.

### Metrics

| Metric | Meaning |
|------|--------|
| `mean` | Average similarity (stability) |
| `std` | Dispersion of neighbors |
| `gap` | max − min similarity |
| `mean_top3` | Robust stability |

### Example

```bash
python -m src.visualization.plot_stability_dispersion   --words computer   --metric std   --topk 10   --from_decade 1930s   --to_decade 2010s
```

### Interpretation

- Low std → compact neighborhood
- High std → heterogeneous neighborhood
- Early zeros usually indicate **data sparsity**, not stability

This is a **quantitative complement** to diachronic associations.

---

## Interpreting Results Correctly

### What Proves Semantic Drift

- Changing neighbors over time
- Shifts in cosine similarity patterns

### What Is Only Illustrative

- PCA
- t-SNE

### Recommended Combination in a Report

1. Diachronic associations (main figure)
2. Stability/dispersion curve (supporting evidence)
3. PCA or t-SNE (illustration)

---

## Common Pitfalls

- Interpreting early decades for rare words
- Treating PCA/t-SNE as quantitative evidence
- Ignoring alignment assumptions

All these issues are explicitly handled or documented in this pipeline.

---

## Summary

This visualization pipeline:

- Is alignment-correct
- Is reproducible
- Separates qualitative and quantitative analysis
- Is suitable for academic reporting

It fulfills the visualization and analysis goals of the project in a **methodologically sound** way.
