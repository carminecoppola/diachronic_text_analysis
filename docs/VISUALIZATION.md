# Visualizzazioni

## Overview

Script: `src/visualization/visualize_drift.py`

Moduli:
- `alignment_utils.py`: Caricamento embeddings e alignment
- `drift_io.py`: I/O vocabolario e matrici
- `neighbors.py`: K-nearest neighbors
- `plotting.py`: Grafici PCA e t-SNE

## PCA (Principal Component Analysis)

Riduzione dimensionalità lineare: 300D → 2D

### Usage

```bash
python src/visualization/visualize_drift.py \
  --words computer technology internet \
  --pca \
  --ref 2010s \
  --connect
```

### Parameters

- `--words`: Lista parole da visualizzare
- `--ref`: Decade di riferimento per alignment
- `--connect`: Connette punti di stessa parola con linee
- `--from_decade`, `--to_decade`: Limita range temporale

### Output

Plot con:
- Punti colorati per parola
- Marker diversi per decade
- Traiettorie temporali se --connect
- Variance explained dalle prime 2 PC

File: `plots/pca_{words}_{from}_{to}.png`

## t-SNE (t-Distributed Stochastic Neighbor Embedding)

Riduzione dimensionalità non-lineare: preserva vicinanze locali

### Usage

```bash
python src/visualization/visualize_drift.py \
  --words computer \
  --tsne \
  --perplexity 5
```

### Parameters

- `--perplexity`: Controlla local vs global structure (default: auto)
- `--label_dx`, `--label_dy`: Offset label per leggibilità

### Output

Plot con layout che preserva clusters semantici

File: `plots/tsne_{words}_{from}_{to}.png`

### Perplexity Choice

- Basso (5-10): enfatizza local clusters
- Alto (30-50): struttura più globale
- Default auto: min(30, n_points/5)

## Nearest Neighbors

Analisi contestuale parola in decade specifico

### Usage

```bash
python src/visualization/visualize_drift.py \
  --words computer \
  --neighbors \
  --topk 10 \
  --ref 1950s
```

### Output

CSV: `results/neighbors_{word}_{ref}.csv`

Format:
```csv
rank,word,similarity
1,calculator,0.876
2,machine,0.834
3,electronic,0.812
...
```

## Window Analysis

Visualizza subset decenni:

```bash
python src/visualization/visualize_drift.py \
  --words computer \
  --pca \
  --from_decade 1950s \
  --to_decade 2010s
```

Utile per:
- Focus su periodi critici
- Ridurre clutter visivo
- Analisi fine-grained

## Interpretazione

### PCA

Assi principali catturano:
- PC1: Spesso temporalità (1900s vs 2000s)
- PC2: Semantic dimensions principali
- Clustering: Parole semanticamente simili

### t-SNE

Preserva:
- Vicinanze locali (co-occorrenze)
- Clusters semantici
- Non preserva distanze globali

### Traiettorie

Lunghezza traiettoria ∝ semantic drift
- Lineare: drift graduale
- Non-lineare: cambiamenti bruschi
- Loops: significati ciclici

## Advanced Usage

### Multiple Words Comparison

```bash
python src/visualization/visualize_drift.py \
  --words computer telephone radio television internet \
  --tsne \
  --connect
```

Mostra evoluzione tecnologie comunicazione

### Theme Analysis

Creare set tematici:

Technology:
```bash
--words computer software hardware digital internet web
```

Politics:
```bash
--words democracy communism capitalism liberal conservative
```

## Output Management

Plots salvati in: `plots/`
Results salvati in: `results/`

Naming convention:
- `pca_word1_word2_1900s_2010s.png`
- `tsne_computer_all.png`
- `neighbors_computer_1990s.csv`

## Customization

Plot styling in `plotting.py`:
- Color maps per parola
- Marker shapes per decade
- Figure size e DPI
- Font sizes e labels

Modificabile per:
- Publication-ready figures
- Presentazioni
- Analisi interattive
