# Semantic Drift Analysis

## Definizione

Semantic drift: cambiamento del significato di una parola tra due periodi temporali.

Metrica: cosine distance tra embeddings allineati

```
drift(w, t1, t2) = 1 - cos(v_w,t1, v_w,t2)

dove cos(a, b) = (a·b) / (||a|| ||b||)
```

Range: [0, 2]
- 0: nessun cambiamento (vettori identici)
- 1: ortogonali (massimo drift plausibile)
- 2: opposizione completa (raro)

## Procrustes Drift

Script: `src/alignment/procrustes_drift.py`

### Local Drift

Confronto tra decenni consecutivi:

```python
for decade_t, decade_t1 in consecutive_pairs:
    # Align t+1 to t
    emb_t1_aligned = align(emb_t1, decade_t, decade_t1)
    
    # Compute drift per word
    for word in vocab:
        v_t = emb_t[word]
        v_t1 = emb_t1_aligned[word]
        drift = 1 - cosine_similarity(v_t, v_t1)
```

Output: `src/alignment/drift_results/drift_{decade}_to_{decade}.json`

Format:
```json
{
  "decade_source": "1980s",
  "decade_target": "1990s",
  "drifts": {
    "computer": 0.234,
    "technology": 0.189,
    "the": 0.003,
    ...
  }
}
```

## Global Drift

Script: `src/alignment/global_drift.py`

### Trajectory Analysis

Evoluzione completa di una parola attraverso tutti i decenni:

```python
def compute_trajectory(word, ref_decade="2010s"):
    trajectory = []
    for decade in DECADES:
        emb_aligned = align_to_reference(emb[decade], ref_decade)
        trajectory.append(emb_aligned[word])
    return trajectory
```

### Cumulative Drift

Drift totale dalla prima all'ultima decade:

```
cumulative_drift(w) = 1 - cos(v_w,1900s, v_w,2010s)
```

Output: `src/alignment/global_results/trajectory_{word}_to_{ref}.json`

Format:
```json
{
  "word": "computer",
  "reference_decade": "2010s",
  "trajectory": {
    "1900s": [0.123, -0.456, ...],
    "1910s": [0.145, -0.423, ...],
    ...
  },
  "cumulative_drift": 0.876
}
```

## Interpretazione

### Drift Range

- drift < 0.1: parola stabile (funzionali, comuni)
- 0.1 < drift < 0.3: drift moderato
- 0.3 < drift < 0.6: drift significativo
- drift > 0.6: cambiamento radicale

### Context Change

Drift alto indica:
- Nuovi contesti d'uso
- Shift semantico (metafora → letterale)
- Specializzazione/generalizzazione
- Cambiamenti culturali/tecnologici

### Examples (attesi)

Alto drift:
- "computer" (persona → macchina)
- "web" (ragnatela → internet)
- "mouse" (animale → device)
- "gay" (allegro → orientamento)

Basso drift:
- "the", "of", "and" (funzioni grammaticali)
- "water", "tree", "house" (oggetti fisici stabili)

## Nearest Neighbors Evolution

Analisi contestuale via nearest neighbors:

```python
for decade in DECADES:
    neighbors = top_k_similar(word, emb[decade], k=10)
```

Mostra shift di contesto:
- 1930s computer: calculator, mathematician, human
- 1990s computer: software, digital, technology

## Statistical Analysis

### Aggregations

- Mean drift per decade pair
- Drift distribution per POS
- Correlation drift vs frequency
- Top-K parole con max drift

### Visualizations

- Scatter plot: frequency vs drift
- Heatmap: drift matrix word × decade
- Line plot: cumulative drift timeline
- Network: semantic clusters evolution

## Output Usage

Results utilizzati per:
1. Visualizzazioni (PCA, t-SNE)
2. Case studies parole specifiche
3. Analisi tematiche (tecnologia, politica, etc)
4. Validazione modello (sanity checks)
