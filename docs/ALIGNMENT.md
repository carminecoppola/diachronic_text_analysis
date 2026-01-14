# Procrustes Alignment

## Problema

Embedding spaces indipendenti hanno rotazioni arbitrarie:
- Non comparabili direttamente
- Cosine similarity misleading tra decenni
- Serve allineamento a reference space comune

## Orthogonal Procrustes

### Teoria

Data matrice embedding source W_s e target W_t, trova rotazione ortogonale R che minimizza:

```
R* = argmin_R ||R·W_s - W_t||²_F

subject to: R^T·R = I  (preserva angoli e distanze)
```

Soluzione closed-form via SVD:
```
U, Σ, V^T = SVD(W_t^T · W_s)
R = U · V^T
```

### Anchor Words

Parole stabili semanticamente usate per alignment:
- Alta frequenza in entrambi i decenni
- Significato presumibilmente invariato
- Evita parole con alto semantic drift

Selection criteria (implementati):
1. Frequenza >= threshold in entrambi i decenni
2. Top K parole per frequenza media
3. Esclusione stopwords opzionale
4. Esclusione parole target di analisi

## Implementazione

Script: `src/alignment/procrustes_drift.py`

### Compute Frequencies

Prerequisito: frequenze parole per decade
- Input: modelli trained
- Output: `alignment/{3gram,5gram}/frequencies/freq_{decade}.json`
- Script: `src/alignment/compute_frequencies.py`

### Alignment Pipeline

```python
for decade_t, decade_t1 in consecutive_decades:
    # 1. Load embeddings
    emb_t = load_embeddings(decade_t)
    emb_t1 = load_embeddings(decade_t1)
    
    # 2. Select anchors
    anchors = select_anchors(freq_t, freq_t1, top_k=5000)
    
    # 3. Extract anchor embeddings
    W_t = emb_t[anchors]
    W_t1 = emb_t1[anchors]
    
    # 4. Compute Procrustes
    R = orthogonal_procrustes(W_t1, W_t)
    
    # 5. Align full space
    emb_t1_aligned = emb_t1 @ R
    
    # 6. Save transform
    save(R, f"alignment_{decade_t}_to_{decade_t1}.pt")
```

### Alignment Chain

Reference: ultimo decennio (2010s)

```
1900s_orig → R1 → 1900s_aligned_to_1910s
1910s_orig → R2 → 1910s_aligned_to_1920s
...
2000s_orig → R11 → 2000s_aligned_to_2010s
2010s_orig → [identity] → 2010s (reference)
```

Per allineare tutti a 2010s:
```
emb_1900s_to_2010s = emb_1900s @ R1 @ R2 @ ... @ R11
```

## Output

Transform matrices: `src/alignment/transforms/`
- `alignment_1900s_to_1910s.pt`
- `alignment_1910s_to_1920s.pt`
- ...
- `alignment_2000s_to_2010s.pt`

Format:
```python
torch.save({
    'R': rotation_matrix,           # (300, 300)
    'decade_source': '1990s',
    'decade_target': '2000s',
    'num_anchors': 5000,
    'anchor_words': [...]
}, path)
```

## Validazione

Check post-alignment:
1. R ortogonalità: `||R^T·R - I||_F < ε`
2. Anchor stability: cosine similarity anchors ~1
3. Embedding norms preserved: `||R·v|| = ||v||`

## Considerations

### Anchor Selection Tradeoff

Too few anchors:
- Alignment unstable
- Overfitting to specific words

Too many anchors:
- Include drifting words
- Suboptimal alignment

Sweet spot: ~5000 parole (10% vocab)

### Reference Choice

2010s scelto come reference:
- Decennio più recente
- Massima qualità corpus
- Interpretabilità risultati rispetto a presente
