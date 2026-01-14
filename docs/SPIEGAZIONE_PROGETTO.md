# Analisi Diacronica con Word Embeddings

## Obiettivo

Studiare evoluzione semantica delle parole nel XX-XXI secolo usando word embeddings (CBOW) allenati su Google Books N-grams.

## Pipeline

FASE 1: Download
- Scarica Google Books 3/5-grams (v3 2020)
- Output: `data/raw/{3gram,5gram}/`

FASE 2: Preprocessing  
- Aggrega n-gram per decennio (1900s-2010s)
- Mantiene n-gram completi per preservare contesto
- Output: `data/processed/{decade}.txt`

FASE 3: Vocabolario
- Estrae parole singole
- Top 50K più frequenti
- Output: `data/processed/vocab.json`

FASE 4: PyTorch Dataset
- CBOW con contesto reale
- TARGET=word_center, CONTEXT=words_around
- Output: `src/data/dataset.py`

FASE 5: Training CBOW
- 12 modelli (uno per decennio)
- Output: `models/{3gram,5gram}/cbow_{decade}.pt`

FASE 6: Allineamento
- Orthogonal Procrustes tra decenni consecutivi
- Output: `src/alignment/transforms/`

FASE 7: Drift Analysis
- Local drift (decade-to-decade)
- Global drift (trajectories)
- Output: `src/alignment/drift_results/`

FASE 8: Visualizzazioni
- PCA, t-SNE, nearest neighbors
- Output: `plots/`, `results/`

## N-gram e Contesto

File processati mantengono n-gram completi (es. "view of the"), non parole singole.

Vantaggi:
- Contesto reale da corpus
- Relazioni sintattiche/semantiche naturali
- Parole con contesti simili → embeddings vicini

## Allineamento (Orthogonal Procrustes)

Problema: embeddings diversi per ogni decennio (rotazioni casuali).

Soluzione: trova matrice rotazione R che minimizza:
```
R* = argmin_R ||R·W_t+1 - W_t||²
con R^T·R = I (preserva angoli)
```

Soluzione closed-form via SVD:
```
U, Σ, V^T = SVD(W_t^T · W_t+1)
R = U · V^T
```

Anchor words: parole stabili semanticamente per alignment.

## Semantic Shift

Definizione: cambiamento significato parola tra periodi.

Formula:
```
shift(w, t1, t2) = 1 - cos(v_w,t1, v_w,t2)
```

Interpretazione:
- shift = 0: invariato
- shift = 1: completamente diverso

Esempi attesi:
- "computer": alto shift (persona → macchina)
- "gay": alto shift (felice → orientamento)
- "the": basso shift (articolo stabile)

## File Structure

```
data/
  raw/{3gram,5gram}/           Dataset Google Books
  processed/{3gram,5gram}/     Aggregati per decennio + vocab
models/{3gram,5gram}/          CBOW trained per decennio
src/
  data/                        Download, preprocessing, vocab, dataset
  models/                      CBOW architecture, training
  alignment/                   Procrustes, drift computation
  visualization/               PCA, t-SNE, neighbors
alignment/{3gram,5gram}/       Frequencies, transforms
plots/                         Output grafici
```

## Parametri Chiave

config.py:
- START_YEAR, END_YEAR: 1900, 2019
- NGRAM_TYPE: 3 o 5
- VOCAB_SIZE: 50001
- EMBEDDING_DIM: 300
- CONTEXT_WINDOW: 1 (3gram) o 2 (5gram)
- BATCH_SIZE: 512
- EPOCHS: 3

## Utilizzo

```bash
# Preprocessing
python src/data/preprocess.py
python src/data/build_vocab.py

# Training
sbatch run_training.sbatch

# Alignment e drift
python src/alignment/compute_frequencies.py
python src/alignment/procrustes_drift.py
python src/alignment/global_drift.py

# Visualizzazioni
python src/visualization/visualize_drift.py --words computer --pca
```

## References

- Hamilton et al. (2016): Diachronic Word Embeddings
- Mikolov et al. (2013): Word2Vec (CBOW)
