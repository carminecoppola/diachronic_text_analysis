# Analisi Diacronica con Word Embeddings

## Obiettivo

**Studiare l'evoluzione del significato di parole** (es. "gay", "computer") attraverso il XX secolo, usando:
- **Word embeddings** (CBOW)
- **Allineamenti** (Orthogonal Procrustes)
- **Metriche** (cosine similarity, semantic shift)

---

## Pipeline (10 Fasi)

```
FASE 1: Download Dataset
   ├─ Scarica Google Books 3-grams v3 (2020) da Google Cloud Storage
   ├─ URL: http://storage.googleapis.com/books/ngrams/books/20200217/eng/3-XXXXX-of-06881.gz
   ├─ File scaricati: 3-02000 to 3-02019 (20 file, range parole comuni)
   ├─ Rimozione POS tags (computer_NOUN → computer)
   └─ Output: data/raw/3gram_filtered.tsv (15GB, ~680M righe)

FASE 2: Preprocessing
   ├─ Filtra anni 1900-2019 (120 anni)
   ├─ Suddividi in 12 decenni (1900s-2010s)
   ├─ Split 3-gram in (word1, word2, word3)
   ├─ Aggrega frequenze per decade
   └─ Output: 12 file data/processed/{decade}.txt

FASE 3: Costruzione Vocabolario
   ├─ Aggrega frequenze da tutti i decenni
   ├─ Seleziona parole più frequenti
   └─ Output: data/processed/vocab.json

FASE 4: PyTorch Dataset
   ├─ Usa 3-gram per contesto CBOW: TARGET = word2, CONTEXT = [word1, word3]
   ├─ Negative sampling per training efficiente
   └─ Output: dataset.py con CBOWNgramDataset

FASE 5: Training CBOW
   ├─ 12 modelli indipendenti (uno per decennio 1900s-2010s)
   ├─ Hyperparams: dim=300, context_size=1 (parola prima+dopo), epochs=5-10
   └─ Output: models/cbow_1900s.pt ... models/cbow_2010s.pt

FASE 6: Allineamento Embeddings
   ├─ Orthogonal Procrustes per allineare decenni consecutivi
   ├─ Baseline: 1900s (riferimento fisso)
   └─ Output: models/aligned_*.npy

FASE 7: Calcolo Metriche
   ├─ Cosine similarity tra decenni
   ├─ Semantic shift (distanza cumulativa)
   └─ Output: results/metrics.csv

FASE 8: Visualizzazioni
   ├─ Plot evoluzione singole parole
   ├─ Heatmap shift temporale
   ├─ t-SNE spazio semantico
   └─ Output: plots/*.png

FASE 9: Nearest Neighbors Analysis
   ├─ Top-10 parole simili per decennio
   ├─ Shift vicinato semantico
   └─ Output: results/neighbors.csv

FASE 10: Analisi Tematiche
   ├─ Cluster parole per topic
   ├─ Evoluzione temi nel tempo
   └─ Output: results/themes.csv
```

---

## Stato Avanzamento

| Fase | Status | Note |
|------|--------|------|
| 1. Download | ✅ COMPLETATA | 551M righe 3-gram, 15GB |
| 2. Preprocessing | ✅ COMPLETATA | 12 decenni, 74K-134K parole/decennio |
| 3. Vocabolario | ✅ COMPLETATA | 50,002 parole comuni |
| 4. PyTorch Dataset | 🔄 PROSSIMA | Test CBOWNgramDataset con 3-gram |
| 5. Training CBOW | IN ATTESA | 12 modelli (1900s-2010s) |
| 6. Allineamento | - | |
| 7-10 | - | |

---

## Base Teorica

### Word Embeddings (CBOW)

**Continuous Bag-of-Words** impara rappresentazioni dense usando 3-gram:
```
3-gram: "computer is running"
Context: ["computer", "running"]  (parola prima + parola dopo)
Target: "is"  (parola centrale)

Obiettivo: predire parola centrale dal suo contesto immediato
→ Rete neurale impara embedding che cattura co-occorrenze reali
```

**Vantaggi con 3-gram**:
- Contesto reale da Google Books (non finestre artificiali)
- Cattura relazioni sintattiche e semantiche naturali
- Parole con contesti simili → embeddings vicini
- Dimensione tipica: 100-300 (qui 300)
- Preserva struttura linguistica ("computer is running" vs "running is computer")

### Allineamento (Orthogonal Procrustes)

Problema: embeddings diversi per ogni decennio (rotazioni casuali).

**Soluzione**: Trova matrice rotazione $R$ che minimizza:

$$
R^* = \arg\min_R ||R \cdot W_{t+1} - W_t||
$$

con vincolo $R^T R = I$ (ortogonalità preserva angoli).

**Risultato**: Embeddings allineati comparabili nel tempo.

### Semantic Shift

**Definizione**: Quanto cambia significato parola tra decenni.

**Formula**:
$$
shift(w, t_1, t_2) = 1 - \cos(v_{w,t_1}, v_{w,t_2})
$$

dove $v_{w,t}$ = embedding parola $w$ al tempo $t$.

**Interpretazione**:
- shift = 0: significato invariato
- shift = 1: significato completamente diverso

**Esempio atteso**:
- "computer": alto shift (persona → macchina)
- "gay": alto shift (felice → omosessuale)
- "the": basso shift (articolo stabile)

---

## Struttura File

```
diachronic_text_analysis/
├── data/
│   ├── raw/                          # Dataset originale
│   │   └── google-books-ngrams-1grams.parquet (14GB)
│   └── processed/                    # Dati processati
│       ├── 1900s.txt ... 1990s.txt  # Decenni (3.1GB)
│       └── vocab.json                # Vocabolario (990KB)
├── models/                           # Modelli salvati
│   ├── cbow_1900s.pt ... cbow_1990s.pt
│   └── aligned_*.npy
├── results/                          # Output analisi
│   ├── metrics.csv
│   ├── neighbors.csv
│   └── themes.csv
├── plots/                            # Visualizzazioni
├── src/                              # Codice
│   ├── config.py                     # Parametri centrali
│   ├── download_ngrams.py            # Fase 1
│   ├── preprocess.py                 # Fase 2
│   ├── build_vocab.py                # Fase 3
│   ├── dataset.py                    # Fase 4
│   ├── train.py                      # Fase 5
│   ├── align.py                      # Fase 6
│   ├── metrics.py                    # Fase 7
│   └── visualize.py                  # Fase 8-10
└── docs/                             # Documentazione
    ├── DATASET.md
    ├── VOCABOLARIO.md
    └── SPIEGAZIONE_PROGETTO.md
```

---

## Configurazione Centralizzata

**File**: [`src/config.py`](../src/config.py)

```python
# Dataset
GOOGLE_BOOKS_DATASET = "seansolari/google-books-ngrams-1grams"
YEAR_RANGE = (1900, 1999)
DECADE_SIZE = 10

# Vocabolario
VOCAB_SIZE = 50000
MIN_VOCAB_FREQ = 100

# CBOW Hyperparameters
EMBEDDING_DIM = 300
WINDOW_SIZE = 5
NEGATIVE_SAMPLES = 5
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
EPOCHS = 5

# Allineamento
ALIGNMENT_METHOD = "procrustes"
BASELINE_DECADE = "1900s"

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
```

**Vantaggio**: Modifica parametri in un solo posto.

---

## Esempi di Utilizzo

### Training CBOW (Fase 5)

```bash
source .venv/bin/activate
python src/train.py
```

**Output atteso**:
```
Training CBOW for decade: 1900s
Epoch 1/5: loss=4.123
Epoch 2/5: loss=3.567
...
Saved: models/cbow_1900s.pt
```

### Allineamento (Fase 6)

```bash
python src/align.py
```

**Output**:
```
Aligning decade 1910s to baseline 1900s...
Procrustes error: 0.0234
Saved: models/aligned_1910s.npy
```

### Calcolo Semantic Shift (Fase 7)

```python
from src.metrics import compute_shift

target_words = ["computer", "gay", "internet"]
shifts = compute_shift(target_words, decade1="1950s", decade2="1990s")

for word, shift in zip(target_words, shifts):
    print(f"{word}: {shift:.4f}")
```

**Output**:
```
computer: 0.7823  # Alto shift
gay: 0.6912       # Alto shift
the: 0.0145       # Basso shift
```

---

## Prossimi Passi

1. **Fase 4**: Implementare `dataset.py` con:
   - Generazione context windows
   - Negative sampling
   - DataLoader PyTorch

2. **Fase 5**: Training loop:
   - 10 modelli CBOW indipendenti
   - Checkpoint ogni epoch
   - Validation loss

3. **Fase 6**: Orthogonal Procrustes:
   - Allineamento sequenziale (1900s → 1910s → ...)
   - Test stabilità (multiple runs)

4. **Fasi 7-10**: Analisi e visualizzazioni.

---

## Risorse

### Papers Principali

1. **Hamilton et al. (2016)**: "Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change"
   - Orthogonal Procrustes per allineamento
   - Analisi statistical significance

2. **Mikolov et al. (2013)**: "Efficient Estimation of Word Representations"
   - Architettura CBOW e Skip-gram
   - Negative sampling

### Tool & Framework

- **PyTorch**: Training embeddings
- **scikit-learn**: Orthogonal Procrustes, t-SNE
- **matplotlib/seaborn**: Visualizzazioni

---

## FAQ

**Q: Perché 10 decenni invece di 100 anni separati?**
A: Trade-off dati/granularità. Decenni hanno abbastanza testo per training stabile.

**Q: Perché CBOW invece di GloVe o transformers?**
A: CBOW è:
- Veloce da trainare
- Interpretabile
- Standard per diachronic analysis

**Q: Come valutare qualità embeddings?**
A: Intrinsic (cosine similarity word pairs) + extrinsic (word analogy tasks).

**Q: Quanto tempo per training?**
A: Dipende da GPU. Stima: 1-2 ore per decennio su GPU moderna (NVIDIA RTX 3090).

---

**Creato**: 5 Gennaio 2026
**Fase Corrente**: 3/10 (Vocabolario completato)
**Prossima Fase**: PyTorch Dataset
