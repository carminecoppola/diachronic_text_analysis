# 📚 Diachronic Text Analysis

Analisi dell'evoluzione semantica delle parole nel tempo (1900-1990) usando word embeddings e Google Books N-grams.

**Stato progetto**: ✅ Fase 3/10 completata (Vocabolario pronto) | 🔜 Prossima: Training embeddings

---

## 🚀 Quick Start per Collaboratori

### Per chi inizia ORA (Fase 4)

**Cosa è già pronto**:
- ✅ Dataset scaricato e processato (766M righe → 113M parole uniche)
- ✅ Vocabolario comune creato (50,001 parole)
- ✅ 10 file decennali pronti per training

**Cosa devi fare**:

1. **Clona e configura ambiente**:
```bash
git clone https://github.com/carminecoppola/diachronic_text_analysis.git
cd diachronic_text_analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Crea il tuo branch di lavoro**:
```bash
# Crea e passa a un nuovo branch con il tuo nome
git checkout -b nome-cognome

# Esempio: git checkout -b mario-rossi
# oppure: git checkout -b feature/dataset-implementation
```

3. **Verifica setup**:
```bash
python src/config.py  # Test configurazione
ls -lh data/processed/*.txt  # Verifica dati (dovrebbero essere presenti)
git branch  # Verifica di essere sul tuo branch (*)
```

4. **Leggi documentazione**:
- [SPIEGAZIONE_PROGETTO.md](docs/SPIEGAZIONE_PROGETTO.md) - Panoramica pipeline
- [DATASET.md](docs/DATASET.md) - Struttura dati
- [VOCABOLARIO.md](docs/VOCABOLARIO.md) - Come funziona vocab

5. **Inizia sviluppo**: Fase 4 - PyTorch Dataset ([vedi sotto](#📋-pipeline-stato-corrente))

---

## 🔀 Workflow Git per Collaboratori

### Ogni collaboratore lavora sul proprio branch

**Mai lavorare direttamente su `main`!** Ogni utente deve creare il proprio branch personale.

#### 1. Prima volta - Crea branch
```bash
# Assicurati di essere su main aggiornato
git checkout main
git pull origin main

# Crea il tuo branch personale
git checkout -b nome-cognome
# oppure con naming più descrittivo: git checkout -b feature/nome-feature
```

#### 2. Durante il lavoro - Salva progressi
```bash
# Verifica modifiche
git status

# Aggiungi file modificati
git add src/dataset.py  # singolo file
# oppure: git add .      # tutti i file

# Commit con messaggio descrittivo
git commit -m "Implementato PyTorch Dataset per CBOW"

# Sincronizza con il tuo branch remoto
git push origin nome-cognome
```

#### 3. Lavoro completato - Richiedi merge
```bash
# Assicurati che tutto sia committato
git status  # Dovrebbe dire "nothing to commit, working tree clean"

# Push finale del branch
git push origin nome-cognome

# A questo punto:
# 1. Vai su GitHub → repository
# 2. Clicca "Compare & pull request"
# 3. Scrivi descrizione del lavoro fatto
# 4. Assegna reviewer
# 5. Attendi approvazione prima del merge su main
```

#### 4. Aggiornare il tuo branch con modifiche da main
```bash
# Se altri hanno fatto modifiche su main mentre lavoravi
git checkout main
git pull origin main
git checkout nome-cognome
git merge main  # Integra modifiche di main nel tuo branch
# Risolvi eventuali conflitti
git push origin nome-cognome
```

### Comandi Git Utili

```bash
# Vedere su quale branch sei
git branch

# Vedere stato modifiche
git status

# Vedere cronologia commit
git log --oneline

# Annullare modifiche non committate
git checkout -- nome-file.py

# Cambiare branch
git checkout nome-branch

# Eliminare branch locale (dopo merge)
git branch -d nome-branch
```

### ⚠️ Regole Importanti

1. **Non fare push su `main` direttamente** - lavora sempre su un branch
2. **Commit frequenti** - salva progressi con messaggi chiari
3. **Pull request per merge** - non mergiare il tuo branch su main autonomamente
4. **Sync regolarmente** - aggiorna il tuo branch da main per evitare conflitti
5. **Testa prima di push** - verifica che il codice funzioni

---

## 🎯 Obiettivo

Studiare il **semantic drift**: come il significato delle parole cambia nel corso del tempo.

### Esempi storici

| Parola | 1900s | 1990s |
|--------|-------|-------|
| **gay** | felice, gioioso | omosessuale |
| **computer** | persona che computa | macchina elettronica |
| **mouse** | animale (topo) | dispositivo input |
| **web** | ragnatela | internet |

### Approccio metodologico

1. **Word embeddings separati per decennio**: un modello CBOW per ogni periodo (1900s, 1910s, ..., 1990s)
2. **Allineamento geometrico**: Orthogonal Procrustes per rendere comparabili gli spazi vettoriali
3. **Metriche quantitative**: cosine similarity, vector displacement, neighbor changes
4. **Visualizzazione**: PCA/t-SNE per tracciare traiettorie semantiche

---

## 📊 Dataset

**Google Books N-grams v3 (2020)**
- Corpus: ~8% di tutti i libri pubblicati
- Lingua: English
- Periodo: **1900-1990** (10 decenni)
- Granularità: **decenni** (non anni singoli)

**Motivazione scelta temporale:**
- Pre-1900: instabilità ortografica
- Post-1990: coverage dataset incompleta
- XX secolo: massimi cambiamenti tecnologici/sociali

⚠️ **I dati NON sono inclusi nel repository**. Vengono scaricati tramite script dedicato.

📖 **Documentazione dettagliata:**
- [DATASET.md](docs/DATASET.md) - Struttura dataset, formati, statistiche
- [SPIEGAZIONE_PROGETTO.md](docs/SPIEGAZIONE_PROGETTO.md) - Metodologia e teoria

---

## 📁 Struttura Progetto

```
diachronic_text_analysis/
├── .venv/                      # Ambiente virtuale (non in Git)
├── data/                       # Dati (non in Git)
│   ├── raw/                    # N-gram filtrati da Google
│   │   ├── 1gram_filtered.tsv
│   │   └── total_counts.txt
│   └── processed/              # Aggregati per decennio
│       ├── 1900s.txt
│       ├── 1910s.txt
│       ├── ...
│       ├── 1990s.txt
│       └── vocab.json          # Vocabolario comune
├── docs/                       # Documentazione
│   ├── DATASET.md              # Struttura dataset dettagliata
│   └── SPIEGAZIONE_PROGETTO.md # Spiegazione metodologia
├── models/                     # Embeddings (non in Git)
│   ├── emb_1900s.pt
│   ├── emb_1910s.pt
│   └── ...
├── plots/                      # Visualizzazioni (non in Git)
├── src/                        # Codice sorgente
│   ├── config.py               # Parametri centrali
│   ├── download_ngrams.py      # FASE 1: Download dataset
│   ├── preprocess.py           # FASE 2: Aggregazione decenni
│   ├── build_vocab.py          # FASE 3: Vocabolario comune
│   ├── dataset.py              # (futuro) PyTorch Dataset
│   ├── model.py                # (futuro) Modello CBOW
│   ├── train.py                # (futuro) Training embeddings
│   ├── align_embeddings.py     # (futuro) Procrustes
│   ├── semantic_drift.py       # (futuro) Metriche drift
│   └── visualize.py            # (futuro) Plot
├── requirements.txt            # Dipendenze Python
├── .gitignore                  # Esclude data/, models/, plots/
└── README.md                   # Questo file
```

---

## 🚀 Setup Iniziale

### 1. Prerequisiti

- Python 3.8+
- pip aggiornato
- ~10GB spazio disco libero (per subset dataset)

### 2. Creazione Ambiente Virtuale

```bash
cd diachronic_text_analysis
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 3. Installazione Dipendenze

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verifica Setup

```bash
python src/config.py
```

Output atteso:
```
✓ Directory create: data, models, plots
📅 Decenni da analizzare: ['1900s', '1910s', ..., '1990s']
📊 Vocabolario: 50000 parole
🧠 Embedding dimension: 300
```

---

## 📋 Pipeline (Stato Corrente)

### ✅ FASE 1-3: Completate

| Fase | Script | Stato | Output | Tempo |
|------|--------|-------|--------|-------|
| 1 | `download_ngrams.py` | ✅ Completato | 766M righe filtrate (14GB) | 4-6h |
| 2 | `preprocess.py` | ✅ Completato | 10 file decenni (3.1GB, 113M parole) | 30-45m |
| 3 | `build_vocab.py` | ✅ Completato | `vocab.json` (50,001 parole) | 5-10m |

**Risultati disponibili**:
- `data/raw/1gram_filtered.tsv` - 766M righe (anni 1900-1990, solo alfabetiche)
- `data/processed/1900s.txt` ... `1990s.txt` - frequenze normalizzate per decennio
- `data/processed/vocab.json` - vocabolario comune (99.98%+ copertura)

📖 **Dettagli**: [DATASET.md](docs/DATASET.md) | [VOCABOLARIO.md](docs/VOCABOLARIO.md)

---

### 🔜 FASE 4-10: Da Implementare

**FASE 4: PyTorch Dataset** (PROSSIMA)
- **File da creare**: `src/dataset.py`
- **Obiettivo**: Carica decenni, genera contesti CBOW, batch iterator
- **Input**: `data/processed/*.txt` + `vocab.json`
- **Output**: PyTorch Dataset class

**FASE 5: Training Embeddings**
- **File da creare**: `src/model.py` (CBOW) + `src/train.py` (loop)
- **Obiettivo**: 10 modelli CBOW (1 per decennio)
- **Output**: `models/emb_1900s.pt` ... `emb_1990s.pt`
- **Tempo stimato**: 2-4h per decennio (GPU)

**FASE 6-10: Alignment, Analisi, Visualizzazione**
- Orthogonal Procrustes alignment
- Metriche semantic drift
- Plot traiettorie PCA/t-SNE

📖 **Dettagli completi**: [SPIEGAZIONE_PROGETTO.md](docs/SPIEGAZIONE_PROGETTO.md)

---

## 📊 Configurazione

Tutti i parametri in [`src/config.py`](src/config.py):

```python
# Periodo
START_YEAR = 1900
END_YEAR = 1990

# Dataset  
NUM_FILES_TO_DOWNLOAD = 24  # 24=completo, 3=test
ONLY_ALPHABETIC = True

# Vocabolario
VOCAB_SIZE = 50000

# Embeddings (per fasi future)
EMBEDDING_DIM = 300
CONTEXT_WINDOW = 5
BATCH_SIZE = 512
EPOCHS = 10
```

---

## 🔧 Comandi Utili

```bash
# Esplora dati
head -20 data/processed/1950s.txt
cat data/processed/vocab_stats.txt

# Conta parole
wc -l data/processed/*.txt

# Verifica dimensioni
du -sh data/raw/ data/processed/

# Test configurazione
python src/config.py
```

---

## 🤝 Suddivisione Lavoro (Suggerita)

| Persona | Fase | Task | Tempo stimato |
|---------|------|------|---------------|
| A | 4 | PyTorch Dataset | 1-2 giorni |
| B | 5a | CBOW Model | 1-2 giorni |
| C | 5b | Training Loop | 2-3 giorni |
| D | 6-7 | Alignment + Metrics | 3-4 giorni |
| Tutti | 8-10 | Analisi + Plot | 2-3 giorni |

**Total**: ~1-2 settimane con collaborazione parallela

---

**Output:**
- `data/processed/vocab.json`: `{word: index}`
- `data/processed/vocab_stats.txt`: statistiche copertura

---

### 🔜 FASE 4: Training Embeddings (Da Implementare)

**Script futuro:** `train.py`

Addestrare modello CBOW separato per ogni decennio.

**Parametri chiave** (in [`config.py`](src/config.py)):
- `EMBEDDING_DIM = 300`
- `CONTEXT_WINDOW = 5`
- `EPOCHS = 10`

**Output previsto:** `models/emb_1900s.pt`, ..., `models/emb_1990s.pt`

---

### 🔜 FASE 5-7: Allineamento, Analisi, Visualizzazione (Da Implementare)

Pipeline completa da sviluppare dopo training embeddings.

---

## 📊 Configurazione

Tutti i parametri sono centralizzati in [`src/config.py`](src/config.py):

```python
# Periodo temporale
START_YEAR = 1900
END_YEAR = 1990

# Vocabolario
VOCAB_SIZE = 50000

# Embeddings
EMBEDDING_DIM = 300
CONTEXT_WINDOW = 5

# Training
BATCH_SIZE = 512
LEARNING_RATE = 0.025
EPOCHS = 10
```

Modificare qui per cambiare comportamento del progetto.

---

## 🧪 Verifica Installazione

Dopo setup, eseguire:

```bash
# Test config
python src/config.py

# Verifica che tqdm funzioni (usato nei download)
python -c "from tqdm import tqdm; import time; [time.sleep(0.01) for _ in tqdm(range(100))]"
```

---

## 📝 Note sul Dataset

### Perché solo 3 file di default?

Il dataset completo Google N-grams è **enorme** (centinaia di GB). Per:
- **Test/sviluppo**: 3 file (~5-10GB) sono sufficienti
- **Progetto universitario**: subset rappresentativo è accettabile
- **Ricerca seria**: aumentare `NUM_FILES_TO_DOWNLOAD = 24` in [`config.py`](src/config.py)

### Alternativa: Dati Pre-filtrati

Se il download è troppo lento, è possibile:
1. Preparare dataset campionato offline
2. Metterlo direttamente in `data/raw/1gram_filtered.tsv`
3. Saltare FASE 1 e partire da FASE 2

---

## 📚 Riferimenti

- **Hamilton et al. (2016)**: "Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change"
- **Mikolov et al. (2013)**: "Efficient Estimation of Word Representations in Vector Space" (Word2Vec)

---

## 📄 Licenza

Progetto universitario - uso educativo.

---

**Ultimo aggiornamento**: 5 Gennaio 2026  
**Fase corrente**: 3/10 (Vocabolario Completato)  
**Prossima fase**: FASE 4 - PyTorch Dataset
