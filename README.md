# 📚 Diachronic Text Analysis: Studying Language Evolution Over Time

## 🎯 Cosa fa questo progetto?

Questo progetto studia come **il significato delle parole cambia nel tempo** (fenomeno chiamato **semantic drift**).

### Esempi concreti:

| Parola | Significato 1900s | Significato 1990s+ |
|--------|-------------------|-------------------|
| **gay** | felice, gioioso | omosessuale |
| **cell** | cellula biologica | telefono cellulare |
| **mouse** | animale (topo) | dispositivo computer |
| **web** | ragnatela | internet, world wide web |

### Come lo fa?

1. **Addestra word embeddings** (vettori numerici) per ogni periodo temporale
2. **Allinea gli spazi vettoriali** usando Orthogonal Procrustes
3. **Calcola quanto ogni parola è cambiata** usando distanza coseno
4. **Visualizza i risultati** con grafici t-SNE/PCA

**Framework**: PyTorch per deep learning + scipy per allineamento matematico.

---

## Struttura del Progetto

```
diachronic_text_analysis/
├── .venv/                      # Ambiente virtuale Python (non versionato)
├── data/
│   ├── raw/                    # Dati grezzi organizzati per periodo
│   └── processed/              # Dati preprocessati (tokenizzati, normalizzati)
├── models/                     # Pesi dei modelli di embedding per periodo
├── alignment/                  # Matrici di trasformazione Procrustes
├── plots/                      # Visualizzazioni (t-SNE, PCA, grafici drift)
├── src/                        # Codice sorgente
│   ├── __init__.py
│   ├── config.py               # Configurazioni centrali del progetto
│   ├── preprocess.py           # Pipeline di preprocessing del testo
│   ├── build_vocab.py          # Costruzione vocabolario
│   ├── dataset.py              # Dataset e DataLoader PyTorch
│   ├── models/
│   │   ├── __init__.py
│   │   └── word_embedding.py   # Modelli CBOW/Skip-gram
│   ├── train_embeddings.py     # Training embeddings per periodo
│   ├── alignment.py            # Allineamento Orthogonal Procrustes
│   ├── semantic_drift.py       # Calcolo metriche di drift
│   └── visualize_drift.py      # Generazione visualizzazioni
├── notebooks/                  # Notebook Jupyter per analisi esplorative
├── requirements.txt            # Dipendenze del progetto
└── README.md                   # Questo file
```

---

## Setup Ambiente

### 1. Prerequisiti
- Python 3.8 o superiore
- pip aggiornato

### 2. Creazione Ambiente Virtuale

**Linux/macOS:**
```bash
cd diachronic_text_analysis
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```cmd
cd diachronic_text_analysis
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Installazione Dipendenze

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Risorse NLP (eseguire dopo installazione)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

Per spaCy (modello inglese):
```bash
python -m spacy download en_core_web_sm
```

---

## Pipeline del Progetto

### Fase 1: Acquisizione Dati
- Scaricare corpus storico con metadati temporali
- Organizzare in `data/raw/` per periodo

### Fase 2: Preprocessing
```bash
python src/preprocess.py
```
- Normalizzazione, tokenizzazione, lemmatizzazione
- Output: file per periodo in `data/processed/`

### Fase 3: Costruzione Vocabolario
```bash
python src/build_vocab.py
```
- Creazione vocabolario globale/per periodo
- Gestione frequenze minime

### Fase 4: Training Embeddings
```bash
python src/train_embeddings.py
```
- Addestramento modello CBOW per ogni periodo
- Salvataggio pesi in `models/`

### Fase 5: Allineamento Spazi
```bash
python src/alignment.py
```
- Orthogonal Procrustes alignment
- Salvataggio matrici in `alignment/`

### Fase 6: Analisi Semantic Drift
```bash
python src/semantic_drift.py
```
- Calcolo metriche di variazione semantica
- Export risultati in CSV

### Fase 7: Visualizzazione
```bash
python src/visualize_drift.py
```
- Generazione grafici t-SNE/PCA
- Salvataggio in `plots/`

---

## 📊 Stato Progetto

### ✅ Completato
- **Fase 1**: Setup ambiente e dipendenze
- **Fase 2**: Preprocessing testo (5M+ tokens processati)

### 🔄 In Corso
- **Fase 3**: Costruzione vocabolario

### 📋 Da Fare
- Fase 4: Training word embeddings
- Fase 5: Allineamento Procrustes
- Fase 6: Calcolo semantic drift
- Fase 7: Visualizzazioni

---

## 📓 Notebook Jupyter

Per una **guida interattiva** che spiega il progetto passo-passo:

```bash
jupyter notebook notebooks/01_intro_e_preprocessing.ipynb
```

Il notebook include:
- Spiegazione teorica del semantic drift
- Esplorazione dati preprocessati
- Visualizzazioni statistiche
- Esempi di parole che cambiano significato

---

## 📚 Dataset

**Nota**: I dati NON sono versionati su Git (`.gitignore`).

### Per training vero (HPC/locale):
Usa il tuo corpus storico privato in `data/raw/`

### Per demo/test (Colab):
```bash
python src/generate_demo_data.py  # Genera dati sintetici di esempio
```

### Dataset consigliati (pubblici):
1. **Google Books N-grams**: ampio, multi-lingua
2. **COHA (Corpus of Historical American English)**: bilanciato
3. **New York Times Archive**: alta qualità

---

## 🛠️ Troubleshooting

### Python 3.14 + spaCy
spaCy 3.x ha problemi con Python 3.14. Soluzione:
- Usa NLTK (già configurato di default in `config.py`)
- Oppure usa Python 3.11/3.12 con spaCy

### Memoria insufficiente
Riduci `BATCH_SIZE` in `config.py` o processa meno periodi.

---

## 📖 Riferimenti Teorici

- Hamilton et al. (2016). "Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change"
- Kulkarni et al. (2015). "Statistically Significant Detection of Linguistic Change"
- Orthogonal Procrustes: metodo matematico per allineare matrici

---

## 👨‍💻 Autore

Progetto didattico per studio NLP e analisi diacronica del linguaggio.
