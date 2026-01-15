### Esempio: output alignment separati

Per generare i file di alignment per un dataset specifico (es. 5gram-full) e salvarli nella struttura corretta:

```bash
# Output in alignment/5gram-full/drift_results, .../global_results, .../transforms
ALIGNMENT_NGRAM=5gram-full python src/alignment/procrustes_drift.py
```

Per usare una sottocartella personalizzata per i tuoi alignment (es. test_camilla):

```bash
ALIGNMENT_NGRAM=5gram-full ALIGNMENT_SUBDIR=test_camilla python src/alignment/procrustes_drift.py
# Output in alignment/5gram-full/test_camilla
```

Le directory vengono create automaticamente se non esistono.
# Diachronic Text Analysis

Analisi dell'evoluzione semantica delle parole nel XX-XXI secolo tramite word embeddings CBOW allenati su Google Books N-grams.

## Descrizione

Il progetto studia come cambia il significato delle parole tra il 1900 e il 2019 utilizzando:
- Word embeddings (CBOW) allenati per ogni decennio
- Allineamento con Orthogonal Procrustes
- Analisi semantic drift via cosine similarity
- Visualizzazioni PCA/t-SNE

Esempio: "computer" negli anni '30 indica un calcolatore umano, negli anni '90 un dispositivo elettronico.

## Installazione

```bash
git clone https://github.com/carminecoppola/diachronic_text_analysis.git
cd diachronic_text_analysis

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilizzo

### Pipeline

```bash
# Preprocessing e vocabolario
python src/data/preprocess.py
python src/data/build_vocab.py

# Training embeddings
sbatch run_training.sbatch

# Alignment e drift
python src/alignment/compute_frequencies.py
python src/alignment/procrustes_drift.py
python src/alignment/global_drift.py

# Visualizzazioni
python src/visualization/visualize_drift.py --words computer --pca --tsne
```

### Configurazione

Tutti i parametri del progetto sono centralizzati in [src/config.py](src/config.py):
- Periodo temporale e decenni
- Tipo di N-gram (3gram o 5gram)
- Parametri embeddings (dimensione, learning rate, epochs)
- Path di dati, modelli, plot e alignment

**Path configurabili tramite variabili d'ambiente:**
- `DATA_DIR`: directory principale dei dati
- `MODELS_DIR`: directory dove salvare i modelli  
- `PLOTS_DIR`: directory per i plot
- `ALIGNMENT_BASE`: directory base per alignment
- `ALIGNMENT_NGRAM`: sottocartella per tipo n-gram (es. `5gram-full`)
- `ALIGNMENT_SUBDIR`: sottocartella personalizzata per transforms (opzionale)
- `VOCAB_PATH`: path del file vocabulary
- `FREQUENCIES_DIR`: directory delle frequenze

Se le variabili d'ambiente non sono specificate, vengono usati i path di default relativi alla struttura del progetto.

## Contributi

Branch `main` protetto: creare sempre branch personale e aprire Pull Request.

```bash
git checkout main && git pull origin main
git checkout -b nome-cognome
# ... sviluppo ...
git push origin nome-cognome
# Aprire PR su GitHub
```

## Documentazione

Dettagli nelle fasi del progetto:
- [DATASET.md](docs/DATASET.md) - Google Books N-grams, preprocessing, vocabolario
- [TRAINING.md](docs/TRAINING.md) - Architettura CBOW, dataset PyTorch, training
- [ALIGNMENT.md](docs/ALIGNMENT.md) - Orthogonal Procrustes, anchor selection
- [DRIFT.md](docs/DRIFT.md) - Local/global drift, trajectories, nearest neighbors
- [VISUALIZATION.md](docs/VISUALIZATION.md) - PCA, t-SNE, plotting

## Requisiti

Python 3.8+, PyTorch 2.0+, NumPy, SciPy, scikit-learn

Storage: ~20GB dataset, ~10GB modelli
