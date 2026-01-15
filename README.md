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
