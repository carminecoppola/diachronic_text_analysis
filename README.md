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



### Configurazione dei Path (per tutti)

Per rendere la configurazione dei path semplice e condivisibile tra tutti gli utenti, è fornito il file `.env.local.example` nella root del progetto.

**Come usare:**

1. Copia il file `.env.local.example` in `.env.local`:
	```bash
	cp .env.local.example .env.local
	```
2. Modifica i path all'interno di `.env.local` secondo la tua struttura locale.
	Puoi usare symlink per puntare a storage esterni condivisi (es. `/storage/external_01/...`).
3. Carica le variabili d'ambiente prima di lanciare gli script:
	```bash
	export $(grep -v '^#' .env.local | xargs)
	```
	oppure aggiungi le variabili al tuo `.bashrc`/`.zshrc`.

**Path configurabili tramite .env:**
- `DATA_DIR`: directory principale dei dati
- `MODELS_DIR`: directory dove salvare i modelli
- `PLOTS_DIR`: directory per i plot
- `ALIGNMENT_DIR`: directory per alignment

Se le variabili non sono specificate, verranno usati i path di default relativi alla struttura del progetto.

Per cambiare altri parametri (periodo temporale, tipo n-gram, dimensione embeddings, ecc.), modificare direttamente [src/config.py](src/config.py).

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
