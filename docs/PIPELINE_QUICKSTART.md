# Pipeline Quickstart - Fasi di Preprocessing

Guida rapida ai comandi per lanciare le prime 3 fasi della pipeline diachronic text analysis.

## Prerequisiti

```bash
# Attivare virtual environment
source .venv/bin/activate

# Verificare che le directory richieste esistono
python3 -c "from src.config import create_directories; create_directories()"
```

---

## Fase 1: Download N-grams da Google Books

### Modalità Standard (Download automatico)

```bash
cd /home/ccoppola/projects/diachronic_text_analysis

PYTHONPATH=. python3 src/data/download_ngrams.py
```

**Output:**
- File scaricati in: `data/raw/5gram-full/`
- Formato: `ngram` \t `year` \t `match_count` \t `volume_count`
- Circa 30 file (configurabile via `FILE_RANGES` in `src/config.py`)

**Configurazione:**
Modificare `src/config.py` per:
- Cambiare `FILE_RANGES` (range di file da scaricare)
- Cambiare `NUM_FILES_TO_DOWNLOAD`
- Cambiare `NGRAM_TYPE` (3 o 5)

---

## Fase 2: NLP Preprocessing

```bash
cd /home/ccoppola/projects/diachronic_text_analysis

PYTHONPATH=. python3 src/data/preprocess.py
```

**Output:**
- File processati per decennio: `data/processed/5gram-full/1900s.txt`, `1910s.txt`, etc.
- Un n-gram per riga, replicato per frequenza

**Comportamento:**
Lo script legge automaticamente da `data/raw/5gram-full/ngrams.txt` (output della Fase 1) e salva i file processati in `data/processed/5gram-full/` in base alla struttura di directory scaricata.

---

## Fase 3: Build Shared Vocabulary

```bash
cd /home/ccoppola/projects/diachronic_text_analysis

PYTHONPATH=. python3 src/data/build_vocab.py \
  --processed-dir data/processed/5gram-full
```

**Parametri:**
- `--processed-dir`: (obbligatorio) path alla directory con file decennali processati

**Output:**
- `vocab.json`: vocabolario con mapping word→index
- `vocab_stats.txt`: statistiche dettagliate

---

## Comando Completo in Sequenza

Eseguire tutte e 3 le fasi in sequenza:

```bash
#!/bin/bash

cd /home/ccoppola/projects/diachronic_text_analysis
source .venv/bin/activate

echo "Phase 1: Download n-grams..."
PYTHONPATH=. python3 src/data/download_ngrams.py

echo ""
echo "Phase 2: Preprocessing..."
PYTHONPATH=. python3 src/data/preprocess.py

echo ""
echo "Phase 3: Build vocabulary..."
PYTHONPATH=. python3 src/data/build_vocab.py --processed-dir data/processed/5gram-full

echo ""
echo "✓ Pipeline completed!"
```

Salvare come `run_pipeline.sh` ed eseguire:
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

---

## Monitoraggio Progresso

### Verificare file generati

```bash
# Fase 1: File scaricati
ls -lh data/raw/5gram-full/

# Fase 2: File processati
ls -lh data/processed/5gram-full/
wc -l data/processed/5gram-full/*.txt

# Fase 3: Vocabolario
python3 -c "
import json
with open('data/processed/5gram-full/vocab.json') as f:
    vocab = json.load(f)
print(f'Vocabulary size: {len(vocab):,}')
print(f'Sample words: {list(vocab.items())[:10]}')
"
```

### Visualizzare statistiche vocabolario

```bash
head -50 data/processed/5gram-full/vocab_stats.txt
```

---

## Variabili di Ambiente (Advanced)

Personalizzare comportamento via environment variables:

```bash
# Cambiare directory dati principale
DATA_DIR=/path/to/custom/data python3 src/data/preprocess.py

# Cambiare directory modelli
MODELS_DIR=/path/to/models python3 src/models/train.py

# Cambiare directory grafici
PLOTS_DIR=/path/to/plots python3 src/visualization/visualize_drift.py

# Tutte insieme
DATA_DIR=/data MODELS_DIR=/models python3 src/data/preprocess.py
```

---

## Troubleshooting

### Errore: ModuleNotFoundError: No module named 'src'

**Soluzione:** Aggiungere `PYTHONPATH`
```bash
PYTHONPATH=/home/ccoppola/projects/diachronic_text_analysis python3 src/data/preprocess.py
```

### Errore: FileNotFoundError: input file not found

**Soluzione:** Verificare che Phase 1 sia completata
```bash
ls -la data/raw/5gram-full/
```

### Errore: No directories with decade files found

**Soluzione:** Verificare che Phase 2 sia completata
```bash
ls -la data/processed/5gram-full/
```

### Processamento lento

**Ottimizzazioni:**
- Ridurre `FILE_RANGES` in `src/config.py` (meno file da scaricare)
- Ridurre `NUM_FILES_TO_DOWNLOAD`
- Usare dataset non espanso: `--expanded False`

---

## Prossimi Passi

Dopo completamento delle 3 fasi:

```bash
# Fase 4: Training embeddings
PYTHONPATH=. python3 src/models/train.py

# Fase 5: Compute alignment
PYTHONPATH=. python3 src/alignment/procrustes_drift.py

# Fase 6: Visualize results
PYTHONPATH=. python3 src/visualization/visualize_drift.py
```

Vedi [TRAINING.md](TRAINING.md) per dettagli sul training.

---

## Riferimenti Configurazione

Per modifiche avanzate, consultare:
- `src/config.py` - Parametri globali (VOCAB_SIZE, NGRAM_TYPE, anni, etc.)
- `src/data/preprocess.py` - Parametri NLP (MIN_TOKEN_LENGTH, MIN_ALPHA_RATIO, etc.)
- `src/data/build_vocab.py` - Parametri vocabolario

Per domande sulla pipeline vedere [SPIEGAZIONE_PROGETTO.md](SPIEGAZIONE_PROGETTO.md)
