"""
Configuration centrale del progetto Diachronic Text Analysis.

Contiene tutti i parametri globali utilizzati dalle varie fasi:
- Periodo temporale e granularità
- Parametri vocabolario
- Parametri embeddings
- Path dei dati

Modificare qui per cambiare comportamento globale del progetto.
"""

import os

# ============================================================================
# PERIODO TEMPORALE
# ============================================================================

# Finestra temporale di analisi
START_YEAR = 1900
END_YEAR = 2019  # Inclusivo (per v3 2020 dati disponibili fino al 2019)

# Decenni da analizzare (generati automaticamente)
# Questo genera: ["1900s", "1910s", "1920s", ..., "1990s"]
# Ogni decade raggruppa 10 anni: 1900s = anni 1900-1909, 1910s = 1910-1919, ecc.
DECADES = [f"{year}s" for year in range(START_YEAR, END_YEAR + 1, 10)]

# Motivazione scelta temporale:
# - XX-XXI secolo: massimi cambiamenti semantici (tecnologia, società, digitale)
# - Pre-1900: problemi di OCR e ortografia non standardizzata
# - 1900-2019: qualità OCR ottima, coverage Google Books alta
# - Include rivoluzione digitale (computer, internet, smartphone)
# - Granularità decennale: ottimale per catturare drift semantico senza rumore

# ============================================================================
# DATASET (Google Books N-grams v3)
# ============================================================================

LANGUAGE = "eng"
DATASET_VERSION = "20200217"  # v3 2020

# Tipo N-gram: 3 (window=1) o 5 (window=2)
NGRAM_TYPE = 5

NGRAMS_BASE_URL = f"http://storage.googleapis.com/books/ngrams/books/{DATASET_VERSION}/{LANGUAGE}"

# Range file da scaricare (campionamento alfabetico)
if NGRAM_TYPE == 3:
    FILE_RANGES = [(2000, 2020)]
    NUM_FILES_TO_DOWNLOAD = 20
elif NGRAM_TYPE == 5:
    FILE_RANGES = [(2000, 2005), (3000, 3005), (4000, 4005), 
                   (5000, 5005), (6000, 6005), (7000, 7005)]
    NUM_FILES_TO_DOWNLOAD = 30
else:
    FILE_RANGES = [(2000, 2020)]
    NUM_FILES_TO_DOWNLOAD = 20

MIN_CORPUS_OCCURRENCES = 100

# Filtraggio n-gram
ONLY_ALPHABETIC = True  # Mantieni solo n-gram con parole alfabetiche

# ============================================================================
# VOCABOLARIO
# ============================================================================

VOCAB_SIZE = 50001  # Top 50K parole + UNK
MIN_VOCAB_FREQ = 100  # Soglia minima frequenza globale
UNK_TOKEN = "<UNK>"  # Token per parole out-of-vocabulary

# ============================================================================
# WORD EMBEDDINGS
# ============================================================================

EMBEDDING_DIM = 300  # Dimensione vettori (standard Word2Vec)
CONTEXT_WINDOW = 2 if NGRAM_TYPE == 5 else 1  # Finestra contesto
NEGATIVE_SAMPLES = 3  # Negative sampling

# Training
BATCH_SIZE = 512
LEARNING_RATE = 0.025
EPOCHS = 3

# ============================================================================
# PATH DATI
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Directory separate per tipo n-gram (3gram, 5gram)
NGRAM_SUFFIX = f"{NGRAM_TYPE}gram"
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw", NGRAM_SUFFIX)
# DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed", NGRAM_SUFFIX)
DATA_PROCESSED_DIR = "/home/vbucciero/diachronic_text_analysis/data/processed/5gram-full"
MODELS_DIR = os.path.join("/storage/external_01/diachronic_text_analysis/models", NGRAM_SUFFIX)
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
ALIGNMENT_DIR = os.path.join(PROJECT_ROOT, "alignment")

# Directory per dataset espanso (distribuzione alfabetica bilanciata)
DATA_RAW_EXPANDED_DIR = os.path.join(DATA_DIR, "raw", f"{NGRAM_SUFFIX}_expanded")
DATA_PROCESSED_EXPANDED_DIR = os.path.join(DATA_DIR, "processed", f"{NGRAM_SUFFIX}_expanded")

# File specifici
TOTAL_COUNTS_FILE = os.path.join(DATA_RAW_DIR, "total_counts.txt")
VOCAB_FILE = os.path.join(DATA_PROCESSED_DIR, "vocab.json")

# ============================================================================
# VISUALIZZAZIONE E LOGGING
# ============================================================================

# Parametri visualizzazione
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
PCA_N_COMPONENTS = 2
FIGURE_SIZE = (12, 8)
DPI = 300
PERIOD_COLORS = None

# Logging
LOG_LEVEL = 'INFO'
SHOW_PROGRESS_BAR = True

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_directories():
    """Crea directory necessarie."""
    for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, PLOTS_DIR, ALIGNMENT_DIR, 
              DATA_RAW_EXPANDED_DIR, DATA_PROCESSED_EXPANDED_DIR]:
        os.makedirs(d, exist_ok=True)

def get_decade_from_year(year: int) -> str:
    """Converte anno in decennio (1985 → "1980s")."""
    return f"{(year // 10) * 10}s"

def get_processed_file_path(decade: str) -> str:
    """Path file processato per decennio."""
    return os.path.join(DATA_PROCESSED_DIR, f"{decade}.txt")

def get_model_path(period: str) -> str:
    """Path file modello per periodo."""
    return os.path.join(MODELS_DIR, f"emb_{period}.pt")

def get_vocab_path() -> str:
    """Path file vocabolario."""
    return VOCAB_FILE
