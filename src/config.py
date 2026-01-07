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
END_YEAR = 1990  # Inclusivo (decade 1990-1999)

# Decenni da analizzare (generati automaticamente)
# Questo genera: ["1900s", "1910s", "1920s", ..., "1990s"]
# Ogni decade raggruppa 10 anni: 1900s = anni 1900-1909, 1910s = 1910-1919, ecc.
DECADES = [f"{year}s" for year in range(START_YEAR, END_YEAR + 1, 10)]

# Motivazione scelta temporale:
# - XX secolo offre stabilità linguistica e massimi cambiamenti semantici (tecnologia, società)
# - Pre-1900: problemi di OCR e ortografia non standardizzata
# - Post-1990: coverage limitata nel dataset (decade incompleta)
# - Granularità decennale: ottimale per catturare drift semantico senza rumore

# ============================================================================
# DATASET (Google Books N-grams v3)
# ============================================================================

# Lingua da analizzare
LANGUAGE = "eng"  # English

# Versione dataset (v3 2020 è la più recente)
DATASET_VERSION = "20200217"  # v3 (2020)

# URL base Google Cloud Storage (V3 2020 usa path diverso!)
NGRAMS_BASE_URL = "http://storage.googleapis.com/books/ngrams/books/20200217/eng"

# Numero di file da scaricare (su 24 totali)
# DATASET COMPLETO: tutti i 24 file (~26GB compressi)
NUM_FILES_TO_DOWNLOAD = 24

# Soglia minima occorrenze nel corpus (dataset già filtrato a 40+)
MIN_CORPUS_OCCURRENCES = 100

# ============================================================================
# PREPROCESSING
# ============================================================================

# Filtri token
MIN_TOKEN_LENGTH = 2
MAX_TOKEN_LENGTH = 20
ONLY_ALPHABETIC = True  # Solo lettere (no numeri, simboli)

# Normalizzazione
LOWERCASE = True

# ============================================================================
# VOCABOLARIO
# ============================================================================

# Dimensione vocabolario (top-K parole più frequenti)
# 50K è un buon trade-off tra copertura (parole comuni) e gestibilità computazionale
# Con ~15M parole uniche totali, 50K cattura le più stabili attraverso il secolo
VOCAB_SIZE = 50001  # Include UNK token (50000 + 1)

# Frequenza minima globale per inclusione nel vocabolario
# Filtra parole troppo rare che potrebbero essere errori OCR o nomi propri
MIN_VOCAB_FREQ = 100

# Token speciale per parole sconosciute (out-of-vocabulary)
# Durante il training, parole non nel vocabolario vengono mappate a <UNK>
UNK_TOKEN = "<UNK>"

# ============================================================================
# WORD EMBEDDINGS (per fasi successive)
# ============================================================================

# Dimensione vettori embedding
# 300 dimensioni è standard per word embeddings (Word2Vec, GloVe usano 100-300)
# Maggiore dimensionalità = maggiore capacità espressiva, ma più costoso
EMBEDDING_DIM = 300

# Finestra di contesto per CBOW/Skip-gram
# CONTEXT_WINDOW=5 significa: considera 5 parole prima + 5 dopo = 10 parole totali
# Finestre più grandi catturano relazioni semantiche più distanti
CONTEXT_WINDOW = 3

# Negative sampling
# Per ogni parola positiva, campiona 5 parole negative (per loss contrastivo)
# Trade-off: più samples = training più lento ma migliore qualità
NEGATIVE_SAMPLES = 3

# Training parameters
BATCH_SIZE = 512        # Numero di esempi per batch (più alto = più veloce, più RAM)
LEARNING_RATE = 0.025   # Learning rate iniziale (standard per Word2Vec)
EPOCHS = 1             # Numero di passaggi completi sul dataset

# ============================================================================
# PATH DATI
# ============================================================================

# Root del progetto (directory contenente src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory dati (NON in Git)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Directory modelli (NON in Git)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Directory plot (NON in Git)
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

# Directory alignment (NON in Git)
ALIGNMENT_DIR = os.path.join(PROJECT_ROOT, "alignment")

# File specifici
TOTAL_COUNTS_FILE = os.path.join(DATA_RAW_DIR, "total_counts.txt")
VOCAB_FILE = os.path.join(DATA_PROCESSED_DIR, "vocab.json")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_directories():
    """Crea tutte le directory necessarie se non esistono."""
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(ALIGNMENT_DIR, exist_ok=True)
    print(f"✓ Directory create: {DATA_DIR}, {MODELS_DIR}, {PLOTS_DIR}")


def get_decade_from_year(year: int) -> str:
    """
    Converte un anno nel suo decennio.
    
    Args:
        year: Anno (es. 1985)
    
    Returns:
        Stringa decennio (es. "1980s")
    """
    decade_start = (year // 10) * 10
    return f"{decade_start}s"


def get_processed_file_path(decade: str) -> str:
    """
    Restituisce il path del file processato per un decennio.
    
    Args:
        decade: Stringa decennio (es. "1990s")
    
    Returns:
        Path completo (es. "data/processed/1990s.txt")
    """
    return os.path.join(DATA_PROCESSED_DIR, f"{decade}.txt")


if __name__ == "__main__":
    # Test configurazione
    create_directories()
    print(f"\n📅 Decenni da analizzare: {DECADES}")
    print(f"📊 Vocabolario: {VOCAB_SIZE} parole")
    print(f"🧠 Embedding dimension: {EMBEDDING_DIM}")
    print(f"📁 Path dati: {DATA_DIR}")
# Parametri t-SNE
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000

# Parametri PCA
PCA_N_COMPONENTS = 2

# Dimensioni figure
FIGURE_SIZE = (12, 8)
DPI = 300

# Colori per periodi (generati automaticamente se None)
PERIOD_COLORS = None


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Livello di verbosità (opzioni: 'DEBUG', 'INFO', 'WARNING', 'ERROR')
LOG_LEVEL = 'INFO'

# Stampa progress bar durante training
SHOW_PROGRESS_BAR = True


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_period_file_path(period: str, data_type: str = 'processed') -> str:
    """
    Restituisce il path del file per un dato periodo.
    
    Args:
        period: Nome del periodo (es. '1900s')
        data_type: Tipo di dati ('raw' o 'processed')
    
    Returns:
        Path completo del file
    """
    if data_type == 'raw':
        return os.path.join(DATA_RAW_DIR, f"{period}.txt")
    elif data_type == 'processed':
        return os.path.join(DATA_PROCESSED_DIR, f"{period}.txt")
    else:
        raise ValueError(f"data_type deve essere 'raw' o 'processed', ricevuto: {data_type}")



def get_model_path(period: str) -> str:
    """
    Restituisce il path del file modello per un dato periodo.
    
    Args:
        period: Nome del periodo (es. "1990s")
    
    Returns:
        Path del file modello
    """
    return os.path.join(MODELS_DIR, f"emb_{period}.pt")


def get_vocab_path() -> str:
    """
    Restituisce il path del file vocabolario.
    
    Returns:
        Path del file vocabolario
    """
    return VOCAB_FILE
