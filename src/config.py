"""
Configurazione Centrale del Progetto

Questo file contiene tutte le configurazioni e costanti utilizzate
nel progetto di analisi diacronica. Modificare questi parametri
per adattare il progetto a dataset e obiettivi specifici.
"""

import os
from pathlib import Path

# ============================================================================
# DIRECTORY PATHS
# ============================================================================

# Root directory del progetto
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Directories per dati
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Directories per output
MODELS_DIR = PROJECT_ROOT / "models"
ALIGNMENT_DIR = PROJECT_ROOT / "alignment"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Crea directories se non esistono
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, 
                 ALIGNMENT_DIR, PLOTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

# Preprocessing options
LOWERCASE = True                    # Converti tutto in minuscolo
REMOVE_PUNCTUATION = True          # Rimuovi punteggiatura
REMOVE_NUMBERS = False             # Rimuovi numeri (utili per date)
MIN_TOKEN_LENGTH = 2               # Lunghezza minima token
MAX_TOKEN_LENGTH = 50              # Lunghezza massima token
REMOVE_STOPWORDS = False           # Rimozione stopwords (opzionale)
USE_LEMMATIZATION = True           # Usa lemmatizzazione invece di stemming

# NLP tool (opzioni: 'nltk', 'spacy')
NLP_TOOL = 'nltk'  # Cambiato a NLTK per compatibilità Python 3.14
SPACY_MODEL = 'en_core_web_sm'


# ============================================================================
# TEMPORAL CONFIGURATION
# ============================================================================

# Definizione periodi temporali (esempio per decenni)
# Modificare in base al corpus specifico
PERIODS = [
    "1900s",  # 1900-1909
    "1910s",  # 1910-1919
    "1920s",  # 1920-1929
    "1930s",  # 1930-1939
    "1940s",  # 1940-1949
    "1950s",  # 1950-1959
    "1960s",  # 1960-1969
    "1970s",  # 1970-1979
    "1980s",  # 1980-1989
    "1990s",  # 1990-1999
]

# Periodo di riferimento per allineamento (primo periodo di default)
REFERENCE_PERIOD = PERIODS[0]


# ============================================================================
# VOCABULARY CONFIGURATION
# ============================================================================

# Frequenza minima per inclusione nel vocabolario
MIN_WORD_FREQ = 5

# Dimensione massima vocabolario (None = illimitato)
MAX_VOCAB_SIZE = 50000

# Token speciali
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Indici token speciali
PAD_IDX = 0
UNK_IDX = 1


# ============================================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================================

# Architettura modello (opzioni: 'cbow', 'skipgram')
MODEL_TYPE = 'cbow'

# Dimensione embedding
EMBEDDING_DIM = 300

# Finestra di contesto
CONTEXT_WINDOW = 5  # parole prima + dopo la target

# Training hyperparameters
BATCH_SIZE = 512
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
NEGATIVE_SAMPLES = 5  # per negative sampling

# Device
DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'

# Random seed per riproducibilità
RANDOM_SEED = 42


# ============================================================================
# ALIGNMENT CONFIGURATION
# ============================================================================

# Numero minimo di parole comuni per allineamento
MIN_COMMON_WORDS = 1000

# Usa solo parole ad alta frequenza per allineamento
USE_TOP_FREQ_FOR_ALIGNMENT = True
TOP_FREQ_N = 5000  # top N parole più frequenti


# ============================================================================
# SEMANTIC DRIFT CONFIGURATION
# ============================================================================

# Parole target da analizzare (None = automatico basato su frequenza)
TARGET_WORDS = None  # es. ['gay', 'cell', 'mouse', 'web', 'tweet']

# Se TARGET_WORDS è None, seleziona top N parole con maggiore drift
AUTO_SELECT_TOP_DRIFT = 50

# Numero di nearest neighbors da analizzare
K_NEAREST_NEIGHBORS = 10

# Threshold di similarità per filtrare risultati
MIN_SIMILARITY = 0.3


# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Metodo di riduzione dimensionalità (opzioni: 'tsne', 'pca')
REDUCTION_METHOD = 'tsne'

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

def get_period_file_path(period: str, data_type: str = 'processed') -> Path:
    """
    Restituisce il path del file per un dato periodo.
    
    Args:
        period: Nome del periodo (es. '1900s')
        data_type: Tipo di dati ('raw' o 'processed')
    
    Returns:
        Path completo del file
    """
    if data_type == 'raw':
        return RAW_DATA_DIR / f"{period}.txt"
    elif data_type == 'processed':
        return PROCESSED_DATA_DIR / f"{period}.txt"
    else:
        raise ValueError(f"data_type deve essere 'raw' o 'processed', ricevuto: {data_type}")


def get_model_path(period: str) -> Path:
    """
    Restituisce il path del file modello per un dato periodo.
    
    Args:
        period: Nome del periodo
    
    Returns:
        Path del file modello
    """
    return MODELS_DIR / f"emb_{period}.pt"


def get_vocab_path() -> Path:
    """
    Restituisce il path del file vocabolario.
    
    Returns:
        Path del file vocabolario
    """
    return PROCESSED_DATA_DIR / "vocab.pkl"
