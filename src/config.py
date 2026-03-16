"""
Central configuration for the Diachronic Text Analysis project.

Stores global parameters used across all pipeline stages:
- Time period and granularity
- Vocabulary parameters
- Embedding parameters
- Data paths

Modify this file to change global project behavior.
"""

import os
from pathlib import Path

# ============================================================================
# TIME PERIOD
# ============================================================================

# Analysis time window
START_YEAR = 1900
END_YEAR = 2019  # Inclusive (v3 2020 data available up to 2019)

# Decades to analyze (auto-generated)
# Generates: ["1900s", "1910s", "1920s", ..., "2010s"]
DECADES = [f"{year}s" for year in range(START_YEAR, END_YEAR + 1, 10)]

# ============================================================================
# DATASET (Google Books N-grams v3)
# ============================================================================

LANGUAGE = "eng"
DATASET_VERSION = "20200217"  # v3 2020

# N-gram type: 3 or 5 (affects context window size)
NGRAM_TYPE = 5

NGRAMS_BASE_URL = f"http://storage.googleapis.com/books/ngrams/books/{DATASET_VERSION}/{LANGUAGE}"

# File ranges to download (alphabetical sampling)
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

# N-gram filtering: keep only alphabetic n-grams
ONLY_ALPHABETIC = True

# ============================================================================
# VOCABULARY
# ============================================================================

VOCAB_SIZE = 50001  # Top 50K words + UNK token
MIN_VOCAB_FREQ = 100  # Minimum frequency threshold
UNK_TOKEN = "<UNK>"  # Out-of-vocabulary token

# ============================================================================
# WORD EMBEDDINGS
# ============================================================================

EMBEDDING_DIM = 300  # Vector dimension (standard Word2Vec)
CONTEXT_WINDOW = 2 if NGRAM_TYPE == 5 else 1
NEGATIVE_SAMPLES = 3

# Training parameters
BATCH_SIZE = 512
LEARNING_RATE = 0.025
EPOCHS = 3

# ============================================================================
# DATA PATHS
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Base data directory - configurable via environment variable for all data path resolution
BASE_DATA_DIR = os.environ.get("DATA_DIR", DATA_DIR)

"""
Main paths are configurable via environment variables:
  - DATA_DIR: main data directory (used for raw, processed, expanded data)
  - MODELS_DIR: model save directory
  - PLOTS_DIR: plots directory
  - ALIGNMENT_DIR: alignment directory

If not specified, defaults are used relative to the project root.
Usage: DATA_DIR=/path/to/data MODELS_DIR=/path/to/models python src/models/train.py
"""

# Separate directories by n-gram type (3gram, 5gram)
NGRAM_SUFFIX = f"{NGRAM_TYPE}gram-full"
DATA_RAW_DIR = os.path.join(BASE_DATA_DIR, "raw", NGRAM_SUFFIX)
DATA_PROCESSED_DIR = os.path.join(BASE_DATA_DIR, "processed", NGRAM_SUFFIX)
MODELS_DIR = Path(os.environ.get(
    "MODELS_DIR",
    os.path.join(PROJECT_ROOT, "models", NGRAM_SUFFIX)
))
PLOTS_DIR = os.environ.get(
    "PLOTS_DIR",
    os.path.join(PROJECT_ROOT, "plots")
)
ALIGNMENT_BASE = os.environ.get("ALIGNMENT_BASE", os.path.join(PROJECT_ROOT, "alignment"))
ALIGNMENT_NGRAM = os.environ.get("ALIGNMENT_NGRAM", NGRAM_SUFFIX)
ALIGNMENT_ROOT = os.path.join(ALIGNMENT_BASE, ALIGNMENT_NGRAM)
ALIGNMENT_TRANSFORMS_DIR = os.path.join(ALIGNMENT_ROOT, os.environ.get("ALIGNMENT_SUBDIR", "transforms"))
ALIGNMENT_DRIFT_RESULTS_DIR = os.path.join(ALIGNMENT_ROOT, "drift_results")
ALIGNMENT_GLOBAL_RESULTS_DIR = os.path.join(ALIGNMENT_ROOT, "global_results")

# Expanded dataset directories (balanced alphabetical distribution)
DATA_RAW_EXPANDED_DIR = os.path.join(BASE_DATA_DIR, "raw", f"{NGRAM_SUFFIX}_expanded")
DATA_PROCESSED_EXPANDED_DIR = os.path.join(BASE_DATA_DIR, "processed", f"{NGRAM_SUFFIX}_expanded")

# File paths
TOTAL_COUNTS_FILE = os.path.join(DATA_RAW_DIR, "total_counts.txt")
VOCAB_FILE = os.environ.get("VOCAB_PATH", os.path.join(DATA_PROCESSED_DIR,"vocab.json"))
FREQUENCIES_DIR = Path(os.environ.get("FREQUENCIES_DIR", os.path.join(ALIGNMENT_BASE, NGRAM_SUFFIX, "frequencies")))

# ============================================================================
# VISUALIZATION AND LOGGING
# ============================================================================

# Visualization parameters
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
PCA_N_COMPONENTS = 2
FIGURE_SIZE = (12, 8)
DPI = 300
PERIOD_COLORS = None

# Logging configuration = 'INFO'
SHOW_PROGRESS_BAR = True

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_directories():
    """Create required directories for the entire pipeline.
    
    Creates:
    - Raw and processed data directories (standard and expanded)
    - Model and plots directories
    - Alignment directories (transforms, drift results, global results)
    - Frequencies directory
    """
    dirs = [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        DATA_RAW_EXPANDED_DIR,
        DATA_PROCESSED_EXPANDED_DIR,
        MODELS_DIR,
        PLOTS_DIR,
        ALIGNMENT_ROOT,
        ALIGNMENT_TRANSFORMS_DIR,
        ALIGNMENT_DRIFT_RESULTS_DIR,
        ALIGNMENT_GLOBAL_RESULTS_DIR,
        FREQUENCIES_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def get_decade_from_year(year: int) -> str:
    """Convert year to decade string (1985 -> '1980s')."""
    return f"{(year // 10) * 10}s"

def get_processed_file_path(decade: str) -> str:
    """Get path to processed file for a given decade."""
    return os.path.join(DATA_PROCESSED_DIR, f"{decade}.txt")

def get_model_path(period: str) -> str:
    """Get path to model file for a given period."""
    return os.path.join(MODELS_DIR, f"emb_{period}.pt")

def get_preprocess_input_file(expanded: bool = True) -> str:
    """Get path to n-gram input file for preprocessing.
    
    Args:
        expanded: If True (default), returns path in expanded dataset directory.
                 If False, returns path in standard dataset directory.
    
    Returns:
        Path to n-gram file for preprocessing input.
    """
    base_dir = DATA_RAW_EXPANDED_DIR if expanded else DATA_RAW_DIR
    return os.path.join(base_dir, "ngrams.txt")

def get_preprocess_output_dir(expanded: bool = False) -> str:
    """Get output directory for preprocessing.
    
    Args:
        expanded: If True, returns expanded processed directory.
                 If False (default), returns standard processed directory.
    
    Returns:
        Path to preprocessing output directory.
    """
    return DATA_PROCESSED_EXPANDED_DIR if expanded else DATA_PROCESSED_DIR

def get_vocab_path(processed_dir: str = None) -> str:
    """Get path to vocabulary file.
    
    Args:
        processed_dir: Optional custom processed directory path.
                      If provided, returns vocab.json from this directory.
                      If None (default), returns default VOCAB_FILE.
    
    Returns:
        Path to vocabulary JSON file.
    """
    if processed_dir:
        return os.path.join(processed_dir, "vocab.json")
    return VOCAB_FILE
