"""
DIACHRONIC TEXT ANALYSIS - Central configuration and path definitions.
Single source of truth for all parameters affecting data paths, models, and pipeline stages.
Override paths via environment variables: DATA_DIR, MODELS_DIR, PLOTS_DIR, ALIGNMENT_BASE.
"""

import os
from pathlib import Path

# TIME PERIOD CONFIGURATION
# 120-year period (1900-2019) = 12 decades
START_YEAR = 1900
END_YEAR = 2019
DECADES = [f"{year}s" for year in range(START_YEAR, END_YEAR + 1, 10)]  # Auto-generated list

# DATASET CONFIGURATION - Google Books N-grams v3 (1900-2019, English)
LANGUAGE = "eng"
DATASET_VERSION = "20200217"

# N-GRAM TYPE: 3 or 5 (affects context window, directory names, dataset size)
# NGRAM_TYPE=5: window=2 (larger context), better semantics, larger models
# NGRAM_TYPE=3: window=1 (smaller context), faster training, smaller models
NGRAM_TYPE = 5

NGRAMS_BASE_URL = f"http://storage.googleapis.com/books/ngrams/books/{DATASET_VERSION}/{LANGUAGE}"

# Download file ranges (alphabetical sampling)
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

MIN_CORPUS_OCCURRENCES = 100  # Minimum frequency threshold
ONLY_ALPHABETIC = True  # Keep only alphabetic n-grams (no numbers/punctuation)

# VOCABULARY CONFIGURATION
# Fixed vocabulary (top 50K words + UNK token) built once from entire corpus
# Ensures consistent word→index mapping across all decades and training runs
VOCAB_SIZE = 50001             # 50,000 words + 1 UNK token
MIN_VOCAB_FREQ = 100           # Minimum frequency for inclusion
UNK_TOKEN = "<UNK>"            # Out-of-vocabulary token

# WORD EMBEDDINGS - CBOW Configuration
# CBOW: Predicts word from context using neural network with word embedding hidden layer
EMBEDDING_DIM = 300            # Word vector dimension (Word2Vec standard)
CONTEXT_WINDOW = 2 if NGRAM_TYPE == 5 else 1  # Context words per side (derived from NGRAM_TYPE)
NEGATIVE_SAMPLES = 3           # Negative samples per positive example

# Training parameters
BATCH_SIZE = 512               # Samples per batch
LEARNING_RATE = 0.025          # Learning rate for SGD
EPOCHS = 3                      # Number of passes through training data

# DATA PATHS - Derived from PROJECT_ROOT, customizable via environment variables
# Priority: Environment variables > Default paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))

# N-gram suffix (e.g., "5gram" when NGRAM_TYPE=5) used in all directory names
NGRAM_SUFFIX = f"{NGRAM_TYPE}gram"

# Raw data directories (unprocessed files from Google Books)
DATA_RAW_DIR = os.path.join(BASE_DATA_DIR, "raw", NGRAM_SUFFIX)
DATA_RAW_EXPANDED_DIR = os.path.join(BASE_DATA_DIR, "raw", f"{NGRAM_SUFFIX}-expanded")

# Processed data directories (aggregated frequencies per decade)
DATA_PROCESSED_DIR = os.path.join(BASE_DATA_DIR, "processed", NGRAM_SUFFIX)
DATA_PROCESSED_EXPANDED_DIR = os.path.join(BASE_DATA_DIR, "processed", f"{NGRAM_SUFFIX}-expanded")

# Models directory (trained CBOW checkpoints)
MODELS_DIR = Path(os.environ.get("MODELS_DIR", os.path.join(PROJECT_ROOT, "models", f"{NGRAM_SUFFIX}-full")))

# Visualization output
PLOTS_DIR = os.environ.get("PLOTS_DIR", os.path.join(PROJECT_ROOT, "plots"))

# Alignment results (Procrustes transforms, drift analysis, frequencies)
ALIGNMENT_BASE = os.environ.get("ALIGNMENT_BASE", os.path.join(PROJECT_ROOT, "alignment"))
ALIGNMENT_ROOT = os.path.join(ALIGNMENT_BASE, os.environ.get("ALIGNMENT_NGRAM", NGRAM_SUFFIX))
ALIGNMENT_TRANSFORMS_DIR = os.path.join(ALIGNMENT_ROOT, os.environ.get("ALIGNMENT_SUBDIR", "transforms"))
ALIGNMENT_DRIFT_RESULTS_DIR = os.path.join(ALIGNMENT_ROOT, "drift_results")
ALIGNMENT_GLOBAL_RESULTS_DIR = os.path.join(ALIGNMENT_ROOT, "global_results")

# Vocabulary and metadata files
TOTAL_COUNTS_FILE = os.path.join(DATA_RAW_DIR, "total_counts.txt")
VOCAB_FILE = os.environ.get("VOCAB_PATH", os.path.join(DATA_PROCESSED_DIR, "vocab.json"))
VOCAB_FILE_EXPANDED = os.environ.get("VOCAB_PATH", os.path.join(DATA_PROCESSED_EXPANDED_DIR, "vocab.json"))
FREQUENCIES_DIR = Path(os.environ.get("FREQUENCIES_DIR", os.path.join(ALIGNMENT_BASE, NGRAM_SUFFIX, "frequencies")))
FREQUENCIES_DIR_EXPANDED = Path(os.environ.get("FREQUENCIES_DIR_EXPANDED",os.path.join(ALIGNMENT_BASE, f"{NGRAM_SUFFIX}-full", "frequencies"),))

# VISUALIZATION AND ANALYSIS CONFIGURATION
TSNE_PERPLEXITY = 30           # t-SNE perplexity (standard: 30)
TSNE_N_ITER = 1000             # t-SNE iterations
PCA_N_COMPONENTS = 2           # PCA dimensions
FIGURE_SIZE = (12, 8)          # Figure size (width, height) in inches
DPI = 300                       # Resolution (dots per inch)
PERIOD_COLORS = None           # Color map for decades (None = matplotlib default)
SHOW_PROGRESS_BAR = True       # Show progress bars in terminal

# UTILITY FUNCTIONS - Path construction and directory management
def create_directories():
    """Create all required directories for the pipeline."""
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
        FREQUENCIES_DIR_EXPANDED,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def get_decade_from_year(year: int) -> str:
    """Convert year to decade string (e.g., 1985 → "1980s")."""
    return f"{(year // 10) * 10}s"

def get_processed_file_path(decade: str) -> str:
    """Get path to preprocessed frequency file for a decade."""
    return os.path.join(DATA_PROCESSED_DIR, f"{decade}.txt")

def get_model_path(period: str) -> str:
    """Get path to trained CBOW model checkpoint for a period."""
    return os.path.join(MODELS_DIR, f"emb_{period}.pt")

def get_preprocess_input_file(expanded: bool = True) -> str:
    """Get path to raw n-gram input file for preprocessing."""
    base_dir = DATA_RAW_EXPANDED_DIR if expanded else DATA_RAW_DIR
    filename = f"{NGRAM_TYPE}gram_filtered_expanded.tsv" if expanded else f"{NGRAM_TYPE}gram_filtered.tsv"
    return os.path.join(base_dir, filename)

def get_preprocess_output_dir(expanded: bool = False) -> str:
    """Get output directory for preprocessed decade files."""
    return DATA_PROCESSED_EXPANDED_DIR if expanded else DATA_PROCESSED_DIR

def get_vocab_path(processed_dir: str = None) -> str:
    """Get path to vocabulary JSON file (word → index mapping)."""
    if processed_dir:
        return os.path.join(processed_dir, "vocab.json")
    return VOCAB_FILE
