"""
Anchor word selection and Orthogonal Procrustes alignment for diachronic embeddings.

1. Selects anchor words using frequency-based filtering
2. Loads CBOW embeddings from independently trained models
3. Aligns embedding spaces via Orthogonal Procrustes

"""

import json
import numpy as np
from pathlib import Path


# Configuration
FREQUENCIES_DIR = Path(__file__).parent / "frequencies"

# Nel progetto reale:
# non scriverai a mano {"computer", ...}
# userai:
#   stopword list
#   oppure top-N parole più frequenti globali
#   oppure filtri POS (se disponibili)
EXCLUDE_WORDS = {"computer", "machine", "network", "software", "internet"}

#Questo in teoria non serve se il data set ha già le frequenze 
def load_frequencies(decade: str, freq_dir: Path = FREQUENCIES_DIR) -> dict:
    """
    Load word frequency dictionary from JSON file.
    
    Args:
        decade: Decade name (e.g., "1980s")
        freq_dir: Directory containing frequency JSON files
    
    Returns:
        Dictionary mapping word -> count
    """
    freq_file = freq_dir / f"freq_{decade}.json"
    
    if not freq_file.exists():
        raise FileNotFoundError(f"Frequency file not found: {freq_file}")
    
    with open(freq_file, 'r', encoding='utf-8') as f:
        freq = json.load(f)
    
    return freq

# Questo è da modificare non so come sono stati costruiti gli embeddings
def load_embeddings_from_cbow(embeddings_file: str = "embeddings_cbow.pkl"):
    """
    Load pre-trained CBOW embeddings from pickle file.
    
    Note: This requires running run_toy_cbow.py first.
    
    Args:
        embeddings_file: Path to pickle file containing embeddings_by_decade
    
    Returns:
        Dictionary: embeddings_by_decade
    """
    import pickle
    
    file_path = Path(__file__).parent / embeddings_file
    
    if not file_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {file_path}\n"
                               "Run run_toy_cbow.py first to generate embeddings.")
    
    with open(file_path, 'rb') as f:
        embeddings_by_decade = pickle.load(f)
    
    return embeddings_by_decade



def select_anchor_words(freq_t: dict, freq_t1: dict, min_count: int = 2, exclude=None) -> list:
    """
    Select anchor words shared by two consecutive decades.
    
    Anchor words are those that:
    1. Appear in both decades
    2. Have frequency >= min_count in both decades
    
    Args:
        freq_t: Frequency dictionary for decade t
        freq_t1: Frequency dictionary for decade t+1
        min_count: Minimum token frequency required in both decades (default: 2)
    
    Returns:
        Sorted list of anchor words
    """
    # Find words present in both decades
    words_t = set(freq_t.keys())
    words_t1 = set(freq_t1.keys())
    shared_words = words_t & words_t1
    
    # Filter by minimum count in both decades
    anchors = [
        word for word in shared_words
        if freq_t[word] >= min_count and freq_t1[word] >= min_count and word not in (exclude or set())
    ]
    
    # Return sorted for reproducibility
    return sorted(anchors)


def build_alignment_matrices(anchor_words: list, E_t: np.ndarray, vocab_t: dict,
                            E_t1: np.ndarray, vocab_t1: dict) -> tuple:
    """
    Build alignment matrices from anchor word embeddings.
    
    For each anchor word, retrieves its embedding from both decades.
    Returns two matrices: one per decade, both with shape (|anchors|, d).
    
    Args:
        anchor_words: List of anchor words
        E_t: Embedding matrix for decade t, shape (|V_t|, d)
        vocab_t: Vocabulary dict for decade t: {word -> index}
        E_t1: Embedding matrix for decade t+1, shape (|V_t1|, d)
        vocab_t1: Vocabulary dict for decade t+1: {word -> index}
    
    Returns:
        Tuple (X, Y) where:
        - X ∈ R^{n×d}: embeddings of anchors from decade t
        - Y ∈ R^{n×d}: embeddings of anchors from decade t+1
    """
    X_rows = []
    Y_rows = []
    
    for word in anchor_words:
        # Get indices in both decades
        idx_t = vocab_t[word]
        idx_t1 = vocab_t1[word]
        
        # Get embeddings
        X_rows.append(E_t[idx_t])
        Y_rows.append(E_t1[idx_t1])
    
    X = np.vstack(X_rows)  # shape: (|anchors|, d)
    Y = np.vstack(Y_rows)  # shape: (|anchors|, d)
    
    return X, Y


def orthogonal_procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute orthogonal Procrustes rotation matrix.
    
    Finds the orthogonal matrix R that minimizes ||Y - X @ R||_F^2
    using SVD decomposition of X^T @ Y.
    
    Args:
        X: Matrix of shape (n, d) - source embeddings
        Y: Matrix of shape (n, d) - target embeddings
    
    Returns:
        R: Orthogonal rotation matrix of shape (d, d)
    """
    # Compute the covariance matrix
    M = X.T @ Y

    # SVD decomposition: M = U @ S @ V^T
    U, _, V_T = np.linalg.svd(M)

    # Orthogonal rotation matrix: R = U @ V^T
    R = U @ V_T

    return R


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        u, v: 1D numpy arrays
    
    Returns:
        Cosine similarity (scalar)
    """
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    if norm_u == 0 or norm_v == 0:
        return 0.0
    
    return np.dot(u, v) / (norm_u * norm_v)



def main():
    """Main execution: anchor selection + Procrustes alignment."""
    
    print(f"Loading frequency files from: {FREQUENCIES_DIR}\n")
    
    # Step 1: Load frequencies and select anchors
    freq_1980s = load_frequencies("1980s")
    freq_1990s = load_frequencies("1990s")
    
    print(f"1980s vocabulary size: {len(freq_1980s)}")
    print(f"1990s vocabulary size: {len(freq_1990s)}")
    print()
    
    # Select anchor words with min_count=2
    min_count = 2
    anchors = select_anchor_words(freq_1980s, freq_1990s, min_count=min_count, exclude=EXCLUDE_WORDS)
    
    # Print anchor selection results
    print(f"{'='*60}")
    print(f"Anchor Word Selection (1980s -> 1990s)")
    print(f"{'='*60}")
    print(f"Minimum frequency threshold: {min_count}")
    print(f"Number of anchor words: {len(anchors)}")
    print(f"\nAnchor words: {anchors}")
    print()
    
    # Print detailed statistics for each anchor
    print(f"{'='*60}")
    print(f"Anchor Word Frequencies")
    print(f"{'='*60}")
    print(f"{'Word':<15} {'1980s':<10} {'1990s':<10}")
    print(f"{'-'*35}")
    for word in anchors:
        count_1980s = freq_1980s[word]
        count_1990s = freq_1990s[word]
        print(f"{word:<15} {count_1980s:<10} {count_1990s:<10}")
    print()
    
    # Step 2: Load CBOW embeddings
    print(f"{'='*60}")
    print("Loading CBOW embeddings...")
    print(f"{'='*60}")
    
    try:
        embeddings_by_decade = load_embeddings_from_cbow()
        print("✓ Embeddings loaded successfully\n")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nTo generate embeddings, run:")
        print("  python run_toy_cbow.py\n")
        print("This will save embeddings_cbow.pkl")
        return
    
    # Extract embeddings for 1980s and 1990s
    E_1980s = embeddings_by_decade["1980s"]["E"]
    vocab_1980s = embeddings_by_decade["1980s"]["vocab"]
    E_1990s = embeddings_by_decade["1990s"]["E"]
    vocab_1990s = embeddings_by_decade["1990s"]["vocab"]
    
    print(f"1980s embeddings shape: {E_1980s.shape}")
    print(f"1990s embeddings shape: {E_1990s.shape}")
    print(f"Embedding dimension: {E_1980s.shape[1]}\n")
    
    # Step 3: Build alignment matrices
    print(f"{'='*60}")
    print("Building alignment matrices...")
    print(f"{'='*60}")
    
    X, Y = build_alignment_matrices(anchors, E_1980s, vocab_1980s, E_1990s, vocab_1990s)
    
    print(f"Source matrix X (1980s anchors): {X.shape}")
    print(f"Target matrix Y (1990s anchors): {Y.shape}\n")
    
    # Step 4: Compute Orthogonal Procrustes
    print(f"{'='*60}")
    print("Computing Orthogonal Procrustes alignment...")
    print(f"{'='*60}")
    
    R = orthogonal_procrustes(X, Y)
    
    print(f"Rotation matrix R shape: {R.shape}")
    
    # Check orthogonality: R @ R^T should be identity (within numerical tolerance)
    is_orthogonal = np.allclose(R @ R.T, np.eye(R.shape[0]), atol=1e-6)
    print(f"Is orthogonal (within 1e-6 tolerance)? {is_orthogonal}")
    
    # Compute Frobenius norm of deviation from orthogonality
    orthog_error = np.linalg.norm(R @ R.T - np.eye(R.shape[0]))
    print(f"Orthogonality error: {orthog_error:.6f}\n")
    
    # Sanity check: compute Frobenius errors before and after alignment
    before = np.linalg.norm(Y - X, ord="fro")
    after  = np.linalg.norm(Y - X @ R, ord="fro")
    after_T = np.linalg.norm(Y - X @ R.T, ord="fro")

    print(f"Frobenius error (no alignment): {before:.6f}")
    print(f"Frobenius error (X @ R):        {after:.6f}")
    print(f"Frobenius error (X @ R.T):      {after_T:.6f}\n\n")

    # Step 5: Apply alignment
    print(f"{'='*60}")
    print("Applying alignment to 1980s embeddings...")
    print(f"{'='*60}")

    # non allinei entrambi
    # scegli uno spazio di riferimento (qui: 1990s)
    # allinei solo lo spazio precedente
    E_1980s_aligned = E_1980s @ R
    
    print(f"Aligned 1980s embeddings shape: {E_1980s_aligned.shape}\n")
    
    # Step 6: Sanity check - compute cosine similarities for "computer"
    if "computer" in vocab_1980s and "computer" in vocab_1990s:
        print(f"{'='*60}")
        print("Sanity Check: Word 'computer' alignment")
        print(f"{'='*60}")
        
        idx_1980s = vocab_1980s["computer"]
        idx_1990s = vocab_1990s["computer"]
        
        # Embeddings before and after alignment
        vec_1980s_original = E_1980s[idx_1980s]
        vec_1980s_aligned = E_1980s_aligned[idx_1980s]
        vec_1990s = E_1990s[idx_1990s]
        
        # Compute cosine similarities
        sim_before = cosine_similarity(vec_1980s_original, vec_1990s)
        sim_after = cosine_similarity(vec_1980s_aligned, vec_1990s)
        
        print(f"Cosine similarity BEFORE alignment: {sim_before:.6f}")
        print(f"Cosine similarity AFTER alignment:  {sim_after:.6f}")
        print(f"Improvement: {sim_after - sim_before:+.6f}\n")
    else:
        print("Word 'computer' not found in both decades.\n")


    # Step 7: Sanity check - compute cosine similarities for all anchors
    print("="*60)
    print("Sanity Check: Anchor words alignment")
    print("="*60)

    anchor_deltas = []
    for w in anchors:
        if w not in vocab_1980s or w not in vocab_1990s:
            print(f"Skipping '{w}' (not in both vocabularies)")
            continue

        i80 = vocab_1980s[w]
        i90 = vocab_1990s[w]

        before = cosine_similarity(E_1980s[i80], E_1990s[i90])
        after  = cosine_similarity(E_1980s_aligned[i80], E_1990s[i90])
        delta = after - before
        anchor_deltas.append(delta)

        print(f"{w:<8} cos before: {before:+.6f} | cos after: {after:+.6f} | delta: {delta:+.6f}")

    if anchor_deltas:
        print(f"\nMean anchor delta: {sum(anchor_deltas)/len(anchor_deltas):+.6f}\n")
    else:
        print("\nNo valid anchors for check.\n")


    # Step 8: Semantic drift analysis on common vocabulary
    print("="*60)
    print("Semantic drift (1980s -> 1990s) on common vocabulary")
    print("="*60)

    common_words = set(vocab_1980s.keys()) & set(vocab_1990s.keys())
    common_words = [w for w in common_words if w not in anchors]  # exclude anchors (optional)

    drift_scores = []
    for w in common_words:
        i80 = vocab_1980s[w]
        i90 = vocab_1990s[w]

        sim = cosine_similarity(E_1980s_aligned[i80], E_1990s[i90])
        drift = 1.0 - sim
        drift_scores.append((w, drift, sim))

    # Sort by drift descending (largest change first)
    drift_scores.sort(key=lambda x: x[1], reverse=True)

    print(f"Common words (excluding anchors): {len(common_words)}")
    print("\nTop-10 drifting words:")
    for w, drift, sim in drift_scores[:10]:
        print(f"{w:<12} drift: {drift:+.6f} | cosine(after): {sim:+.6f}")

    # Also explicitly print 'computer' if present
    for w, drift, sim in drift_scores:
        if w == "computer":
            print(f"\n'computer' drift: {drift:+.6f} | cosine(after): {sim:+.6f}")
            break

    
    return anchors, E_1980s_aligned, E_1990s, R


if __name__ == "__main__":
    anchors, E_1980s_aligned, E_1990s, R = main()
