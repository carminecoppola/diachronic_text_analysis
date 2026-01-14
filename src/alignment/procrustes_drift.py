"""
Anchor word selection and Orthogonal Procrustes alignment for diachronic embeddings.

1. Selects anchor words using frequency-based filtering
2. Loads CBOW embeddings from independently trained models
3. Aligns embedding spaces via Orthogonal Procrustes

"""

import json
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DECADES

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
FREQUENCIES_DIR = PROJECT_ROOT / "alignment" / "5gram" / "frequencies"
MODELS_DIR = PROJECT_ROOT / "models"/ "5gram"  
VOCAB_FILE = PROJECT_ROOT / "data" / "processed" / "5gram" / "vocab.json"
DRIFT_OUTPUT_DIR = PROJECT_ROOT / "src" / "alignment" / "drift_results"
ALIGNMENT_DIR = PROJECT_ROOT / "src" / "alignment" / "transforms"
ALIGNMENT_DIR.mkdir(parents=True, exist_ok=True)

def load_common_vocab(vocab_file: Path = VOCAB_FILE) -> dict:
    with open(vocab_file, "r", encoding="utf-8") as f:
        return json.load(f)

def load_frequencies(decade, freq_dir: Path = FREQUENCIES_DIR) -> dict:
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

def load_all_frequencies(decades, freq_dir: Path = FREQUENCIES_DIR) -> dict:
    """
    Load word frequency dictionaries for all decades.

    Args:
        decades: list of decade strings (e.g. ["1900s", "1910s", ...])
        freq_dir: directory containing frequency JSON files

    Returns:
        Dictionary: {decade -> frequency_dict}
    """
    freq_by_decade = {}

    for decade in decades:
        freq_by_decade[decade] = load_frequencies(decade, freq_dir)

    return freq_by_decade


def load_embedding_matrix_pt(decade: str, models_dir: Path = MODELS_DIR) -> np.ndarray:
    """
    Load CBOW input embedding matrix (word embeddings) from a PyTorch .pt state_dict.

    We extract: state_dict["embeddings.weight"] with shape (vocab_size, embedding_dim).
    """
    pt_path = models_dir / f"emb_{decade}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing embedding file: {pt_path}")

    state_dict = torch.load(pt_path, map_location="cpu")

    key = "embeddings.weight"
    if key not in state_dict:
        raise KeyError(
            f"Key '{key}' not found in {pt_path}.\n"
            f"Available keys: {list(state_dict.keys())}"
        )

    E = state_dict[key].detach().cpu().numpy()
    return E


def compute_global_min_count(freq_by_decade: dict, ratio: float = 1e-5, min_floor: int = 50) -> int:
    """
    Compute a global min_count based on all decades.

    Args:
        freq_by_decade: {decade -> freq_dict}
        ratio: fraction of total tokens to use
        min_floor: minimum threshold

    Returns:
        int: global min_count
    """
    total_tokens = 0
    for freq in freq_by_decade.values():
        total_tokens += sum(freq.values())

    avg_tokens = total_tokens / len(freq_by_decade)
    return max(min_floor, int(ratio * avg_tokens))



def select_anchor_words(freq_t: dict, freq_t1: dict, min_count: int) -> list:
    """
    Select anchor words shared by two consecutive decades.
    
    Anchor words are those that:
    1. Appear in both decades
    2. Have frequency >= min_count in both decades
    
    Args:
        freq_t: Frequency dictionary for decade t
        freq_t1: Frequency dictionary for decade t+1
        min_count: Minimum token frequency required in both decades
    
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
        if freq_t[word] >= min_count and freq_t1[word] >= min_count 
    ]
    
    # Return sorted for reproducibility
    return sorted(anchors)


def build_alignment_matrices(anchor_words: list, E_t: np.ndarray, E_t1: np.ndarray, vocab: dict) -> tuple:
    
    """
    Build alignment matrices X, Y from anchor embeddings for two decades,
    assuming a COMMON vocabulary mapping word -> index shared across decades.

    Skips anchor words that are missing from vocab or whose index is out of range.
    """
     
    X_rows = []
    Y_rows = []
    used_anchors = []
    
    for word in anchor_words:
        # Get indices in both decades
        idx = vocab.get(word, None)
        if idx is None:
            print(f"Skipping anchor '{word}': not in vocab")
            continue
        if idx >= E_t.shape[0] or idx >= E_t1.shape[0]:
            print(f"Skipping anchor '{word}': index {idx} out of range")
            continue
        
        # Get embeddings
        X_rows.append(E_t[idx])
        Y_rows.append(E_t1[idx])
        used_anchors.append(word)
    
    if not X_rows or not Y_rows:
        raise ValueError("No valid anchor words found for building alignment matrices.")
    
    X = np.vstack(X_rows)  # shape: (|anchors|, d)
    Y = np.vstack(Y_rows)  # shape: (|anchors|, d)
    
    return X, Y, used_anchors


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

def cosine_all_rows(A, B):
    # A,B: (V,d)
    num = np.sum(A * B, axis=1)
    den = (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))
    return num / den  # (V,)


def get_valid_drift_indices(vocab: dict, freq_t: dict, freq_t1: dict, used_anchors: list, min_drift_count: int, V: int) -> np.ndarray:
    """
    Return indices of words eligible (excludes the "rare words") for semantic drift analysis.

    Conditions:
    - not an anchor word
    - frequency >= min_drift_count in both decades

    Args:
        vocab: common vocab {word -> index}
        freq_t: frequency dict for decade t
        freq_t1: frequency dict for decade t+1
        used_anchors: list of anchor words actually used in alignment
        min_drift_count: minimum frequency threshold
        V: vocabulary size (embedding rows)

    Returns:
        np.ndarray of valid word indices
    """
    mask = np.ones(V, dtype=bool)

    # Exclude anchors from drift analysis
    anchor_idx = np.array(
        [vocab[w] for w in used_anchors if w in vocab],
        dtype=int
    )
    mask[anchor_idx] = False

    valid_idx = []
    for w, idx in vocab.items():
        if idx >= V:
            continue
        if not mask[idx]:
            continue
        if freq_t.get(w, 0) >= min_drift_count and freq_t1.get(w, 0) >= min_drift_count:
            valid_idx.append(idx)

    return np.array(valid_idx, dtype=int)


def main():
    """Main execution: anchor selection + Procrustes alignment."""
    
    print(f"Loading frequency files from: {FREQUENCIES_DIR}\n")
    
    # Step 1: Load frequencies and select anchors
    print(f"{'='*60}")
    print(f"Loading frequencies for {len(DECADES)} decades...")
    print(f"{'='*60}")

    freq_by_decade = load_all_frequencies(DECADES)

    for decade in DECADES:
        print(f"{decade} vocabulary size: {len(freq_by_decade[decade])}")
    print()


    # Select anchor words with global_min_count
    global_min_count = compute_global_min_count(freq_by_decade)
    # Load common vocabulary
    vocab = load_common_vocab()

    for i in range(len(DECADES) - 1):
        d_t = DECADES[i]
        d_t1 = DECADES[i + 1]

        anchors = select_anchor_words(freq_by_decade[d_t], freq_by_decade[d_t1], min_count=global_min_count)
    
        # Print anchor selection results
        print(f"{'='*60}")
        print(f"Anchor selection: {d_t} -> {d_t1}")
        print(f"{'='*60}")
        print(f"min_count = {global_min_count}")
        print(f"Number of anchors: {len(anchors)}\n")

        # Sort anchors by average frequency (descending)
        anchors_sorted = sorted(
            anchors,
            key=lambda w: (freq_by_decade[d_t][w] + freq_by_decade[d_t1][w]) / 2,
            reverse=True
        )

        # print("Top-10 anchor words (by avg frequency):")
        # print(f"{'Word':<15} {d_t:<10} {d_t1:<10}")
        # print("-" * 35)

        # for word in anchors_sorted[:10]:
        #     c_t = freq_by_decade[d_t][word]
        #     c_t1 = freq_by_decade[d_t1][word]
        #     print(f"{word:<15} {c_t:<10} {c_t1:<10}")

        # print()

        # Step 2: Load CBOW embeddings
        print(f"{'='*60}")
        print("Loading CBOW embeddings...")
        print(f"{'='*60}")

        E_t = load_embedding_matrix_pt(d_t)
        E_t1 = load_embedding_matrix_pt(d_t1)

        #Sanity check: embedding rows vs vocab size
        print(f"E_{d_t} shape:", E_t.shape)
        print("vocab size:", len(vocab))

        if E_t.shape[0] != len(vocab):
            print("⚠️ WARNING: embedding rows != vocab size")
            print("This means training vocab != common vocab, alignment will break.")
        else:
            print("✓ Rows match vocab size: OK\n")
        
        print(f"E_{d_t1} shape:", E_t1.shape)
        print("vocab size:", len(vocab))
        if E_t1.shape[0] != len(vocab):
            print("⚠️ WARNING: embedding rows != vocab size")
            print("This means training vocab != common vocab, alignment will break.")
        else:
            print("✓ Rows match vocab size: OK\n")
    
        # Step 3: Build alignment matrices
        print(f"{'='*60}")
        print("Building alignment matrices...")
        print(f"{'='*60}")
        
        X, Y, used_anchors = build_alignment_matrices(anchors, E_t, E_t1, vocab)
        
        print(f"Source matrix X ({d_t} anchors): {X.shape}")
        print(f"Target matrix Y ({d_t1} anchors): {Y.shape}\n")
        print(f"Anchors requested: {len(anchors)}, used: {len(used_anchors)}\n")

        # Step 4: Compute Orthogonal Procrustes
        print(f"{'='*60}")
        print("Computing Orthogonal Procrustes alignment...")
        print(f"{'='*60}")
        
        # Center the matrices
        mu_X = X.mean(axis=0, keepdims=True)   # mean of the anchor in the space t 
        mu_Y = Y.mean(axis=0, keepdims=True)   # mean of the anchor in the space t+1

        Xc = X - mu_X
        Yc = Y - mu_Y

        R = orthogonal_procrustes(Xc, Yc)
        
        print(f"Rotation matrix R shape: {R.shape}")
        
        # Check orthogonality: R @ R^T should be identity (within numerical tolerance)
        is_orthogonal = np.allclose(R @ R.T, np.eye(R.shape[0]), atol=1e-6)
        print(f"Is orthogonal (within 1e-6 tolerance)? {is_orthogonal}")
        
        # Compute Frobenius norm of deviation from orthogonality
        orthog_error = np.linalg.norm(R @ R.T - np.eye(R.shape[0]))
        print(f"Orthogonality error: {orthog_error:.6f}\n")
        
        # Sanity check: compute Frobenius errors before and after alignment
        before = np.linalg.norm(Yc - Xc, ord="fro")
        after  = np.linalg.norm(Yc - Xc @ R, ord="fro")
        after_T = np.linalg.norm(Yc - Xc @ R.T, ord="fro")

        print(f"Frobenius error (no alignment): {before:.6f}")
        print(f"Frobenius error (X @ R):        {after:.6f}")
        print(f"Frobenius error (X @ R.T):      {after_T:.6f}\n\n")

        # Save alignment object
        alignment_obj = {
            "from": d_t,
            "to": d_t1,
            "R": torch.from_numpy(R).float(),
            "mu_X": torch.from_numpy(mu_X).float(),
            "mu_Y": torch.from_numpy(mu_Y).float(),
            "n_anchors": len(used_anchors),
            "min_count": global_min_count,
        }

        out_path = ALIGNMENT_DIR / f"alignment_{d_t}_to_{d_t1}.pt"

        torch.save(alignment_obj, out_path)

        print(f"Saved alignment: {out_path.name}\n")
    
        # Step 5: Apply alignment
        print(f"{'='*60}")
        print(f"Applying alignment to {d_t} embeddings...")
        print(f"{'='*60}")

        # non allinei entrambi
        # scegli uno spazio di riferimento
        # allinei solo lo spazio precedente
        E_t_aligned = (E_t - mu_X) @ R + mu_Y   # centering also E_t and R
        
        print(f"Aligned {d_t} embeddings shape: {E_t_aligned.shape}\n")

        # Step 6: Sanity check - compute cosine similarities for "computer"
        target_word = "computer"   # cambia se vuoi

        if target_word in vocab:
            idx = vocab[target_word]
            vec_t_orig = E_t[idx]
            vec_t_algn = E_t_aligned[idx]
            vec_t1 = E_t1[idx]

            sim_before = cosine_similarity(vec_t_orig, vec_t1)
            sim_after  = cosine_similarity(vec_t_algn, vec_t1)

            print(f"{'='*60}")
            print(f"Sanity check: '{target_word}' ({d_t} -> {d_t1})")
            print(f"{'='*60}")
            print(f"Cosine BEFORE: {sim_before:.6f}")
            print(f"Cosine AFTER:  {sim_after:.6f}")
            print(f"Delta:         {sim_after - sim_before:+.6f}\n")
        else:
            print(f"Target word '{target_word}' not in common vocab.\n")


        # Step 7: Sanity check - compute cosine similarities for some used anchors
        print(f"{'='*60}")
        print(f"Sanity check: anchor alignment ({d_t} -> {d_t1})")
        print(f"{'='*60}")

        sample = used_anchors[:50]  # per non stampare troppo; oppure random sample
        deltas = []

        for w in sample:
            idx = vocab[w]
            b = cosine_similarity(E_t[idx], E_t1[idx])
            a = cosine_similarity(E_t_aligned[idx], E_t1[idx])
            deltas.append(a - b)

        mean_delta = float(np.mean(deltas)) if deltas else 0.0
        
        print(f"Anchor sample size: {len(sample)}")
        print(f"Mean cosine delta (after-before): {mean_delta:+.6f}\n")


        # Step 8: Semantic drift analysis on common vocabulary
        print("="*60)
        print("Semantic drift on common vocabulary")
        print("="*60)

        V = E_t1.shape[0]

        cos = cosine_all_rows(E_t_aligned, E_t1)
        drift = 1.0 - cos

        # Exclude anchors from drift analysis
        min_drift_count = 190  # puoi cambiarlo quando vuoi

        valid_idx = get_valid_drift_indices(
            vocab=vocab,
            freq_t=freq_by_decade[d_t],
            freq_t1=freq_by_decade[d_t1],
            used_anchors=used_anchors,
            min_drift_count=min_drift_count,
            V=V
        )
        print(f"valid_idx size: {len(valid_idx)} (min_drift_count={min_drift_count})\n")
        if len(valid_idx) == 0:
            print("⚠️ No eligible words for drift. Lower min_drift_count or relax filters.\n")
            continue

        # Top-K drifting words
        K = 50
        idx_sorted = np.argsort(drift[valid_idx])[::-1]
        top_idx = valid_idx[idx_sorted[:K]]

        # inverti vocab per stampare le parole
        idx2word = {i:w for w,i in vocab.items()}

        print("="*60)
        print(f"Top-{K} drifting words ({d_t} -> {d_t1})")
        print("="*60)
        for i in top_idx[:10]:
            w = idx2word[i]
            print(f"{w:<15} drift: {drift[i]:+.6f} | cosine(after): {cos[i]:+.6f}")

        # Step 9: Save drift results to JSON
        out_data = [
            {
                "word": idx2word[i],
                "drift": float(drift[i]),
                "cosine": float(cos[i]),
                "from": d_t,
                "to": d_t1,
            }
            for i in top_idx
        ]

        out_file = DRIFT_OUTPUT_DIR / f"drift_{d_t}_to_{d_t1}.json"

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)

        print(f"\nSaved: drift_{d_t}_to_{d_t1}.json")

    return  



if __name__ == "__main__":
    main()
    