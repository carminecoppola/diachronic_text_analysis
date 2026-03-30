"""
Nearest-neighbor utilities for aligned embeddings.

This module computes cosine similarities and extracts top-k nearest
neighbors for a given target word embedding.
"""

import numpy as np


def cosine_topk(vec: np.ndarray,
                mat: np.ndarray,
                k: int,
                exclude_index: int | None = None):
    """
    Compute top-k cosine similarities between a vector and all rows of a matrix.

    Parameters
    ----------
    vec : np.ndarray
        Target vector of shape (D,).
    mat : np.ndarray
        Matrix of candidate vectors of shape (N, D).
    k : int
        Number of nearest neighbors to return.
    exclude_index : int or None
        Optional index to exclude (typically the target word itself).

    Returns
    -------
    (indices, similarities)
        indices : np.ndarray
            Indices of the top-k neighbors.
        similarities : np.ndarray
            Corresponding cosine similarity values.
    """
    # Normalize vectors to unit length for cosine similarity
    vec_n = vec / np.linalg.norm(vec)
    mat_n = mat / np.linalg.norm(mat, axis=1, keepdims=True)

    # Compute cosine similarity as dot product
    sims = mat_n @ vec_n

    # Optionally exclude the target word itself
    if exclude_index is not None:
        sims[exclude_index] = -np.inf

    # Extract top-k indices
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]

    return idx, sims[idx]


def write_neighbors_csv(out_csv,
                        words,
                        decade,
                        vocab,
                        inv_vocab,
                        E_aligned,
                        topk):
    """
    Write nearest neighbors for each target word into a CSV file.

    Each row has the format:
        word,decade,rank,neighbor,cosine

    Parameters
    ----------
    out_csv : Path
        Output CSV path.
    words : list[str]
        Target words.
    decade : str
        Decade identifier.
    vocab : dict[str, int]
        Mapping from word to index.
    inv_vocab : dict[int, str]
        Mapping from index to word.
    E_aligned : np.ndarray
        Aligned embedding matrix.
    topk : int
        Number of neighbors to extract.
    """
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("word,decade,rank,neighbor,cosine\n")

        for w in words:
            if w not in vocab:
                continue

            wi = vocab[w]
            idx, sims = cosine_topk(E_aligned[wi], E_aligned,
                                    k=topk,
                                    exclude_index=wi)

            for r, (ni, s) in enumerate(zip(idx, sims), start=1):
                f.write(f"{w},{decade},{r},{inv_vocab[ni]},{float(s)}\n")
