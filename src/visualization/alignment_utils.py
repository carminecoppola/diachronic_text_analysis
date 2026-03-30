"""
Utilities for aligning embedding spaces across decades.

This module implements forward and backward composition of alignment
transformations so that embeddings from any decade can be mapped into
the space of an arbitrary reference decade.
"""

from pathlib import Path
import numpy as np
import torch

from src.config import DECADES


def load_alignment_pair(alignment_dir: Path, dec_from: str, dec_to: str) -> dict:
    """
    Load a single alignment transformation between two consecutive decades.

    Each alignment file is expected to contain:
        - R: orthogonal rotation matrix
        - mu_X: centroid of source space
        - mu_Y: centroid of target space

    The transformation is defined as:
        E_to = (E_from - mu_X) @ R + mu_Y

    Parameters
    ----------
    alignment_dir : Path
        Directory containing alignment files.
    dec_from : str
        Source decade.
    dec_to : str
        Target decade.

    Returns
    -------
    dict
        Dictionary containing 'R', 'mu_X', and 'mu_Y'.
    """
    p = alignment_dir / f"alignment_{dec_from}_to_{dec_to}.pt"
    if not p.exists():
        raise FileNotFoundError(f"Missing alignment file: {p}")

    obj = torch.load(p, map_location="cpu")

    # Ensure required keys are present
    for k in ("R", "mu_X", "mu_Y"):
        if k not in obj:
            raise KeyError(f"Missing key '{k}' in {p}")

    return obj


def apply_pair_transform(E: np.ndarray, R: torch.Tensor,
                         mu_X: torch.Tensor, mu_Y: torch.Tensor) -> np.ndarray:
    """
    Apply a forward alignment transform.

    Maps embeddings from the source decade into the target decade space.

    Parameters
    ----------
    E : np.ndarray
        Embedding matrix in source space.
    R : torch.Tensor
        Orthogonal rotation matrix.
    mu_X : torch.Tensor
        Source centroid.
    mu_Y : torch.Tensor
        Target centroid.

    Returns
    -------
    np.ndarray
        Transformed embeddings in the target space.
    """
    Rn = R.detach().cpu().numpy()
    mx = mu_X.detach().cpu().numpy()
    my = mu_Y.detach().cpu().numpy()
    return (E - mx) @ Rn + my


def apply_pair_inverse_transform(E_to: np.ndarray, R: torch.Tensor,
                                 mu_X: torch.Tensor, mu_Y: torch.Tensor) -> np.ndarray:
    """
    Apply the inverse of an alignment transform.

    Since R is orthogonal (from Procrustes alignment),
    the inverse rotation is simply R^T.

    Parameters
    ----------
    E_to : np.ndarray
        Embeddings in the target space.
    R : torch.Tensor
        Orthogonal rotation matrix.
    mu_X : torch.Tensor
        Source centroid.
    mu_Y : torch.Tensor
        Target centroid.

    Returns
    -------
    np.ndarray
        Embeddings mapped back to the source space.
    """
    Rn = R.detach().cpu().numpy()
    mx = mu_X.detach().cpu().numpy()
    my = mu_Y.detach().cpu().numpy()
    return (E_to - my) @ Rn.T + mx


def align_embeddings_to_ref(alignment_dir: Path,
                            E_dec: np.ndarray,
                            decade: str,
                            ref_decade: str) -> np.ndarray:
    """
    Align embeddings from an arbitrary decade into the space of a reference decade.

    This function supports:
    - forward alignment (older -> newer)
    - backward alignment (newer -> older) via inverse transforms

    The alignment is performed by composing pairwise transformations
    between consecutive decades.

    Parameters
    ----------
    alignment_dir : Path
        Directory containing alignment files.
    E_dec : np.ndarray
        Embeddings of the input decade.
    decade : str
        Input decade.
    ref_decade : str
        Reference decade.

    Returns
    -------
    np.ndarray
        Embeddings mapped into the reference decade space.
    """
    if decade not in DECADES:
        raise ValueError(f"decade={decade} not in DECADES={DECADES}")
    if ref_decade not in DECADES:
        raise ValueError(f"ref_decade={ref_decade} not in DECADES={DECADES}")

    i = DECADES.index(decade)
    j = DECADES.index(ref_decade)

    E = E_dec

    # No alignment needed if already in reference space
    if i == j:
        return E

    # Forward alignment: compose transforms from decade -> ref_decade
    if i < j:
        for k in range(i, j):
            d_from = DECADES[k]
            d_to = DECADES[k + 1]
            T = load_alignment_pair(alignment_dir, d_from, d_to)
            E = apply_pair_transform(E, T["R"], T["mu_X"], T["mu_Y"])
        return E

    # Backward alignment: apply inverse transforms from decade -> ref_decade
    for k in range(i - 1, j - 1, -1):
        d_from = DECADES[k]
        d_to = DECADES[k + 1]
        T = load_alignment_pair(alignment_dir, d_from, d_to)
        E = apply_pair_inverse_transform(E, T["R"], T["mu_X"], T["mu_Y"])

    return E
