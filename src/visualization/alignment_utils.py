from pathlib import Path

import numpy as np
import torch

from src.config import DECADES


def load_alignment_pair(alignment_dir: Path, dec_from: str, dec_to: str) -> dict:
    """
    Load alignment file mapping embeddings from dec_from -> dec_to:
        E_to = (E_from - mu_X) @ R + mu_Y
    """
    p = alignment_dir / f"alignment_{dec_from}_to_{dec_to}.pt"
    if not p.exists():
        raise FileNotFoundError(f"Missing alignment file: {p}")

    obj = torch.load(p, map_location="cpu")
    for k in ("R", "mu_X", "mu_Y"):
        if k not in obj:
            raise KeyError(f"Missing key '{k}' in {p}. Keys={list(obj.keys())}")
    return obj


def apply_pair_transform(E: np.ndarray, R: torch.Tensor, mu_X: torch.Tensor, mu_Y: torch.Tensor) -> np.ndarray:
    """
    Forward transform:
        E' = (E - mu_X) @ R + mu_Y
    """
    Rn = R.detach().cpu().numpy()
    mx = mu_X.detach().cpu().numpy()
    my = mu_Y.detach().cpu().numpy()
    return (E - mx) @ Rn + my


def apply_pair_inverse_transform(E_to: np.ndarray, R: torch.Tensor, mu_X: torch.Tensor, mu_Y: torch.Tensor) -> np.ndarray:
    """
    Inverse transform (assuming R is orthogonal from Procrustes):
        E_from = (E_to - mu_Y) @ R^T + mu_X
    """
    Rn = R.detach().cpu().numpy()
    mx = mu_X.detach().cpu().numpy()
    my = mu_Y.detach().cpu().numpy()
    return (E_to - my) @ Rn.T + mx


def align_embeddings_to_ref(alignment_dir: Path, E_dec: np.ndarray, decade: str, ref_decade: str) -> np.ndarray:
    """
    Align embeddings from 'decade' into the space of 'ref_decade'.

    Works both directions:
    - if decade older than ref -> compose forward transforms
    - if decade newer than ref -> compose inverse transforms
    """
    if decade not in DECADES:
        raise ValueError(f"decade={decade} not in DECADES={DECADES}")
    if ref_decade not in DECADES:
        raise ValueError(f"ref_decade={ref_decade} not in DECADES={DECADES}")

    i = DECADES.index(decade)
    j = DECADES.index(ref_decade)

    E = E_dec

    # same decade → nothing to do
    if i == j:
        return E

    # ---------- FORWARD: decade -> ... -> ref ----------
    if i < j:
        for k in range(i, j):
            d_from = DECADES[k]
            d_to = DECADES[k + 1]
            T = load_alignment_pair(alignment_dir, d_from, d_to)
            E = apply_pair_transform(E, T["R"], T["mu_X"], T["mu_Y"])
        return E

    # ---------- BACKWARD: decade -> ... -> ref ----------
    # invert each (older -> newer) alignment
    for k in range(i - 1, j - 1, -1):
        d_from = DECADES[k]       # older
        d_to = DECADES[k + 1]     # newer
        T = load_alignment_pair(alignment_dir, d_from, d_to)
        E = apply_pair_inverse_transform(E, T["R"], T["mu_X"], T["mu_Y"])

    return E
