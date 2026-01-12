from pathlib import Path

import numpy as np
import torch

from src.config import DECADES


def load_alignment_pair(alignment_dir: Path, dec_from: str, dec_to: str) -> dict:
    p = alignment_dir / f"alignment_{dec_from}_to_{dec_to}.pt"
    if not p.exists():
        raise FileNotFoundError(f"Missing alignment file: {p}")

    obj = torch.load(p, map_location="cpu")
    for k in ("R", "mu_X", "mu_Y"):
        if k not in obj:
            raise KeyError(f"Missing key '{k}' in {p}. Keys={list(obj.keys())}")
    return obj


def apply_pair_transform(E: np.ndarray, R: torch.Tensor, mu_X: torch.Tensor, mu_Y: torch.Tensor) -> np.ndarray:
    Rn = R.detach().cpu().numpy()
    mx = mu_X.detach().cpu().numpy()
    my = mu_Y.detach().cpu().numpy()
    return (E - mx) @ Rn + my


def align_embeddings_to_ref(alignment_dir: Path, E_dec: np.ndarray, decade: str, ref_decade: str) -> np.ndarray:
    """
    Compone gli alignment decade->...->ref_decade (forward), assumendo ref >= decade nella lista DECADES.
    """
    i = DECADES.index(decade)
    j = DECADES.index(ref_decade)
    if i > j:
        raise ValueError("ref_decade must be later (more recent) than decade for forward composition.")

    E = E_dec
    for k in range(i, j):
        d_t = DECADES[k]
        d_t1 = DECADES[k + 1]
        T = load_alignment_pair(alignment_dir, d_t, d_t1)
        E = apply_pair_transform(E, T["R"], T["mu_X"], T["mu_Y"])
    return E
