import json
from pathlib import Path

import numpy as np
import torch


def load_vocab(vocab_path: Path) -> dict:
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embedding_matrix(models_dir: Path, decade: str) -> np.ndarray:
    pt_path = models_dir / f"emb_{decade}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing embedding file: {pt_path}")

    state_dict = torch.load(pt_path, map_location="cpu")
    key = "embeddings.weight"
    if key not in state_dict:
        raise KeyError(f"Key '{key}' not found in {pt_path}. Keys={list(state_dict.keys())}")

    return state_dict[key].detach().cpu().numpy()
