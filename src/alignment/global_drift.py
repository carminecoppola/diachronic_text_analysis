import json
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DECADES

PROJECT_ROOT = Path(__file__).parent.parent.parent

MODELS_DIR = PROJECT_ROOT / "models" / "5gram"
VOCAB_FILE = PROJECT_ROOT / "data" / "processed" / "5gram" / "vocab.json"
ALIGNMENT_DIR = PROJECT_ROOT / "src" / "alignment" / "transforms"
OUT_DIR = PROJECT_ROOT / "src" / "alignment" / "global_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_common_vocab(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_embedding_matrix_pt(decade: str) -> np.ndarray:
    pt_path = MODELS_DIR / f"emb_{decade}.pt"
    state = torch.load(pt_path, map_location="cpu")
    return state["embeddings.weight"].detach().cpu().numpy()

def load_alignment_pair(d_t: str, d_t1: str):
    path = ALIGNMENT_DIR / f"alignment_{d_t}_to_{d_t1}.pt"
    obj = torch.load(path, map_location="cpu")
    return {
        "R": obj["R"].numpy(),
        "mu_X": obj["mu_X"].numpy(),
        "mu_Y": obj["mu_Y"].numpy(),
    }

def apply_pair_transform(E: np.ndarray, R: np.ndarray, mu_X: np.ndarray, mu_Y: np.ndarray) -> np.ndarray:
    return (E - mu_X) @ R + mu_Y

def align_decade_to_reference(decade: str, ref_decade: str, E_dec: np.ndarray) -> np.ndarray:
    """
    Align embeddings of `decade` into the space of `ref_decade`,
    by applying consecutive pairwise transforms decade->decade+1->...->ref.
    """
    if decade == ref_decade:
        return E_dec

    i = DECADES.index(decade)
    j = DECADES.index(ref_decade)
    if i > j:
        raise ValueError("This pipeline assumes ref_decade is later than decade (forward composition).")

    E = E_dec
    for k in range(i, j):
        d_t = DECADES[k]
        d_t1 = DECADES[k + 1]
        T = load_alignment_pair(d_t, d_t1)
        E = apply_pair_transform(E, T["R"], T["mu_X"], T["mu_Y"])
    return E

def cosine(u, v):
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))

def main():
    vocab = load_common_vocab(VOCAB_FILE)
    ref = "2010s"
    E_ref = load_embedding_matrix_pt(ref)

    # E_ref è già nello spazio di riferimento
    # Puoi fare drift globale: decade -> ref per ogni parola
    # Esempio: traiettoria di una parola
    target_word = "computer"
    idx = vocab.get(target_word, None)
    if idx is None:
        print(f"'{target_word}' not in vocab")
        return

    trajectory = []
    for d in DECADES:
        E_d = load_embedding_matrix_pt(d)
        E_d_aligned = align_decade_to_reference(d, ref, E_d)
        sim = cosine(E_d_aligned[idx], E_ref[idx])
        drift = 1.0 - sim
        trajectory.append({"decade": d, "cosine_to_2010s": sim, "drift_to_2010s": drift})

    out_path = OUT_DIR / f"trajectory_{target_word}_to_{ref}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path.name}")

if __name__ == "__main__":
    main()
