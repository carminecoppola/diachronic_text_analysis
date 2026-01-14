from pathlib import Path

import numpy as np


def cosine_topk(query_vec: np.ndarray, E: np.ndarray, k: int = 10, exclude_index: int | None = None):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    sims = En @ q

    if exclude_index is not None:
        sims[exclude_index] = -np.inf

    k = min(k, len(sims) - 1)
    top_idx = np.argpartition(-sims, kth=k)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return top_idx, sims[top_idx]


def write_neighbors_csv(
    out_csv: Path,
    words: list[str],
    decade: str,
    vocab: dict,
    inv_vocab: dict,
    E_aligned: np.ndarray,
    topk: int,
):
    rows = []
    for w in words:
        idx = vocab[w]
        top_idx, top_sims = cosine_topk(E_aligned[idx], E_aligned, k=topk, exclude_index=idx)
        for rank, (ni, sim) in enumerate(zip(top_idx, top_sims), start=1):
            rows.append((w, decade, rank, inv_vocab.get(int(ni), f"idx_{ni}"), float(sim)))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("word,decade,rank,neighbor,cosine\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n")
