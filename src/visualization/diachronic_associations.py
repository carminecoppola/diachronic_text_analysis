import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.config import DECADES, MODELS_DIR, VOCAB_FILE, ALIGNMENT_TRANSFORMS_DIR
from src.visualization.drift_io import load_vocab, load_embedding_matrix
from src.visualization.alignment_utils import align_embeddings_to_ref
from src.visualization.neighbors import cosine_topk


DECADE_RE = re.compile(r"^(\d{4})s$")


def decade_to_year(dec: str) -> int:
    m = DECADE_RE.match(dec)
    if not m:
        raise ValueError(f"Invalid decade format: {dec!r}")
    return int(m.group(1))


def compute_neighbors_over_time(word: str, vocab: dict, inv_vocab: dict, aligned_by_decade: dict[str, np.ndarray], topk: int):
    if word not in vocab:
        raise ValueError(f"Word {word!r} not in vocab.")

    rows = []
    widx = vocab[word]
    for dec in DECADES:
        E = aligned_by_decade[dec]
        top_idx, top_sims = cosine_topk(E[widx], E, k=topk, exclude_index=widx)
        for rank, (ni, sim) in enumerate(zip(top_idx, top_sims), start=1):
            rows.append({
                "word": word,
                "decade": dec,
                "year": decade_to_year(dec),
                "rank": rank,
                "neighbor": inv_vocab.get(int(ni), f"idx_{ni}"),
                "cosine": float(sim),
            })
    return rows


def plot_diachronic_associations(rows: list[dict], out_png: Path, title: str, min_cos: float | None = None):
    # group points
    years = np.array([r["year"] for r in rows], dtype=float)
    cos = np.array([r["cosine"] for r in rows], dtype=float)
    labels = [r["neighbor"] for r in rows]

    plt.figure(figsize=(12, 4))
    plt.scatter(years, cos, s=28)

    # annotate each point
    for x, y, lab in zip(years, cos, labels):
        plt.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)

    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Cosine similarity to target")

    if min_cos is not None:
        plt.ylim(bottom=min_cos)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Diachronic associations plot: neighbors & cosine over decades.")
    ap.add_argument("--word", required=True, help="Target word (must be in vocab).")
    ap.add_argument("--ref", default="2010s", help="Reference decade for alignment (any decade supported).")
    ap.add_argument("--topk", type=int, default=6, help="Top-k neighbors per decade.")
    ap.add_argument("--out_dir", default="drift_plots", help="Output directory.")
    ap.add_argument("--min_cos", type=float, default=None, help="Optional y-axis lower bound.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alignment_dir = Path(ALIGNMENT_TRANSFORMS_DIR)
    vocab = load_vocab(VOCAB_FILE)
    inv_vocab = {idx: w for w, idx in vocab.items()}

    if args.ref not in DECADES:
        raise ValueError(f"ref decade {args.ref} not in DECADES={DECADES}")

    # Align each decade to ref
    aligned_by_decade = {}
    for dec in DECADES:
        E_dec = load_embedding_matrix(MODELS_DIR, dec)
        E_aligned = align_embeddings_to_ref(alignment_dir, E_dec, dec, args.ref)
        aligned_by_decade[dec] = E_aligned

    rows = compute_neighbors_over_time(args.word, vocab, inv_vocab, aligned_by_decade, topk=args.topk)

    # write CSV
    out_csv = out_dir / f"diachronic_neighbors_{args.word}_ref_{args.ref}_top{args.topk}.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("word,decade,year,rank,neighbor,cosine\n")
        for r in rows:
            f.write(f"{r['word']},{r['decade']},{r['year']},{r['rank']},{r['neighbor']},{r['cosine']}\n")

    # plot
    out_png = out_dir / f"diachronic_associations_{args.word}_ref_{args.ref}_top{args.topk}.png"
    title = f'Diachronic associations for "{args.word}": neighbors & cosine similarity'
    plot_diachronic_associations(rows, out_png, title=title, min_cos=args.min_cos)

    print(f"Wrote:\n- {out_png}\n- {out_csv}")


if __name__ == "__main__":
    main()
