# This script computes and plots diachronic associations for a single target word.
# It operates directly on embeddings (not on results CSVs):
# - Loads and aligns embeddings for each decade into a reference decade space
# - Computes top-k nearest neighbors by cosine similarity at each decade
# - Writes an aggregated CSV (word,decade,year,rank,neighbor,cosine)
# - Produces a simple scatter plot (year vs cosine) annotated with neighbor strings

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.config import DECADES, MODELS_DIR, VOCAB_FILE, ALIGNMENT_TRANSFORMS_DIR
from src.visualization.drift_io import load_vocab, load_embedding_matrix
from src.visualization.alignment_utils import align_embeddings_to_ref
from src.visualization.neighbors import cosine_topk


# Regex used to convert decade identifiers (e.g., "1910s") to their numeric year (1910)
DECADE_RE = re.compile(r"^(\d{4})s$")


def decade_to_year(dec: str) -> int:
    """
    Convert a decade string like "1910s" into an integer year 1910.

    Parameters
    ----------
    dec : str
        Decade label.

    Returns
    -------
    int
        Start year of the decade.
    """
    m = DECADE_RE.match(dec)
    if not m:
        raise ValueError(f"Invalid decade format: {dec!r}")
    return int(m.group(1))


def compute_neighbors_over_time(word: str, vocab: dict, inv_vocab: dict, aligned_by_decade: dict[str, np.ndarray], topk: int):
    """
    For each decade, compute the top-k cosine neighbors of a target word.

    The embeddings for each decade are assumed to have already been aligned
    into the reference space.

    Parameters
    ----------
    word : str
        Target word for which neighbors are computed.
    vocab : dict
        word -> index mapping.
    inv_vocab : dict
        index -> word mapping.
    aligned_by_decade : dict[str, np.ndarray]
        decade -> aligned embedding matrix.
    topk : int
        Number of neighbors per decade.

    Returns
    -------
    list[dict]
        A list of rows, where each row includes:
        - word
        - decade
        - year
        - rank
        - neighbor
        - cosine
    """
    if word not in vocab:
        raise ValueError(f"Word {word!r} not in vocab.")

    rows = []
    widx = vocab[word]

    # Iterate through decades in configured chronological order
    for dec in DECADES:
        E = aligned_by_decade[dec]

        # Compute top-k cosine neighbors excluding the target word itself
        top_idx, top_sims = cosine_topk(E[widx], E, k=topk, exclude_index=widx)

        # Convert indices to neighbor strings and store row entries
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
    """
    Plot diachronic associations as a scatter plot:
    - x-axis: year (derived from decade)
    - y-axis: cosine similarity between target word and its neighbor
    - each point is annotated with the neighbor string

    Parameters
    ----------
    rows : list[dict]
        Output rows produced by compute_neighbors_over_time.
    out_png : Path
        Output file path for the image.
    title : str
        Plot title.
    min_cos : float | None
        If provided, set a fixed lower bound on the y-axis.
    """
    # Extract arrays from rows for plotting
    years = np.array([r["year"] for r in rows], dtype=float)
    cos = np.array([r["cosine"] for r in rows], dtype=float)
    labels = [r["neighbor"] for r in rows]

    plt.figure(figsize=(12, 4))
    plt.scatter(years, cos, s=28)

    # Annotate each point with the neighbor label.
    # This is a simple annotation approach and may overlap if many points cluster.
    for x, y, lab in zip(years, cos, labels):
        plt.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)

    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Cosine similarity to target")

    # Optional fixed lower bound for better visual consistency across words
    if min_cos is not None:
        plt.ylim(bottom=min_cos)

    plt.tight_layout()

    # Ensure output directory exists
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Save plot
    plt.savefig(out_png, dpi=220)
    plt.close()


def main():
    # CLI for generating the plot and the corresponding CSV
    ap = argparse.ArgumentParser(description="Diachronic associations plot: neighbors & cosine over decades.")
    ap.add_argument("--word", required=True, help="Target word (must be in vocab).")
    ap.add_argument("--ref", default="2010s", help="Reference decade for alignment (any decade supported).")
    ap.add_argument("--topk", type=int, default=6, help="Top-k neighbors per decade.")
    ap.add_argument("--out_dir", default="drift_plots", help="Output directory.")
    ap.add_argument("--min_cos", type=float, default=None, help="Optional y-axis lower bound.")
    args = ap.parse_args()

    # Prepare output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare alignment directory
    alignment_dir = Path(ALIGNMENT_TRANSFORMS_DIR)

    # Load vocab and inverse vocab
    vocab = load_vocab(VOCAB_FILE)
    inv_vocab = {idx: w for w, idx in vocab.items()}

    # Validate reference decade
    if args.ref not in DECADES:
        raise ValueError(f"ref decade {args.ref} not in DECADES={DECADES}")

    # Align each decade embedding matrix into the reference space
    aligned_by_decade = {}
    for dec in DECADES:
        E_dec = load_embedding_matrix(MODELS_DIR, dec)
        E_aligned = align_embeddings_to_ref(alignment_dir, E_dec, dec, args.ref)
        aligned_by_decade[dec] = E_aligned

    # Compute neighbors and cosine similarities for all decades
    rows = compute_neighbors_over_time(args.word, vocab, inv_vocab, aligned_by_decade, topk=args.topk)

    # Write aggregated CSV for reproducibility and later analysis
    out_csv = out_dir / f"diachronic_neighbors_{args.word}_ref_{args.ref}_top{args.topk}.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("word,decade,year,rank,neighbor,cosine\n")
        for r in rows:
            f.write(f"{r['word']},{r['decade']},{r['year']},{r['rank']},{r['neighbor']},{r['cosine']}\n")

    # Plot output
    out_png = out_dir / f"diachronic_associations_{args.word}_ref_{args.ref}_top{args.topk}.png"
    title = f'Diachronic associations for "{args.word}": neighbors & cosine similarity'
    plot_diachronic_associations(rows, out_png, title=title, min_cos=args.min_cos)

    # Print output locations to the console for convenience
    print(f"Wrote:\n- {out_png}\n- {out_csv}")


if __name__ == "__main__":
    main()
