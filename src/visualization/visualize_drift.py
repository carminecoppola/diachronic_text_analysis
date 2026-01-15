import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.config import DECADES
from src.visualization.drift_io import load_vocab, load_embedding_matrix
from src.visualization.alignment_utils import align_embeddings_to_ref
from src.visualization.neighbors import write_neighbors_csv
from src.visualization.plotting import plot_pca_window, plot_tsne


from src.config import MODELS_DIR, VOCAB_FILE, ALIGNMENT_TRANSFORMS_DIR, PLOTS_DIR

from pathlib import Path
ALIGNMENT_DIR = Path(ALIGNMENT_TRANSFORMS_DIR)
PLOTS_DIR_DEFAULT = PLOTS_DIR
RESULTS_DIR_DEFAULT = "results"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--words", nargs="+", required=True)
    ap.add_argument("--ref", default="2010s")

    ap.add_argument("--pca", action="store_true")
    ap.add_argument("--tsne", action="store_true")

    ap.add_argument("--neighbors", action="store_true")
    ap.add_argument("--topk", type=int, default=10)

    ap.add_argument("--from_decade", default=None)
    ap.add_argument("--to_decade", default=None)

    ap.add_argument("--connect", action="store_true")
    ap.add_argument("--label_dx", type=float, default=0.0)  # 0 = auto
    ap.add_argument("--label_dy", type=float, default=0.0)  # 0 = auto
    ap.add_argument("--perplexity", type=int, default=None)

    ap.add_argument("--plots_dir", default=str(PLOTS_DIR_DEFAULT))
    ap.add_argument("--results_dir", default=str(RESULTS_DIR_DEFAULT))
    args = ap.parse_args()

    plots_dir = Path(args.plots_dir); plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir); results_dir.mkdir(parents=True, exist_ok=True)

    if args.ref not in DECADES:
        raise ValueError(f"ref decade {args.ref} not in DECADES={DECADES}")

    vocab = load_vocab(VOCAB_FILE)
    inv_vocab = {idx: w for w, idx in vocab.items()}

    missing = [w for w in args.words if w not in vocab]
    if missing:
        raise ValueError(f"Words not in vocab: {missing}")

    aligned_by_decade = {}

    for decade in DECADES:
        E_dec = load_embedding_matrix(MODELS_DIR, decade)
        E_aligned = align_embeddings_to_ref(ALIGNMENT_DIR, E_dec, decade, args.ref)
        aligned_by_decade[decade] = E_aligned

        if args.neighbors:
            out_csv = results_dir / f"neighbors_{decade}.csv"
            write_neighbors_csv(
                out_csv=out_csv,
                words=args.words,
                decade=decade,
                vocab=vocab,
                inv_vocab=inv_vocab,
                E_aligned=E_aligned,
                topk=args.topk,
            )

    # points matrix: per decade, per word (stesso ordine sempre)
    points_all = []
    for decade in DECADES:
        E_aligned = aligned_by_decade[decade]
        for w in args.words:
            points_all.append(E_aligned[vocab[w]])
    X_all = np.vstack(points_all)

    if args.pca:
        pca = PCA(n_components=2, random_state=0)
        X2_all = pca.fit_transform(X_all)

        out = plots_dir / (
            f"pca_{'_'.join(args.words)}_ref_{args.ref}_win_{args.from_decade or 'ALL'}_{args.to_decade or 'ALL'}.png"
        )
        plot_pca_window(
            X2_all=X2_all,
            words=args.words,
            ref_decade=args.ref,
            out_path=out,
            connect=args.connect,
            label_dx=args.label_dx,
            label_dy=args.label_dy,
            from_dec=args.from_decade,
            to_dec=args.to_decade,
        )

    if args.tsne:
        n = X_all.shape[0]
        if args.perplexity is None:
            perp = min(30, max(2, (n - 1) // 3))
        else:
            perp = args.perplexity
        if perp >= n:
            raise ValueError(f"perplexity ({perp}) must be < n_samples ({n})")

        tsne = TSNE(
            n_components=2,
            init="random",
            learning_rate="auto",
            random_state=0,
            perplexity=perp,
        )
        X2 = tsne.fit_transform(X_all)

        out = plots_dir / f"tsne_{'_'.join(args.words)}_ref_{args.ref}_perp_{perp}.png"
        plot_tsne(
            X2=X2,
            words=args.words,
            ref_decade=args.ref,
            out_path=out,
            connect=args.connect,
            label_dx=args.label_dx,
            label_dy=args.label_dy,
            perplexity=perp,
        )


if __name__ == "__main__":
    main()
