import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.visualization.plotting import plot_trajectory
from src.config import DECADES, MODELS_DIR, VOCAB_FILE, ALIGNMENT_TRANSFORMS_DIR, PLOTS_DIR
from src.visualization.drift_io import load_vocab, load_embedding_matrix
from src.visualization.alignment_utils import align_embeddings_to_ref
from src.visualization.neighbors import write_neighbors_csv


ALIGNMENT_DIR = Path(ALIGNMENT_TRANSFORMS_DIR)
PLOTS_DIR_DEFAULT = PLOTS_DIR
RESULTS_DIR_DEFAULT = "results"


def _pick_background_indices(vocab_size: int, exclude: set[int], n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    candidates = np.array([i for i in range(vocab_size) if i not in exclude], dtype=int)
    if len(candidates) == 0:
        return np.array([], dtype=int)
    n = min(n, len(candidates))
    return rng.choice(candidates, size=n, replace=False)


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
    ap.add_argument("--perplexity", type=int, default=None)

    # NEW: background points for stable PCA/t-SNE
    ap.add_argument("--background_n", type=int, default=1500,
                    help="Number of background vocab vectors (from ref decade) used to fit PCA/t-SNE. "
                         "0 disables background (not recommended).")
    ap.add_argument("--seed", type=int, default=0)

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

    # --- Load and align embeddings for each decade
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

    # --- Build trajectory matrix: (word, decade) points
    points_all = []
    for decade in DECADES:
        E_aligned = aligned_by_decade[decade]
        for w in args.words:
            points_all.append(E_aligned[vocab[w]])
    X_traj = np.vstack(points_all)  # shape: (len(words)*len(decades), dim)

    # --- Background (from ref decade space) to stabilize PCA/t-SNE
    E_ref = aligned_by_decade[args.ref]
    exclude = {vocab[w] for w in args.words}
    bg_idx = _pick_background_indices(E_ref.shape[0], exclude=exclude, n=max(0, args.background_n), seed=args.seed)
    X_bg = E_ref[bg_idx] if len(bg_idx) else None

    if args.pca:
        if X_bg is not None and len(X_bg) > 0:
            pca = PCA(n_components=2, random_state=args.seed)
            pca.fit(X_bg)
            X2_traj = pca.transform(X_traj)
            ev = (float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1]))
        else:
            # fallback (not recommended)
            pca = PCA(n_components=2, random_state=args.seed)
            X2_traj = pca.fit_transform(X_traj)
            ev = (float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1]))

        out = plots_dir / (
            f"pca_{'_'.join(args.words)}_ref_{args.ref}_win_{args.from_decade or 'ALL'}_{args.to_decade or 'ALL'}.png"
        )
        plot_trajectory(
            X2=X2_traj,
            words=args.words,
            title=f"PCA aligned to {args.ref} | var: PC1={ev[0]*100:.1f}%, PC2={ev[1]*100:.1f}%",
            out_path=out,
            connect=args.connect,
            xlabel="PC1",
            ylabel="PC2",
        )


    if args.tsne:
        # For t-SNE we embed (background + trajectory) but plot only trajectory
        if X_bg is not None and len(X_bg) > 0:
            X_for_tsne = np.vstack([X_bg, X_traj])
            offset = len(X_bg)
        else:
            X_for_tsne = X_traj
            offset = 0

        n = X_for_tsne.shape[0]
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
            random_state=args.seed,
            perplexity=perp,
        )
        X2_all = tsne.fit_transform(X_for_tsne)
        X2_traj = X2_all[offset:, :]

        out = plots_dir / f"tsne_{'_'.join(args.words)}_ref_{args.ref}_perp_{perp}.png"
        plot_trajectory(
            X2=X2_traj,
            words=args.words,
            title=f"t-SNE aligned to {args.ref} (perplexity={perp})",
            out_path=out,
            connect=args.connect,
            xlabel="dim1",
            ylabel="dim2",
        )


if __name__ == "__main__":
    main()
