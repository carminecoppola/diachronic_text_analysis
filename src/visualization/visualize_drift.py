# This script is the main entry point for diachronic visualization:
# - Loads embeddings for each decade
# - Aligns them into a reference decade space
# - Optionally computes nearest neighbors and writes per-decade CSV outputs
# - Projects trajectories into 2D using PCA and/or t-SNE (with optional background stabilization)
# - Produces plots in the configured plots directory

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# plot_trajectory is a generic plotting routine used for both PCA and t-SNE outputs
from src.visualization.plotting import plot_trajectory

# Global configuration:
# - DECADES: ordered list of decades considered in the project
# - MODELS_DIR: directory containing embedding files emb_<decade>.pt
# - VOCAB_FILE: path to the JSON vocabulary (word -> index)
# - ALIGNMENT_TRANSFORMS_DIR: directory containing pairwise alignment transforms
# - PLOTS_DIR: default output directory for plots
from src.config import DECADES, MODELS_DIR, VOCAB_FILE, ALIGNMENT_TRANSFORMS_DIR, PLOTS_DIR

# I/O helpers to load vocab and embedding tensors
from src.visualization.drift_io import load_vocab, load_embedding_matrix

# Alignment helper that maps embeddings from an arbitrary decade into the reference decade space
from src.visualization.alignment_utils import align_embeddings_to_ref

# Neighbors CSV writer used when --neighbors is enabled
from src.visualization.neighbors import write_neighbors_csv


# Resolve the alignment directory as a Path object for filesystem operations
ALIGNMENT_DIR = Path(ALIGNMENT_TRANSFORMS_DIR)

# Default directories used when CLI args are not explicitly provided
PLOTS_DIR_DEFAULT = PLOTS_DIR
RESULTS_DIR_DEFAULT = "results"


def _pick_background_indices(vocab_size: int, exclude: set[int], n: int, seed: int = 0) -> np.ndarray:
    """
    Select a random subset of vocabulary indices to be used as "background" points.

    Background points are used to stabilize PCA/t-SNE:
    - PCA is fit on the background (in the reference decade space)
    - t-SNE is computed on (background + trajectory) so the embedding is not based only on few points

    Parameters
    ----------
    vocab_size : int
        Total number of tokens in the vocabulary.
    exclude : set[int]
        Indices to exclude (typically the indices of the target words).
    n : int
        Number of indices to sample.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Sampled indices (unique, without replacement).
    """
    # Use numpy's Generator API for reproducible sampling
    rng = np.random.default_rng(seed)

    # Build candidate indices excluding the target word indices (so they are not sampled as background)
    candidates = np.array([i for i in range(vocab_size) if i not in exclude], dtype=int)

    # If all indices are excluded, there is no background available
    if len(candidates) == 0:
        return np.array([], dtype=int)

    # Ensure we do not sample more indices than available
    n = min(n, len(candidates))

    # Sample without replacement to avoid duplicates
    return rng.choice(candidates, size=n, replace=False)


def main():
    # Define the command-line interface
    ap = argparse.ArgumentParser()

    # Target words to track through time (must exist in vocab)
    ap.add_argument("--words", nargs="+", required=True)

    # Reference decade: all decades will be aligned into this space
    ap.add_argument("--ref", default="2010s")

    # Flags controlling which projections are computed
    ap.add_argument("--pca", action="store_true")
    ap.add_argument("--tsne", action="store_true")

    # If enabled, the script writes per-decade neighbors CSVs into results_dir
    ap.add_argument("--neighbors", action="store_true")
    ap.add_argument("--topk", type=int, default=10)

    # Optional temporal window arguments (currently used for naming the PCA output file;
    # trajectory points are still produced for all decades unless further filtering is added)
    ap.add_argument("--from_decade", default=None)
    ap.add_argument("--to_decade", default=None)

    # If enabled, connect points across decades with a line
    ap.add_argument("--connect", action="store_true")

    # Optional perplexity override for t-SNE
    ap.add_argument("--perplexity", type=int, default=None)

    # Background size to stabilize PCA/t-SNE:
    # - PCA: fit on background from the reference decade space
    # - t-SNE: embed background + trajectory, then plot only trajectory points
    ap.add_argument(
        "--background_n",
        type=int,
        default=1500,
        help="Number of background vocab vectors (from ref decade) used to fit PCA/t-SNE. "
             "0 disables background (not recommended)."
    )

    # Random seed used for background sampling and PCA/t-SNE random state
    ap.add_argument("--seed", type=int, default=0)

    # Output directories; defaults come from project config
    ap.add_argument("--plots_dir", default=str(PLOTS_DIR_DEFAULT))
    ap.add_argument("--results_dir", default=str(RESULTS_DIR_DEFAULT))
    args = ap.parse_args()

    # Create output directories if missing
    plots_dir = Path(args.plots_dir); plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir); results_dir.mkdir(parents=True, exist_ok=True)

    # Validate reference decade
    if args.ref not in DECADES:
        raise ValueError(f"ref decade {args.ref} not in DECADES={DECADES}")

    # Load vocabulary (word -> index)
    vocab = load_vocab(VOCAB_FILE)

    # Build inverse vocabulary (index -> word) used for neighbor name lookup
    inv_vocab = {idx: w for w, idx in vocab.items()}

    # Ensure all requested target words are present in the vocabulary
    missing = [w for w in args.words if w not in vocab]
    if missing:
        raise ValueError(f"Words not in vocab: {missing}")

    # --- Load and align embeddings for each decade
    # aligned_by_decade maps:
    #   decade -> aligned embedding matrix (numpy or numpy-like)
    # The alignment maps every decade into the reference decade space.
    aligned_by_decade = {}
    for decade in DECADES:
        # Load decade embedding tensor from MODELS_DIR (on CPU)
        E_dec = load_embedding_matrix(MODELS_DIR, decade)

        # Align the embedding matrix into the reference space
        E_aligned = align_embeddings_to_ref(ALIGNMENT_DIR, E_dec, decade, args.ref)

        # Cache it for later trajectory/projection computations
        aligned_by_decade[decade] = E_aligned

        # Optionally compute and write nearest neighbors for the requested words
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
    # The order is:
    #   for decade in DECADES:
    #     for word in args.words:
    #       append vector for (word, decade)
    # This defines X_traj with shape:
    #   (len(words) * len(decades), embedding_dim)
    points_all = []
    for decade in DECADES:
        E_aligned = aligned_by_decade[decade]
        for w in args.words:
            points_all.append(E_aligned[vocab[w]])

    # Stack into a single matrix suitable for PCA/t-SNE transforms
    X_traj = np.vstack(points_all)  # shape: (len(words)*len(decades), dim)

    # --- Background (from ref decade space) to stabilize PCA/t-SNE
    # Background is sampled from the reference decade embedding matrix in aligned space.
    # This prevents PCA/t-SNE from being dominated by the few trajectory points only.
    E_ref = aligned_by_decade[args.ref]

    # Exclude the target words from background sampling to avoid trivial inclusion
    exclude = {vocab[w] for w in args.words}

    # Sample background indices
    bg_idx = _pick_background_indices(
        E_ref.shape[0],
        exclude=exclude,
        n=max(0, args.background_n),
        seed=args.seed
    )

    # Gather background vectors if any were sampled
    X_bg = E_ref[bg_idx] if len(bg_idx) else None

    # --- PCA projection (optional)
    if args.pca:
        # Recommended: fit PCA on background (stable basis) then transform trajectory points
        if X_bg is not None and len(X_bg) > 0:
            pca = PCA(n_components=2, random_state=args.seed)
            pca.fit(X_bg)
            X2_traj = pca.transform(X_traj)
            ev = (float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1]))
        else:
            # Fallback: fit PCA on trajectory itself (less stable, depends strongly on selected words)
            pca = PCA(n_components=2, random_state=args.seed)
            X2_traj = pca.fit_transform(X_traj)
            ev = (float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1]))

        # Output file naming includes reference decade and optional window identifiers
        out = plots_dir / (
            f"pca_{'_'.join(args.words)}_ref_{args.ref}_win_{args.from_decade or 'ALL'}_{args.to_decade or 'ALL'}.png"
        )

        # Plot trajectory (points are in the same order as X_traj)
        plot_trajectory(
            X2=X2_traj,
            words=args.words,
            title=f"PCA aligned to {args.ref} | var: PC1={ev[0]*100:.1f}%, PC2={ev[1]*100:.1f}%",
            out_path=out,
            connect=args.connect,
            xlabel="PC1",
            ylabel="PC2",
        )

    # --- t-SNE projection (optional)
    if args.tsne:
        # For t-SNE, the embedding is computed on (background + trajectory) to reduce instability.
        # We then plot only the trajectory part of the resulting 2D embedding.
        if X_bg is not None and len(X_bg) > 0:
            X_for_tsne = np.vstack([X_bg, X_traj])
            offset = len(X_bg)  # number of background points at the beginning
        else:
            X_for_tsne = X_traj
            offset = 0

        # Total number of points embedded by t-SNE
        n = X_for_tsne.shape[0]

        # Choose a perplexity automatically if not provided.
        # The rule of thumb is: perplexity < n_samples, and typically between 5 and 50.
        if args.perplexity is None:
            perp = min(30, max(2, (n - 1) // 3))
        else:
            perp = args.perplexity

        # Hard constraint in scikit-learn: perplexity must be < n_samples
        if perp >= n:
            raise ValueError(f"perplexity ({perp}) must be < n_samples ({n})")

        # Configure TSNE. init="random" plus a fixed random_state ensures reproducibility.
        tsne = TSNE(
            n_components=2,
            init="random",
            learning_rate="auto",
            random_state=args.seed,
            perplexity=perp,
        )

        # Compute 2D embedding for all points (background + trajectory, if background enabled)
        X2_all = tsne.fit_transform(X_for_tsne)

        # Extract only trajectory points (skip the background prefix)
        X2_traj = X2_all[offset:, :]

        # Output file naming includes reference decade and perplexity
        out = plots_dir / f"tsne_{'_'.join(args.words)}_ref_{args.ref}_perp_{perp}.png"

        # Plot trajectory using the same routine as PCA
        plot_trajectory(
            X2=X2_traj,
            words=args.words,
            title=f"t-SNE aligned to {args.ref} (perplexity={perp})",
            out_path=out,
            connect=args.connect,
            xlabel="dim1",
            ylabel="dim2",
        )


# Standard Python module entry point
if __name__ == "__main__":
    main()
