"""
Plot diachronic associations ("blog style") from existing per-decade CSVs.

Input: results/neighbors_<decade>.csv files produced by visualize_drift.py --neighbors
Each file expected header:
    word,decade,rank,neighbor,cosine

Output: a single PNG (default into plots/) showing:
    x = year (from decade)
    y = cosine similarity to the target word
    label = neighbor word
"""

# This script differs from diachronic_associations.py:
# - It does NOT load embeddings
# - It does NOT recompute neighbors
# - It only reads already-produced per-decade results CSVs
# This makes it useful when results were computed once and plots are regenerated many times.

import argparse
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

from src.config import DECADES, PLOTS_DIR


# Regex used to parse decade labels like "1900s" -> "1900"
DECADE_RE = re.compile(r"^(\d{4})s$")


def decade_to_year(dec: str) -> int:
    """
    Convert decade label into start year integer.

    Parameters
    ----------
    dec : str
        Decade label (e.g., "1910s").

    Returns
    -------
    int
        Start year (e.g., 1910).
    """
    m = DECADE_RE.match(dec)
    if not m:
        raise ValueError(f"Invalid decade format: {dec!r} (expected '1900s', '1910s', ...)")
    return int(m.group(1))


def decade_slice(from_dec: str | None, to_dec: str | None) -> list[str]:
    """
    Return a contiguous slice of DECADES between from_dec and to_dec inclusive.

    If both bounds are None, return all decades.

    Parameters
    ----------
    from_dec, to_dec : str | None
        Decade bounds.

    Returns
    -------
    list[str]
        Decade labels in chronological order.
    """
    if from_dec is None and to_dec is None:
        return list(DECADES)
    if from_dec is None:
        from_dec = DECADES[0]
    if to_dec is None:
        to_dec = DECADES[-1]
    if from_dec not in DECADES:
        raise ValueError(f"--from_decade {from_dec!r} not in DECADES={DECADES}")
    if to_dec not in DECADES:
        raise ValueError(f"--to_decade {to_dec!r} not in DECADES={DECADES}")
    a = DECADES.index(from_dec)
    b = DECADES.index(to_dec)
    if a > b:
        raise ValueError("--from_decade must be <= --to_decade")
    return list(DECADES[a : b + 1])


def read_neighbors_csv(csv_path: Path) -> list[tuple[str, str, int, str, float]]:
    """
    Read a neighbors CSV produced by the visualization pipeline.

    Expected header:
        word,decade,rank,neighbor,cosine

    Returns
    -------
    list[tuple[str, str, int, str, float]]
        Rows in the form: (word, decade, rank, neighbor, cosine)
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        expected = ["word", "decade", "rank", "neighbor", "cosine"]
        if header[:5] != expected:
            raise ValueError(f"Unexpected header in {csv_path}: {header} (expected {expected})")

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            word = parts[0]
            decade = parts[1]
            rank = int(parts[2])
            neighbor = parts[3]
            cosine = float(parts[4])
            rows.append((word, decade, rank, neighbor, cosine))
    return rows


def plot_diachronic_associations(
    points: list[tuple[int, float, str]],
    out_png: Path,
    title: str,
    y_label: str,
    min_cos: float | None = None,
):
    """
    Plot diachronic association points with label overlap avoidance.

    points is a list of:
        (year, cosine, label)

    The data points remain fixed.
    Only the text labels are moved via adjustText for readability.

    Parameters
    ----------
    points : list[tuple[int, float, str]]
        (year, cosine, neighbor_label)
    out_png : Path
        Output path for the PNG.
    title : str
        Plot title.
    y_label : str
        Y-axis label.
    min_cos : float | None
        Optional y-axis lower bound.
    """
    years = np.array([p[0] for p in points], dtype=float)
    cos = np.array([p[1] for p in points], dtype=float)
    labels = [p[2] for p in points]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.scatter(years, cos, s=30, zorder=2)

    # Create text labels at the original point positions.
    # adjustText will move ONLY these text objects.
    texts = [ax.text(x, y, lab, fontsize=8, zorder=3) for x, y, lab in zip(years, cos, labels)]

    # ---- Expand axes automatically to create "room" for labels ----
    # X: years are discrete; add a constant margin for label placement.
    x_min, x_max = years.min(), years.max()
    ax.set_xlim(x_min - 5, x_max + 5)

    # Y: add proportional padding to avoid a cramped vertical scale.
    # This is robust even when values are close together.
    y_min, y_max = cos.min(), cos.max()
    dy = y_max - y_min
    if dy <= 1e-9:
        dy = 0.05  # fallback if all cosines are equal or nearly equal
    ax.set_ylim(y_min - 0.35 * dy, y_max + 0.35 * dy)

    # Optional lower bound (useful for consistent scaling across figures)
    if min_cos is not None:
        ax.set_ylim(bottom=min_cos)

    # ---- Relax label overlaps (texts only; points untouched) ----
    adjust_text(
        texts,
        ax=ax,
        expand_points=(1.2, 1.4),
        expand_text=(1.2, 1.4),
        arrowprops=dict(arrowstyle="-", lw=0.4, color="gray"),
    )

    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(y_label)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    # CLI for generating the plot from existing per-decade neighbors CSV files
    ap = argparse.ArgumentParser(
        description="Plot diachronic associations from existing results/neighbors_<decade>.csv files."
    )
    ap.add_argument("--word", required=True, help="Target word (must exist in the CSVs).")
    ap.add_argument("--results_dir", default="results", help="Directory containing neighbors_<decade>.csv files.")
    ap.add_argument("--topk", type=int, default=6, help="Top-k neighbors per decade to plot.")
    ap.add_argument("--from_decade", default=None, help="e.g. 1910s")
    ap.add_argument("--to_decade", default=None, help="e.g. 1960s")
    ap.add_argument("--min_cos", type=float, default=None, help="Optional y-axis lower bound.")
    ap.add_argument("--out_png", default=None, help="Optional explicit output path for PNG.")
    args = ap.parse_args()

    # Select the time window (or all decades)
    decs = decade_slice(args.from_decade, args.to_decade)

    # Validate results directory
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    # Collect (year, cosine, neighbor) points across all selected decades
    points: list[tuple[int, float, str]] = []

    for dec in decs:
        # Each decade is expected to have a file results/neighbors_<decade>.csv
        csv_path = results_dir / f"neighbors_{dec}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing {csv_path}. Generate it first with:\n"
                f"  python -m src.visualization.visualize_drift --words {args.word} --neighbors ..."
            )

        rows = read_neighbors_csv(csv_path)

        # Keep only the requested word and top-k ranks
        rows = [r for r in rows if r[0] == args.word and r[2] <= args.topk]

        # Convert decade label to numeric year for plotting
        year = decade_to_year(dec)

        # Add one point per neighbor (rank 1..topk) for this decade
        for (word, decade, rank, neighbor, cosine) in rows:
            # Skip non-finite values defensively
            if not np.isfinite(cosine):
                continue
            points.append((year, cosine, neighbor))

    # If no points were collected, plotting would be meaningless; raise an explicit error
    if len(points) == 0:
        raise ValueError(
            f"No points found for word={args.word!r} in the selected decades.\n"
            f"Check:\n"
            f"- that results/neighbors_<decade>.csv exist\n"
            f"- that they contain rows for this word\n"
            f"- that --topk is not too small\n"
        )

    # Output path:
    # - default: use configured PLOTS_DIR and an informative name
    # - or use user-provided --out_png
    if args.out_png is None:
        out_dir = Path(PLOTS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"diachronic_associations_{args.word}_{decs[0]}_{decs[-1]}_top{args.topk}.png"
    else:
        out_png = Path(args.out_png)

    title = f'Diachronic associations for "{args.word}": neighbors & cosine similarity'
    y_label = f"Cosine similarity to '{args.word}'"

    # Produce the plot
    plot_diachronic_associations(
        points=points,
        out_png=out_png,
        title=title,
        y_label=y_label,
        min_cos=args.min_cos,
    )

    # Print output path for convenience in terminal workflows
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
