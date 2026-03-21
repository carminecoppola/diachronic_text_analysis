"""
Plot semantic stability/dispersion over time from results/neighbors_<decade>.csv.

Expected per-decade CSV header (produced by visualize_drift.py --neighbors):
    word,decade,rank,neighbor,cosine

Examples:
  # Stability (mean cosine) for dog over all decades
  python -m src.visualization.plot_stability_dispersion --words dog --metric mean --topk 10

  # Dispersion (std) for dog between 1910s and 1960s
  python -m src.visualization.plot_stability_dispersion --words dog --metric std --topk 10 --from_decade 1910s --to_decade 1960s

  # Compare two runs/models saved in different results dirs
  python -m src.visualization.plot_stability_dispersion --words dog --metric mean --topk 10 \
    --results_dirs results_baseline results_weighted --labels baseline weighted
"""

import argparse
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt

from src.config import DECADES, PLOTS_DIR


DECADE_RE = re.compile(r"^(\d{4})s$")


def decade_to_year(dec: str) -> int:
    m = DECADE_RE.match(dec)
    if not m:
        raise ValueError(f"Invalid decade format: {dec!r}")
    return int(m.group(1))


def decade_slice(from_dec: str | None, to_dec: str | None) -> list[str]:
    if from_dec is None and to_dec is None:
        return list(DECADES)
    if from_dec is None:
        from_dec = DECADES[0]
    if to_dec is None:
        to_dec = DECADES[-1]
    if from_dec not in DECADES or to_dec not in DECADES:
        raise ValueError(f"from/to must be in DECADES={DECADES}")
    a = DECADES.index(from_dec)
    b = DECADES.index(to_dec)
    if a > b:
        raise ValueError("--from_decade must be <= --to_decade")
    return list(DECADES[a : b + 1])


def read_neighbors_csv(csv_path: Path):
    # word,decade,rank,neighbor,cosine
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


def compute_metric(values: np.ndarray, metric: str) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")

    if metric == "mean":
        return float(values.mean())
    if metric == "std":
        return float(values.std(ddof=0))
    if metric == "cv":
        m = values.mean()
        s = values.std(ddof=0)
        return float(s / m) if m != 0 else float("nan")
    if metric == "gap":
        return float(values.max() - values.min())
    if metric == "mean_top3":
        k = min(3, values.size)
        # values are not guaranteed sorted; sort descending first
        vv = np.sort(values)[::-1]
        return float(vv[:k].mean())

    raise ValueError(f"Unknown metric: {metric}")


def main():
    ap = argparse.ArgumentParser(description="Plot semantic stability/dispersion over decades from neighbors CSVs.")
    ap.add_argument("--words", nargs="+", required=True, help="Target words (must exist in neighbors CSVs).")
    ap.add_argument("--metric", default="mean",
                    choices=["mean", "std", "cv", "gap", "mean_top3"],
                    help="Aggregation metric over top-k cosine similarities.")
    ap.add_argument("--topk", type=int, default=10, help="Use ranks 1..topk from each decade CSV.")
    ap.add_argument("--from_decade", default=None)
    ap.add_argument("--to_decade", default=None)

    # allow comparing multiple experiments/runs (different results dirs)
    ap.add_argument("--results_dirs", nargs="+", default=["results"],
                    help="One or more directories containing neighbors_<decade>.csv")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Labels for each results dir (same length as results_dirs).")

    ap.add_argument("--out_png", default=None, help="Optional output path for PNG.")
    ap.add_argument("--title", default=None, help="Optional title override.")
    args = ap.parse_args()

    decs = decade_slice(args.from_decade, args.to_decade)
    years = [decade_to_year(d) for d in decs]

    results_dirs = [Path(d) for d in args.results_dirs]
    for d in results_dirs:
        if not d.exists():
            raise FileNotFoundError(f"results_dir not found: {d}")

    if args.labels is None:
        labels = [d.name for d in results_dirs]
    else:
        if len(args.labels) != len(results_dirs):
            raise ValueError("--labels must have same length as --results_dirs")
        labels = args.labels

    # Data structure:
    # series[(run_label, word)] = list of metric values aligned with decs
    series = {}

    for run_dir, run_label in zip(results_dirs, labels):
        # Preload per-decade rows for speed
        rows_by_decade = {}
        for dec in decs:
            csv_path = run_dir / f"neighbors_{dec}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing {csv_path}. Generate it first with visualize_drift.py --neighbors")
            rows_by_decade[dec] = read_neighbors_csv(csv_path)

        for w in args.words:
            y_vals = []
            for dec in decs:
                rows = rows_by_decade[dec]
                vals = [r[4] for r in rows if r[0] == w and r[2] <= args.topk]
                vals = np.array(vals, dtype=float)
                y_vals.append(compute_metric(vals, args.metric))
            series[(run_label, w)] = y_vals

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))

    # one line per (run, word)
    for (run_label, w), y_vals in series.items():
        ax.plot(years, y_vals, marker="o", linewidth=1.6, label=f"{w} ({run_label})")

    ax.set_xlabel("Year")
    ax.set_ylabel(f"{args.metric} over top-{args.topk} cosine")
    if args.title:
        ax.set_title(args.title)
    else:
        ax.set_title(f"Semantic {('stability' if args.metric in ['mean','mean_top3'] else 'dispersion')} over time ({args.metric})")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    if args.out_png is None:
        out_dir = Path(PLOTS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        win = f"{decs[0]}_{decs[-1]}"
        out_png = out_dir / f"stability_dispersion_{args.metric}_top{args.topk}_{win}_{'_'.join(args.words)}.png"
    else:
        out_png = Path(args.out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
