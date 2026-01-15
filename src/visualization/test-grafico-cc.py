import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DECADE_RE = re.compile(r"^(\d{4})s$")


def parse_decade(decade_str: str) -> int:
    """
    Convert '1940s' -> 1940 for sorting.
    """
    m = DECADE_RE.match(str(decade_str).strip())
    if not m:
        raise ValueError(f"Invalid decade format: {decade_str!r} (expected like '1940s')")
    return int(m.group(1))


def load_neighbors_csvs(inputs: list[Path]) -> pd.DataFrame:
    """
    Load and concat multiple CSVs. Validates required columns.
    """
    frames = []
    for p in inputs:
        df = pd.read_csv(p)
        frames.append(df)

    if not frames:
        raise ValueError("No CSVs loaded.")

    df = pd.concat(frames, ignore_index=True)

    required = {"word", "decade", "rank", "neighbor", "cosine"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV(s): {sorted(missing)}")

    # Clean types
    df["word"] = df["word"].astype(str).str.strip()
    df["decade"] = df["decade"].astype(str).str.strip()
    df["neighbor"] = df["neighbor"].astype(str).str.strip()

    # numeric conversions
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["cosine"] = pd.to_numeric(df["cosine"], errors="coerce")

    # Drop critical NaNs
    df = df.dropna(subset=["word", "decade", "rank", "cosine"])

    # Validate rank in 1..6 (softly; keep but warn-like behavior)
    # If you prefer strict, uncomment the next lines.
    # bad_rank = ~df["rank"].between(1, 6)
    # if bad_rank.any():
    #     raise ValueError(f"Found ranks outside 1..6 in rows: {df[bad_rank].head()}")

    # decade sort key
    df["decade_start"] = df["decade"].apply(parse_decade)

    return df


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (word, decade) compute mean/min/max/std of cosine.
    """
    agg = (
        df.groupby(["word", "decade", "decade_start"], as_index=False)
        .agg(
            mean_cosine=("cosine", "mean"),
            min_cosine=("cosine", "min"),
            max_cosine=("cosine", "max"),
            std_cosine=("cosine", "std"),
            n=("cosine", "count"),
        )
        .sort_values(["word", "decade_start"])
        .reset_index(drop=True)
    )

    # std can be NaN if n==1; fill with 0 for cleaner plotting if desired
    agg["std_cosine"] = agg["std_cosine"].fillna(0.0)

    return agg


def plot_stability_curve(agg: pd.DataFrame, out_path: Path, words: list[str] | None = None):
    """
    Line plot: mean_cosine over decades, one line per word.
    """
    plt.figure()
    if words is None:
        words = sorted(agg["word"].unique())

    for w in words:
        sub = agg[agg["word"] == w].sort_values("decade_start")
        if sub.empty:
            continue
        plt.plot(sub["decade_start"], sub["mean_cosine"], marker="o", label=w)

    plt.title("Temporal Evolution of Semantic Similarity")
    plt.xlabel("Decade (start year)")
    plt.ylabel("Mean cosine (top neighbors)")
    if len(words) > 1:
        plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_dispersion(agg: pd.DataFrame, out_path: Path, words: list[str] | None = None):
    """
    Ribbon plot: band [min_cosine, max_cosine] + mean line, per word.
    If multiple words, creates one figure per word (more readable).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if words is None:
        words = sorted(agg["word"].unique())

    if len(words) == 1:
        w = words[0]
        sub = agg[agg["word"] == w].sort_values("decade_start")
        plt.figure()
        x = sub["decade_start"].to_numpy()
        y = sub["mean_cosine"].to_numpy()
        ylo = sub["min_cosine"].to_numpy()
        yhi = sub["max_cosine"].to_numpy()

        plt.plot(x, y, marker="o")
        plt.fill_between(x, ylo, yhi, alpha=0.2)

        plt.title("Semantic Dispersion of Nearest Neighbors Across Decades")
        plt.xlabel("Decade (start year)")
        plt.ylabel("Cosine similarity")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return

    # multiple words -> one plot per word
    stem = out_path.stem
    suffix = out_path.suffix
    for w in words:
        sub = agg[agg["word"] == w].sort_values("decade_start")
        if sub.empty:
            continue
        plt.figure()
        x = sub["decade_start"].to_numpy()
        y = sub["mean_cosine"].to_numpy()
        ylo = sub["min_cosine"].to_numpy()
        yhi = sub["max_cosine"].to_numpy()

        plt.plot(x, y, marker="o")
        plt.fill_between(x, ylo, yhi, alpha=0.2)

        plt.title(f"Semantic Dispersion — {w}")
        plt.xlabel("Decade (start year)")
        plt.ylabel("Cosine similarity")
        plt.tight_layout()

        per_word = out_path.parent / f"{stem}_{w}{suffix}"
        plt.savefig(per_word, dpi=200)
        plt.close()


def generate_brief_report(agg: pd.DataFrame, out_path: Path, words: list[str] | None = None):
    """
    Writes a short (5–8 sentences) report per word, based only on computed metrics.
    """
    if words is None:
        words = sorted(agg["word"].unique())

    lines = []
    for w in words:
        sub = agg[agg["word"] == w].sort_values("decade_start")
        if sub.empty:
            continue

        # Basic descriptors
        mean_min = float(sub["mean_cosine"].min())
        mean_max = float(sub["mean_cosine"].max())
        mean_range = mean_max - mean_min

        # Biggest step change between decades (drift_step proxy)
        y = sub["mean_cosine"].to_numpy()
        x = sub["decade_start"].to_numpy()
        steps = np.abs(np.diff(y))
        if len(steps) > 0:
            idx = int(np.argmax(steps))
            biggest_step = float(steps[idx])
            step_from = int(x[idx])
            step_to = int(x[idx + 1])
        else:
            biggest_step = 0.0
            step_from = int(x[0])
            step_to = int(x[0])

        # Dispersion: average band width and peak band width
        band = (sub["max_cosine"] - sub["min_cosine"]).to_numpy()
        avg_band = float(np.mean(band)) if len(band) else 0.0
        peak_band = float(np.max(band)) if len(band) else 0.0
        peak_dec = int(sub.iloc[int(np.argmax(band))]["decade_start"]) if len(band) else int(x[0])

        # Stability: identify most stable stretch (smallest step) if possible
        if len(steps) > 0:
            stable_idx = int(np.argmin(steps))
            stable_from = int(x[stable_idx])
            stable_to = int(x[stable_idx + 1])
            stable_step = float(steps[stable_idx])
        else:
            stable_from = int(x[0])
            stable_to = int(x[0])
            stable_step = 0.0

        # 5–8 sentences (per word)
        lines.append(f"Word: {w}")
        lines.append(
            f"Across decades, mean cosine ranges from {mean_min:.3f} to {mean_max:.3f} (range {mean_range:.3f})."
        )
        lines.append(
            f"The largest decade-to-decade change in mean cosine is {biggest_step:.3f}, between {step_from}s and {step_to}s."
        )
        lines.append(
            f"Average dispersion (max–min cosine within a decade) is {avg_band:.3f}, with a peak dispersion of {peak_band:.3f} in the {peak_dec}s."
        )
        lines.append(
            f"The most stable adjacent-decade interval (smallest mean-cosine step) is {stable_step:.3f} between {stable_from}s and {stable_to}s."
        )
        # Add 1–2 concise closing sentences based on thresholds (still data-based)
        if mean_range < 0.05:
            lines.append("Overall, the time series is relatively flat, consistent with high semantic stability in the neighborhood.")
        else:
            lines.append("Overall, the time series shows non-trivial variation, consistent with semantic drift in the neighborhood.")
        if peak_band > 0.15:
            lines.append("The wide dispersion peak suggests a decade with less coherent nearest-neighbor structure (higher variability).")
        else:
            lines.append("Dispersion remains moderate across decades, suggesting a fairly coherent nearest-neighbor structure.")

        lines.append("")  # blank line between words

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Generate stability + dispersion plots from neighbors CSVs.")
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more neighbors CSV paths (e.g. results/neighbors_*.csv).",
    )
    ap.add_argument(
        "--words",
        nargs="*",
        default=None,
        help="Optional filter: only these target words.",
    )
    ap.add_argument("--out_dir", default="drift_plots", help="Output directory for plots and report.")
    ap.add_argument("--prefix", default="semantic_drift", help="Prefix for output filenames.")
    args = ap.parse_args()

    inputs = [Path(p) for p in args.inputs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_neighbors_csvs(inputs)

    # Optional filter words
    if args.words:
        wanted = set(args.words)
        df = df[df["word"].isin(wanted)]
        if df.empty:
            raise ValueError(f"No rows left after filtering for words={args.words}")

    agg = aggregate_metrics(df)

    words = sorted(agg["word"].unique())

    # Outputs
    stability_png = out_dir / f"{args.prefix}_stability.png"
    dispersion_png = out_dir / f"{args.prefix}_dispersion.png"
    report_txt = out_dir / f"{args.prefix}_report.txt"
    agg_csv = out_dir / f"{args.prefix}_aggregated.csv"

    agg.to_csv(agg_csv, index=False)

    plot_stability_curve(agg, stability_png, words=words)
    plot_dispersion(agg, dispersion_png, words=words)
    generate_brief_report(agg, report_txt, words=words)

    print(f"Wrote:\n- {stability_png}\n- {dispersion_png}\n- {agg_csv}\n- {report_txt}")


if __name__ == "__main__":
    main()
