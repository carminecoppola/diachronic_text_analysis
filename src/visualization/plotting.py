# Plotting utilities for 2D trajectories (PCA / t-SNE).
# The key design goals are:
# - Keep the plotting routine generic (same function for PCA and t-SNE)
# - Improve readability by expanding axes limits automatically
# - Avoid label overlap by moving only the text (not the data points)

import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

from src.config import DECADES


def _expand_axes(ax, X, frac=0.25):
    """
    Expand axes limits proportionally to data spread.

    This is a visualization-only adjustment:
    - It does not change the data
    - It only increases plot margins so that labels have space

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to modify.
    X : array-like, shape (N, 2)
        2D coordinates to be plotted.
    frac : float
        Fraction of the data range added as padding on each side.
    """
    xmin, xmax = X[:, 0].min(), X[:, 0].max()
    ymin, ymax = X[:, 1].min(), X[:, 1].max()

    dx = xmax - xmin
    dy = ymax - ymin

    # If all points share the same coordinate, force a non-zero range
    # to avoid degenerate axes limits.
    if dx == 0:
        dx = 1.0
    if dy == 0:
        dy = 1.0

    # Add proportional padding to both axes
    ax.set_xlim(xmin - frac * dx, xmax + frac * dx)
    ax.set_ylim(ymin - frac * dy, ymax + frac * dy)


def plot_trajectory(
    X2: np.ndarray,
    words: list[str],
    title: str,
    out_path,
    connect: bool = True,
    xlabel: str = "dim1",
    ylabel: str = "dim2",
):
    """
    Generic plot for PCA / t-SNE trajectories with automatic space expansion
    and collision-free labels.

    Assumptions about X2 ordering:
    - X2 contains one 2D point per (word, decade) pair.
    - The expected ordering is:
        for word in words:
            for decade in DECADES:
                append point(word, decade)
      equivalently, for each word the decade points are contiguous.

    This ordering is used to reconstruct the per-word trajectory.

    Parameters
    ----------
    X2 : np.ndarray
        2D coordinates of shape (len(words)*len(DECADES), 2).
    words : list[str]
        Target words to plot.
    title : str
        Plot title.
    out_path : str or Path
        Output path for the PNG file.
    connect : bool
        If True, connect decade points with a line to show temporal trajectory.
    xlabel, ylabel : str
        Axis labels.
    """
    # Number of decades determines how many points belong to each word trajectory
    n_dec = len(DECADES)

    # Use matplotlib default color cycle; each word gets one color
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(11, 8))

    # Text objects are collected so adjustText can move them to avoid overlaps
    texts = []

    for wi, w in enumerate(words):
        # Compute indices in X2 belonging to this word:
        # The points for word wi occupy the slice:
        #   wi*n_dec, wi*n_dec + 1, ..., wi*n_dec + (n_dec-1)
        idxs = [wi * n_dec + di for di in range(n_dec)]
        pts = X2[idxs]

        # Scatter plot for decade points of the word
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=35,
            color=colors[wi % len(colors)],
            label=w if len(words) > 1 else None,
            zorder=2,  # draw points above lines
        )

        # Optionally connect consecutive decades with a line
        if connect:
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=colors[wi % len(colors)],
                linewidth=1.2,
                zorder=1,  # draw line behind points
            )

        # Add a decade label at each point location.
        # These labels may overlap; adjustText will move them.
        for di, dec in enumerate(DECADES):
            texts.append(
                ax.text(
                    pts[di, 0],
                    pts[di, 1],
                    dec,
                    fontsize=9,
                    zorder=3,  # draw text above points
                )
            )

    # Expand axes BEFORE adjusting text:
    # This gives adjustText room to resolve overlaps without pushing labels outside the visible area.
    _expand_axes(ax, X2, frac=0.30)

    # Repel labels from each other, keeping points fixed.
    # arrowprops draws light connector lines from the moved label to its point.
    adjust_text(
        texts,
        ax=ax,
        expand_points=(1.2, 1.4),
        expand_text=(1.2, 1.4),
        arrowprops=dict(arrowstyle="-", lw=0.4, color="gray"),
    )

    # Titles and axis labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Only show legend when multiple words are plotted
    if len(words) > 1:
        ax.legend()

    # Tight layout reduces unnecessary whitespace while respecting label extents
    fig.tight_layout()

    # Save at high DPI for readability in reports
    fig.savefig(out_path, dpi=300)

    # Always close the figure to avoid memory accumulation in batch runs
    plt.close(fig)
