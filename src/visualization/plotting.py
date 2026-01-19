import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

from src.config import DECADES


def _expand_axes(ax, X, frac=0.25):
    """
    Expand axes limits proportionally to data spread.
    """
    xmin, xmax = X[:, 0].min(), X[:, 0].max()
    ymin, ymax = X[:, 1].min(), X[:, 1].max()

    dx = xmax - xmin
    dy = ymax - ymin
    if dx == 0:
        dx = 1.0
    if dy == 0:
        dy = 1.0

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
    """
    n_dec = len(DECADES)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(11, 8))
    texts = []

    for wi, w in enumerate(words):
        idxs = [wi * n_dec + di for di in range(n_dec)]
        pts = X2[idxs]

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=35,
            color=colors[wi % len(colors)],
            label=w if len(words) > 1 else None,
            zorder=2,
        )

        if connect:
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=colors[wi % len(colors)],
                linewidth=1.2,
                zorder=1,
            )

        for di, dec in enumerate(DECADES):
            texts.append(
                ax.text(
                    pts[di, 0],
                    pts[di, 1],
                    dec,
                    fontsize=9,
                    zorder=3,
                )
            )

    # expand axes BEFORE adjusting text
    _expand_axes(ax, X2, frac=0.30)

    # repel labels from each other (no data distortion)
    adjust_text(
        texts,
        ax=ax,
        expand_points=(1.2, 1.4),
        expand_text=(1.2, 1.4),
        arrowprops=dict(arrowstyle="-", lw=0.4, color="gray"),
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if len(words) > 1:
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
