import numpy as np
import matplotlib.pyplot as plt

from src.config import DECADES


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
        raise ValueError("--from_decade must be <= --to_decade in time order")
    return list(DECADES[a : b + 1])


def _set_margins(ax, pad_frac: float = 0.12):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    sx = max(1e-9, xmax - xmin)
    sy = max(1e-9, ymax - ymin)
    ax.set_xlim(xmin - pad_frac * sx, xmax + pad_frac * sx)
    ax.set_ylim(ymin - pad_frac * sy, ymax + pad_frac * sy)


def _auto_label_offset(ax, user_dx: float, user_dy: float):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    sx = max(1e-9, xmax - xmin)
    sy = max(1e-9, ymax - ymin)

    dx = (0.012 * sx) if user_dx == 0.0 else user_dx
    dy = (0.012 * sy) if user_dy == 0.0 else user_dy

    dx = np.sign(dx) * min(abs(dx), 0.05 * sx)
    dy = np.sign(dy) * min(abs(dy), 0.05 * sy)
    return float(dx), float(dy)


def plot_pca_window(
    X2_all: np.ndarray,
    words: list[str],
    ref_decade: str,
    out_path,
    connect: bool,
    label_dx: float,
    label_dy: float,
    from_dec: str | None,
    to_dec: str | None,
):
    decs_plot = decade_slice(from_dec, to_dec)
    n_dec_total = len(DECADES)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(11, 8))

    # points + lines
    for wi, w in enumerate(words):
        base = wi * n_dec_total
        idxs = [base + DECADES.index(d) for d in decs_plot]

        ax.scatter(X2_all[idxs, 0], X2_all[idxs, 1], color=colors[wi % len(colors)],
                   label=w if len(words) > 1 else None)

        if connect and len(idxs) >= 2:
            ax.plot(X2_all[idxs, 0], X2_all[idxs, 1], color=colors[wi % len(colors)], linewidth=1.2)

    ax.relim()
    ax.autoscale_view()
    _set_margins(ax, pad_frac=0.15)
    dx, dy = _auto_label_offset(ax, label_dx, label_dy)

    # labels
    for wi, w in enumerate(words):
        base = wi * n_dec_total
        for d in decs_plot:
            i = base + DECADES.index(d)
            ax.text(X2_all[i, 0] + dx, X2_all[i, 1] + dy, d,
                    fontsize=9 if len(words) == 1 else 8,
                    clip_on=True)

    title = f"PCA aligned to {ref_decade}"
    if from_dec or to_dec:
        title += f" (window: {decs_plot[0]}–{decs_plot[-1]})"
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if len(words) > 1:
        ax.legend()

    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.1)
    fig.savefig(out_path, dpi=350)
    plt.close(fig)


def plot_tsne(
    X2: np.ndarray,
    words: list[str],
    ref_decade: str,
    out_path,
    connect: bool,
    label_dx: float,
    label_dy: float,
    perplexity: int,
):
    n_dec = len(DECADES)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(11, 8))

    for wi, w in enumerate(words):
        idxs = [wi * n_dec + di for di in range(n_dec)]

        ax.scatter(X2[idxs, 0], X2[idxs, 1], color=colors[wi % len(colors)],
                   label=w if len(words) > 1 else None)
        if connect:
            ax.plot(X2[idxs, 0], X2[idxs, 1], color=colors[wi % len(colors)], linewidth=1.2)

    ax.relim()
    ax.autoscale_view()
    _set_margins(ax, pad_frac=0.15)
    dx, dy = _auto_label_offset(ax, label_dx, label_dy)

    for wi, w in enumerate(words):
        for di, dec in enumerate(DECADES):
            i = wi * n_dec + di
            ax.text(X2[i, 0] + dx, X2[i, 1] + dy, dec,
                    fontsize=9 if len(words) == 1 else 8,
                    clip_on=True)

    ax.set_title(f"t-SNE aligned to {ref_decade} (perplexity={perplexity})")
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    if len(words) > 1:
        ax.legend()

    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.1)
    fig.savefig(out_path, dpi=350)
    plt.close(fig)
