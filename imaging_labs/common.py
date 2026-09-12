"""Shared validation, display windows and reference-based metrics."""
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity


def finite_image(image, ndim=2):
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != ndim or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f'Expected a nonempty, finite {ndim}D array')
    return array


def normalize_display(image, percentiles=(1, 99)):
    """Window for viewing only; do not feed this into quantitative metrics."""
    array = np.asarray(image, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError('Display input must contain finite pixels')
    lo, hi = np.percentile(array, percentiles)
    if hi <= lo:
        lo, hi = array.min(), array.max()
    if hi <= lo:
        return np.zeros_like(array)
    return np.clip((array - lo) / (hi - lo), 0, 1)


def image_metrics(reference, reconstruction, data_range=1.0):
    """Compare on the supplied intensity scale without independent normalization."""
    ref, rec = finite_image(reference), finite_image(reconstruction)
    if ref.shape != rec.shape or data_range <= 0:
        raise ValueError('Matching shapes and positive data_range are required')
    diff = rec - ref
    mse = float(np.mean(diff ** 2))
    minimum_side = min(ref.shape)
    window = min(7, minimum_side if minimum_side % 2 else minimum_side - 1)
    return {
        'mae': float(np.mean(np.abs(diff))),
        'rmse': float(np.sqrt(mse)),
        'psnr_db': float('inf') if mse == 0 else float(10 * np.log10(data_range ** 2 / mse)),
        'ssim': float(structural_similarity(ref, rec, data_range=data_range, win_size=window)) if window >= 3 else float('nan'),
    }


def show_images(images, titles, *, columns=3, cmap='gray', limits=None):
    import matplotlib.pyplot as plt
    rows = int(np.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.5 * rows), squeeze=False)
    for ax, image, title in zip(axes.flat, images, titles):
        kwargs = {} if limits is None else {'vmin': limits[0], 'vmax': limits[1]}
        ax.imshow(image, cmap=cmap, **kwargs)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    for ax in list(axes.flat)[len(images):]:
        ax.axis('off')
    fig.tight_layout()
    return fig


def save_figure(fig, folder, filename):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / filename, dpi=130, bbox_inches='tight')
