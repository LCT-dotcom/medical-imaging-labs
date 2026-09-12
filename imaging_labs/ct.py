"""CT experiments with an exact matched discrete projector for SIRT/CGLS."""
import numpy as np
from scipy.sparse import coo_matrix
from skimage.transform import radon, iradon
from .common import finite_image


def circle_phantom(size=128):
    yy, xx = np.ogrid[:size, :size]
    return (((xx - (size - 1) / 2) ** 2 + (yy - (size - 1) / 2) ** 2) <= (size / 4) ** 2).astype(float)


def projection_matrix(size, angles):
    """Pixel-driven parallel projection with linear detector-bin deposition.

    Flatten sinograms in angle-major order: (angle, detector). This pedagogical
    discretization is distinct from skimage.radon. Its transpose is exact.
    """
    angles = np.asarray(angles, dtype=float)
    if size < 2 or angles.ndim != 1 or len(angles) == 0 or not np.isfinite(angles).all():
        raise ValueError('Positive image size and finite projection angles are required')
    if size > 128:
        raise ValueError('Use size <= 128 for the explicit educational system matrix')
    detectors = int(np.ceil(np.sqrt(2) * size)) + 2
    yy, xx = np.indices((size, size), dtype=float)
    xx, yy = xx.ravel() - (size - 1) / 2, yy.ravel() - (size - 1) / 2
    columns = np.arange(size * size)
    rows_all, cols_all, weights_all = [], [], []
    for index, angle in enumerate(np.deg2rad(angles)):
        coordinate = xx * np.cos(angle) + yy * np.sin(angle) + (detectors - 1) / 2
        left = np.floor(coordinate).astype(int)
        fraction = coordinate - left
        for bins, weights in ((left, 1 - fraction), (left + 1, fraction)):
            valid = (bins >= 0) & (bins < detectors)
            rows_all.append(index * detectors + bins[valid])
            cols_all.append(columns[valid])
            weights_all.append(weights[valid])
    matrix = coo_matrix((np.concatenate(weights_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
                        shape=(len(angles) * detectors, size * size)).tocsr()
    return matrix, detectors


def _problem(matrix, measurements, iterations):
    b = np.asarray(measurements, dtype=float).ravel()
    if b.size != matrix.shape[0] or not np.isfinite(b).all() or iterations < 1:
        raise ValueError('Invalid measurement vector or iteration count')
    return b, np.zeros(matrix.shape[1]), max(np.linalg.norm(b), np.finfo(float).eps)


def sirt(matrix, measurements, iterations=40, relaxation=1.0):
    if not 0 < relaxation <= 1:
        raise ValueError('Use 0 < relaxation <= 1')
    b, x, norm = _problem(matrix, measurements, iterations)
    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    col_sum = np.asarray(matrix.sum(axis=0)).ravel()
    row_weight = np.divide(1., row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
    col_weight = np.divide(1., col_sum, out=np.zeros_like(col_sum), where=col_sum > 0)
    history = [float(np.linalg.norm(b) / norm)]
    for _ in range(iterations):
        x += relaxation * col_weight * (matrix.T @ (row_weight * (b - matrix @ x)))
        x = np.maximum(x, 0)
        history.append(float(np.linalg.norm(b - matrix @ x) / norm))
    return x, np.asarray(history)


def cgls(matrix, measurements, iterations=40, tolerance=1e-10):
    b, x, norm = _problem(matrix, measurements, iterations)
    residual = b.copy()
    gradient = matrix.T @ residual
    direction = gradient.copy()
    gamma = float(gradient @ gradient)
    initial_gamma = gamma
    history = [float(np.linalg.norm(residual) / norm)]
    for _ in range(iterations):
        if gamma <= tolerance ** 2 * max(initial_gamma, np.finfo(float).eps):
            break
        projected = matrix @ direction
        denominator = float(projected @ projected)
        if denominator <= np.finfo(float).tiny:
            break
        step = gamma / denominator
        x += step * direction
        residual -= step * projected
        gradient = matrix.T @ residual
        next_gamma = float(gradient @ gradient)
        direction = gradient + (next_gamma / gamma) * direction
        gamma = next_gamma
        history.append(float(np.linalg.norm(b - matrix @ x) / norm))
    return x, np.asarray(history)


def fourier_filter(sinogram, name='ramp'):
    """Zero-padded continuous ramp/windowed filtering along detector axis.

    This explicitly shows FFT -> filter -> IFFT. It approximates a continuous
    filter and is not identical to scikit-image's discrete ramp correction.
    """
    sinogram = finite_image(sinogram)
    count = sinogram.shape[0]
    padded = 1 << int(np.ceil(np.log2(max(64, 2 * count))))
    frequency = np.fft.fftfreq(padded)
    response = 2 * np.abs(frequency)
    windows = {'ramp': np.ones_like(frequency), 'shepp-logan': np.sinc(frequency),
               'cosine': np.cos(np.pi * frequency),
               'hamming': 0.54 + 0.46 * np.cos(2 * np.pi * frequency),
               'hann': 0.5 + 0.5 * np.cos(2 * np.pi * frequency)}
    if name not in windows:
        raise ValueError(f'Unknown filter: {name}')
    response *= windows[name]
    spectrum = np.fft.fft(sinogram, n=padded, axis=0)
    filtered = np.fft.ifft(spectrum * response[:, None], axis=0).real[:count]
    return filtered, frequency, response
