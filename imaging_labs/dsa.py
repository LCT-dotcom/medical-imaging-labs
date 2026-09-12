"""Signed subtraction, NCC/ECC registration and exploratory vessel enhancement."""
from pathlib import Path
import cv2
import numpy as np
from skimage.filters import frangi
from .common import finite_image, normalize_display


def load_grayscale(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Image not found: {path.name}. Check DATA_ROOT.')
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Cannot decode image: {path.name}')
    return image


def subtract_images(mask, live, sigma=0.0):
    mask, live = finite_image(mask), finite_image(live)
    if mask.shape != live.shape:
        raise ValueError('Mask and live shapes differ; resampling must be explicit')
    if sigma < 0:
        raise ValueError('sigma must be nonnegative')
    if sigma:
        mask = cv2.GaussianBlur(mask, (0, 0), sigma)
        live = cv2.GaussianBlur(live, (0, 0), sigma)
    return live - mask


def register_live(mask, live, *, rotation_search=True, motion='affine'):
    """Coarse/fine NCC rotation followed by ECC; return valid common support.

    ECC estimates a template-to-input warp, hence WARP_INVERSE_MAP when
    resampling the moving image. Borders are excluded from subtraction metrics.
    """
    mask, live = finite_image(mask), finite_image(live)
    if mask.shape != live.shape:
        raise ValueError('Registration requires equal image shapes')
    if np.std(mask) < 1e-8 or np.std(live) < 1e-8:
        raise ValueError('Registration requires nonconstant images')
    modes = {'translation': cv2.MOTION_TRANSLATION, 'euclidean': cv2.MOTION_EUCLIDEAN, 'affine': cv2.MOTION_AFFINE}
    if motion not in modes:
        raise ValueError(f'Unknown motion model: {motion}')
    height, width = mask.shape
    template = normalize_display(mask, (0, 100)).astype('float32')
    moving = normalize_display(live, (0, 100)).astype('float32')
    center = ((width - 1) / 2, (height - 1) / 2)
    support = np.ones(mask.shape, dtype=np.uint8)

    def rotation(angle):
        return cv2.getRotationMatrix2D(center, float(angle), 1.0).astype('float32')

    def ncc(angle):
        matrix = rotation(angle)
        rotated = cv2.warpAffine(moving, matrix, (width, height))
        valid = cv2.warpAffine(support, matrix, (width, height), flags=cv2.INTER_NEAREST) > 0
        valid = cv2.erode(valid.astype('uint8'), np.ones((5, 5), 'uint8')) > 0
        if valid.sum() < max(25, mask.size // 4):
            return -np.inf
        a, b = template[valid], rotated[valid]
        a, b = a - a.mean(), b - b.mean()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    angle = 0.0
    if rotation_search:
        coarse = np.arange(-45, 46, 1)
        angle = float(coarse[np.argmax([ncc(a) for a in coarse])])
        fine = np.arange(angle - 1, angle + 1.001, 0.1)
        angle = float(fine[np.argmax([ncc(a) for a in fine])])
    rotation_matrix = rotation(angle)
    rotated = cv2.warpAffine(moving, rotation_matrix, (width, height))
    rotated_support = cv2.warpAffine(support, rotation_matrix, (width, height), flags=cv2.INTER_NEAREST)
    warp = np.eye(2, 3, dtype='float32')
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    # OpenCV can window its inputs in-place; retain the original ECC inputs.
    shift, response = cv2.phaseCorrelate(template.copy(), rotated.copy(), window)
    if response > 0.1 and abs(shift[0]) < width / 3 and abs(shift[1]) < height / 3:
        warp[:, 2] = shift
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)
    try:
        correlation, warp = cv2.findTransformECC(template, rotated, warp, modes[motion], criteria, rotated_support, 5)
    except cv2.error as error:
        raise RuntimeError('ECC did not converge. Inspect image overlap or try motion="translation".') from error
    # Combine transforms to resample original intensities once.
    coarse3 = np.vstack([rotation_matrix, [0, 0, 1]])
    ecc3 = np.vstack([warp, [0, 0, 1]])
    total = (np.linalg.inv(coarse3) @ ecc3)[:2].astype('float32')
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    registered = cv2.warpAffine(live.astype('float32'), total, (width, height), flags=flags)
    valid = cv2.warpAffine(support, total, (width, height), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    valid = cv2.erode(valid, np.ones((3, 3), dtype='uint8')) > 0
    return {'registered': registered, 'valid': valid, 'rotation_deg': angle,
            'ecc_correlation': float(correlation), 'template_to_live': total}


def enhance_vessels(subtraction):
    """Retain black-hat, multiscale and Frangi variants as exploratory displays."""
    base = np.round(255 * normalize_display(subtraction)).astype('uint8')
    denoised = cv2.fastNlMeansDenoising(base, None, 12, 7, 21)
    blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    bilateral = cv2.bilateralFilter(base, 7, 40, 40)
    multi = np.maximum.reduce([cv2.morphologyEx(bilateral, cv2.MORPH_BLACKHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) for k in (15, 21, 31)])
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    local = clahe.apply(multi)
    sharp = cv2.addWeighted(local, 1.6, cv2.GaussianBlur(local, (0, 0), 1.2), -0.6, 0)
    vesselness = frangi(bilateral.astype(float) / 255, sigmas=range(1, 5), black_ridges=True)
    return {'Subtraction display': base, 'Single-scale black-hat': blackhat,
            'Multiscale black-hat + CLAHE': local, 'Mild sharpening': sharp,
            'Frangi (dark ridges)': normalize_display(vesselness, (0, 100))}
