"""Conventional single-frame MRI loading, geometry and exploratory measurements."""
from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import pydicom
import cv2
from scipy.ndimage import binary_fill_holes
from .common import finite_image, normalize_display, image_metrics


def load_dicom_series(folder):
    """Load one regular, monochrome single-frame series into a RAS+ affine.

    The array order is (row, column, slice). Slice order is determined by the
    projection of ImagePositionPatient onto the slice normal, including oblique
    series. This is an educational loader, not a general DICOM converter.
    """
    files = sorted(Path(folder).glob('*.dcm'))
    if len(files) < 2:
        raise ValueError('Provide at least two .dcm slices from one series')
    datasets = [pydicom.dcmread(path) for path in files]
    if len({str(ds.get('SeriesInstanceUID', '')) for ds in datasets}) != 1:
        raise ValueError('Folder contains multiple DICOM series')
    required = ('ImageOrientationPatient', 'ImagePositionPatient', 'PixelSpacing', 'SeriesInstanceUID')
    if any(any(key not in ds for key in required) for ds in datasets):
        raise ValueError('Required DICOM spatial metadata is missing')
    first = datasets[0]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    column_direction, row_direction = orientation[:3], orientation[3:]
    if not np.allclose([np.linalg.norm(column_direction), np.linalg.norm(row_direction), np.dot(column_direction, row_direction)], [1, 1, 0], atol=1e-4):
        raise ValueError('Invalid DICOM orientation cosines')
    normal = np.cross(column_direction, row_direction)
    spacing = np.asarray(first.PixelSpacing, dtype=float)
    if np.any(spacing <= 0):
        raise ValueError('Pixel spacing must be positive')
    for ds in datasets:
        if int(ds.get('NumberOfFrames', 1)) != 1 or ds.get('PhotometricInterpretation') != 'MONOCHROME2':
            raise ValueError('Only single-frame MONOCHROME2 series are supported')
        if (ds.Rows, ds.Columns) != (first.Rows, first.Columns):
            raise ValueError('DICOM slice shapes differ')
        if not np.allclose(ds.ImageOrientationPatient, orientation, atol=1e-4) or not np.allclose(ds.PixelSpacing, spacing):
            raise ValueError('Inconsistent orientation or pixel spacing in series')
    datasets.sort(key=lambda ds: float(np.dot(np.asarray(ds.ImagePositionPatient, float), normal)))
    positions = np.asarray([ds.ImagePositionPatient for ds in datasets], dtype=float)
    differences = np.diff(positions, axis=0)
    step = differences.mean(axis=0)
    if np.dot(step, normal) <= 1e-5 or not np.allclose(differences, step, atol=0.05, rtol=1e-3):
        raise ValueError('Duplicate or irregularly spaced DICOM slices')
    slices = []
    for ds in datasets:
        array = ds.pixel_array.astype('float32')
        slices.append(array * float(ds.get('RescaleSlope', 1)) + float(ds.get('RescaleIntercept', 0)))
    volume = np.stack(slices, axis=2)
    affine_lps = np.eye(4)
    affine_lps[:3, 0] = row_direction * spacing[0]
    affine_lps[:3, 1] = column_direction * spacing[1]
    affine_lps[:3, 2] = step
    affine_lps[:3, 3] = positions[0]
    affine_ras = np.diag([-1, -1, 1, 1]) @ affine_lps
    return nib.Nifti1Image(volume, affine_ras)


def load_nifti(path, time_index=None):
    image = nib.load(Path(path))
    if len(image.shape) == 4:
        if time_index is None:
            raise ValueError('4D input requires an explicit time_index')
        if not 0 <= time_index < image.shape[3]:
            raise ValueError('time_index is outside the input volume')
        image = nib.Nifti1Image(np.asarray(image.dataobj[..., time_index], dtype='float32'), image.affine)
    if len(image.shape) != 3 or not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-10:
        raise ValueError('A 3D image with a nonsingular spatial affine is required')
    return image


def spatial_summary(image):
    return {'shape': tuple(image.shape), 'voxel_spacing_mm': tuple(float(v) for v in nib.affines.voxel_sizes(image.affine)),
            'voxel_axis_codes': ''.join(nib.aff2axcodes(image.affine)),
            'obliquity_deg': tuple(np.round(np.rad2deg(nib.affines.obliquity(image.affine)), 2))}


def canonical_data(image):
    """Reorder/flip axes toward RAS+; this does not de-oblique the acquisition."""
    canonical = nib.as_closest_canonical(image)
    return canonical.get_fdata(dtype=np.float32)


def compare_volumes(reference, moving, *, same_acquisition=False):
    """Resample in world coordinates; caller must establish acquisition identity.

    This is spatial resampling, not intersubject or motion registration. Intensity
    scaling must already be comparable. No SSIM-maximizing flips are searched.
    """
    if not same_acquisition:
        raise ValueError('Confirm that inputs represent the same acquisition before comparing')
    reference = nib.as_closest_canonical(reference)
    moving = nib.as_closest_canonical(moving)
    ref = reference.get_fdata(dtype=np.float32)
    if moving.shape == reference.shape and np.allclose(moving.affine, reference.affine, rtol=1e-6, atol=1e-5):
        rec = moving.get_fdata(dtype=np.float32)
    else:
        floating = nib.Nifti1Image(moving.get_fdata(dtype=np.float32), moving.affine)
        rec = resample_from_to(floating, reference, order=1, cval=np.nan).get_fdata(dtype=np.float32)
    valid = np.isfinite(ref) & np.isfinite(rec)
    if valid.sum() < 64:
        raise ValueError('Insufficient spatial overlap between image volumes')
    difference = rec[valid] - ref[valid]
    data_range = float(np.ptp(ref[valid]))
    return {'mae': float(np.mean(np.abs(difference))), 'rmse': float(np.sqrt(np.mean(difference ** 2))),
            'reference_range': data_range, 'overlap_fraction': float(valid.mean()),
            'reference': ref, 'aligned': rec, 'valid': valid}


def slice_quality(image):
    """Descriptive slice features, not a clinical SNR or format-quality ranking."""
    raw = finite_image(image)
    normalized = normalize_display(raw)
    u8 = np.round(255 * normalized).astype('uint8')
    _, foreground = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
    if count > 1:
        foreground = labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    else:
        foreground = np.zeros(raw.shape, dtype=bool)
    foreground = binary_fill_holes(foreground)
    inner = cv2.dilate(foreground.astype('uint8'), np.ones((5, 5), 'uint8')) > 0
    outer = cv2.dilate(foreground.astype('uint8'), np.ones((17, 17), 'uint8')) > 0
    ring = outer & ~inner
    # Constant padding is not a valid noise estimate.
    samples = raw[ring]
    noise = float(1.4826 * np.median(np.abs(samples - np.median(samples)))) if samples.size >= 30 else float('nan')
    if not np.isfinite(noise) or noise <= 1e-12:
        noise = float('nan')
    mean_signal = float(raw[foreground].mean()) if foreground.any() else float('nan')
    histogram = np.bincount(u8.ravel(), minlength=256).astype(float)
    probabilities = histogram[histogram > 0] / u8.size
    metrics = {'laplacian_variance': float(cv2.Laplacian(normalized, cv2.CV_64F).var()),
               'gradient_energy': float(np.mean(cv2.Sobel(normalized, cv2.CV_64F, 1, 0) ** 2 + cv2.Sobel(normalized, cv2.CV_64F, 0, 1) ** 2)),
               'entropy_bits': float(-np.sum(probabilities * np.log2(probabilities))),
               'raw_p99_minus_p01': float(np.diff(np.percentile(raw, [1, 99]))[0]),
               'background_mad_sigma_proxy': noise, 'foreground_to_noise_proxy': mean_signal / noise,
               'foreground_pixels': int(foreground.sum()), 'background_pixels': int(ring.sum())}
    return metrics, foreground, ring


def slice_viewer(image, title='MRI', display_slices=None):
    """Native or interpolated slider; interpolation is display-only."""
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display
    data = canonical_data(image)
    native = data.shape[2]
    display_count = native if display_slices is None else int(display_slices)
    if display_count < 1:
        raise ValueError('display_slices must be positive')
    lo, hi = np.percentile(data, [1, 99])
    slider = widgets.IntSlider(min=0, max=display_count-1, value=display_count//2,
                               description='Slice', continuous_update=False)
    output = widgets.Output()

    def draw(change=None):
        coordinate = slider.value * (native-1) / max(display_count-1, 1)
        lower = int(np.floor(coordinate))
        upper = min(lower+1, native-1)
        weight = coordinate-lower
        plane = (1-weight)*data[:, :, lower] + weight*data[:, :, upper]
        with output:
            output.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(plane.T, origin='lower', cmap='gray', vmin=lo, vmax=hi)
            suffix = 'native' if display_count == native else 'interpolated display only'
            ax.set_title(f'{title}: z={coordinate:.2f} ({suffix})')
            ax.set_xlabel('Canonical voxel axis 0')
            ax.set_ylabel('Canonical voxel axis 1')
            display(fig)
            plt.close(fig)
    slider.observe(draw, names='value')
    draw()
    return widgets.VBox([slider, output])
