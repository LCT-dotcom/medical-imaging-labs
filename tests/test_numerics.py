"""Regression checks for numerical and geometry changes made during cleanup."""
import numpy as np
import pytest

from imaging_labs.common import image_metrics, normalize_display
from imaging_labs.ct import projection_matrix, cgls, sirt
from imaging_labs.dsa import subtract_images, register_live
from imaging_labs.mri import load_dicom_series, load_nifti, compare_volumes
from imaging_labs.demo import write_mri_demo


def test_subtraction_preserves_negative_values():
    result = subtract_images(np.array([[250, 0]], dtype=np.uint8),
                             np.array([[10, 255]], dtype=np.uint8))
    np.testing.assert_array_equal(result, [[-240., 255.]])


def test_subtraction_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        subtract_images(np.zeros((2, 3)), np.zeros((3, 2)))


def test_constant_display_and_invalid_pixels():
    np.testing.assert_array_equal(normalize_display(np.ones((3, 3))), 0)
    with pytest.raises(ValueError):
        normalize_display(np.array([[np.nan]]))


def test_metrics_do_not_hide_gain_error():
    ref = np.linspace(0, 1, 64).reshape(8, 8)
    assert image_metrics(ref, ref * 2)['rmse'] > 0.5
    assert image_metrics(ref, ref)['rmse'] == 0


def test_projector_adjoint_identity():
    matrix, _ = projection_matrix(12, np.linspace(0, 180, 17, endpoint=False))
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=matrix.shape[1]), rng.normal(size=matrix.shape[0])
    np.testing.assert_allclose(np.dot(matrix @ x, y), np.dot(x, matrix.T @ y), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('solver', [sirt, cgls])
def test_iterative_reconstruction_reduces_residual(solver):
    matrix, _ = projection_matrix(16, np.linspace(0, 180, 45, endpoint=False))
    yy, xx = np.ogrid[:16, :16]
    phantom = (((xx - 7.5) ** 2 + (yy - 7.5) ** 2) < 16).astype(float)
    x, history = solver(matrix, matrix @ phantom.ravel(), iterations=40)
    assert np.isfinite(x).all()
    assert history[-1] < 0.2 * history[0]
    assert np.sqrt(np.mean((x - phantom.ravel()) ** 2)) < 0.16


@pytest.mark.parametrize('solver', [sirt, cgls])
def test_zero_measurement(solver):
    matrix, _ = projection_matrix(8, [0, 45, 90])
    x, history = solver(matrix, np.zeros(matrix.shape[0]), iterations=4)
    np.testing.assert_array_equal(x, 0)
    assert np.isfinite(history).all()


def test_registration_translation():
    import cv2
    rng = np.random.default_rng(8)
    mask = cv2.GaussianBlur(rng.random((96, 96)).astype('float32'), (0, 0), 2)
    live = cv2.warpAffine(mask, np.array([[1, 0, 4], [0, 1, -3]], dtype='float32'), (96, 96))
    result = register_live(mask, live, rotation_search=False, motion='translation')
    inner = result['valid'].copy()
    inner[:10] = inner[-10:] = False
    inner[:, :10] = inner[:, -10:] = False
    before = np.mean((live[inner] - mask[inner]) ** 2)
    after = np.mean((result['registered'][inner] - mask[inner]) ** 2)
    assert after < before * 0.15


def test_dicom_nifti_roundtrip_and_oblique_sort(tmp_path):
    paths = write_mri_demo(tmp_path, oblique=True)
    dicom = load_dicom_series(paths['dicom'])
    nifti = load_nifti(paths['nifti'])
    np.testing.assert_allclose(dicom.get_fdata(), nifti.get_fdata())
    np.testing.assert_allclose(dicom.affine, nifti.affine, atol=1e-5)
    assert compare_volumes(dicom, nifti, same_acquisition=True)['rmse'] < 1e-3


def test_pair_comparison_requires_acquisition_confirmation(tmp_path):
    paths = write_mri_demo(tmp_path)
    volume = load_nifti(paths['nifti'])
    with pytest.raises(ValueError, match='same acquisition'):
        compare_volumes(volume, volume, same_acquisition=False)


def test_equivalent_axis_orders_compare_without_interpolation(tmp_path):
    import nibabel as nib
    paths = write_mri_demo(tmp_path)
    image = load_nifti(paths['nifti'])
    canonical = nib.as_closest_canonical(image)
    result = compare_volumes(image, canonical, same_acquisition=True)
    assert result['rmse'] == 0
    assert result['overlap_fraction'] == 1


def test_resampling_integer_volume_keeps_fractional_values_and_overlap():
    import nibabel as nib
    values = np.indices((8, 8, 8))[0].astype('uint16')
    moving = nib.Nifti1Image(values, np.eye(4))
    affine = np.eye(4)
    affine[0, 3] = 0.5
    reference = nib.Nifti1Image(values.astype('float32') + 0.5, affine)
    result = compare_volumes(reference, moving, same_acquisition=True)
    assert result['rmse'] == 0
    assert result['overlap_fraction'] == 7/8
    assert np.isnan(result['aligned'][-1]).all()


def test_nifti_timepoint_is_explicit(tmp_path):
    import nibabel as nib
    path = tmp_path / 'timeseries.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8, 2)), np.eye(4)), path)
    with pytest.raises(ValueError, match='time_index'):
        load_nifti(path)
    assert load_nifti(path, time_index=1).shape == (8, 8, 8)


def test_mixed_dicom_series_rejected(tmp_path):
    import pydicom
    paths = write_mri_demo(tmp_path)
    path = next(paths['dicom'].glob('*.dcm'))
    ds = pydicom.dcmread(path)
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.save_as(path, enforce_file_format=True)
    with pytest.raises(ValueError, match='series'):
        load_dicom_series(paths['dicom'])
