"""Small deterministic, synthetic fixtures; no patient data or external assets."""
from pathlib import Path
import numpy as np


def dsa_demo(size=192):
    import cv2
    rng = np.random.default_rng(23)
    yy, xx = np.mgrid[:size, :size]
    mask = 110 + 50*np.exp(-((xx-size*.45)**2+(yy-size*.5)**2)/(size*.35)**2)
    mask += 18*np.cos(xx/12)*np.exp(-((yy-size*.5)/(size*.3))**2)
    mask += rng.normal(0, 1, mask.shape)
    live = mask.copy()
    curve = size*.5 + size*.13*np.sin(yy/22)
    live -= 30*np.exp(-((xx-curve)/2.2)**2)
    matrix = cv2.getRotationMatrix2D(((size-1)/2, (size-1)/2), 12, 1)
    rotated = cv2.warpAffine(live.astype('float32'), matrix, (size, size), borderValue=110)
    return mask, live, rotated


def write_mri_demo(folder, oblique=False):
    import nibabel as nib
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
    folder = Path(folder)
    dicom = folder / 'dicom'
    dicom.mkdir(parents=True, exist_ok=True)
    rows, columns, slices = 40, 44, 12
    yy, xx, zz = np.mgrid[:rows, :columns, :slices]
    volume = (1000*np.exp(-((xx-20)/12)**2-((yy-18)/13)**2-((zz-5)/5)**2)).astype('uint16')
    # A normal with zero z component exercises non-axial slice sorting.
    col_dir = np.array([0., 1., 0.]) if oblique else np.array([1., 0., 0.])
    row_dir = np.array([0., 0., 1.]) if oblique else np.array([0., 1., 0.])
    normal = np.cross(col_dir, row_dir)
    origin = np.array([10., 20., 30.])
    spacing = [1.1, .9]
    step = normal*2.5
    affine = np.eye(4)
    affine[:3, 0], affine[:3, 1], affine[:3, 2], affine[:3, 3] = row_dir*spacing[0], col_dir*spacing[1], step, origin
    series_uid, study_uid = generate_uid(), generate_uid()
    for index in range(slices):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=meta, preamble=b'\0'*128)
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID, ds.StudyInstanceUID = series_uid, study_uid
        ds.Modality = 'MR'
        ds.PatientName = 'SYNTHETIC^DEMO'
        ds.PatientID = 'SYNTHETIC'
        ds.Rows, ds.Columns = rows, columns
        ds.ImageOrientationPatient = list(col_dir)+list(row_dir)
        ds.ImagePositionPatient = list(origin+step*index)
        ds.PixelSpacing, ds.SliceThickness = spacing, 2.5
        ds.InstanceNumber = index+1
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.PixelData = np.ascontiguousarray(volume[:, :, index], dtype='<u2').tobytes()
        ds.save_as(dicom / f'slice_{slices-index:03d}.dcm', enforce_file_format=True)
    nifti = folder / 'synthetic_mri.nii.gz'
    nib.save(nib.Nifti1Image(volume, np.diag([-1, -1, 1, 1]) @ affine), nifti)
    return {'dicom': dicom, 'nifti': nifti}
