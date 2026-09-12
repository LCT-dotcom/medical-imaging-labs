# Local data setup

Extract the authorized coursework archives outside the repository, or into this ignored folder:

```text
data/
  dsa/
    mask.jpg
    live.jpg
    rotated_live.jpg
  ct/
    CT1.bmp
    CT2.bmp
    CT3.bmp
    logan.bmp
    New_CT.png
  mri/
    Neurohacking_data-0.0/
      BRAINIX/
        DICOM/
          T1/*.dcm
          T2/*.dcm
          FLAIR/*.dcm
        NIfTI/
          BRAINIX_NIFTI_T1.nii.gz
          BRAINIX_NIFTI_T2.nii.gz
          BRAINIX_NIFTI_FLAIR.nii.gz
```

`Lab 3 MRI data.zip` contains another archive named `Neurohacking_data-0.0.zip`; extract that inner archive too. Ignore `__MACOSX` and `.DS_Store` metadata.

Set `DATA_ROOT` to the directory containing `dsa`, `ct` and `mri`, and set `MODE='local'`. The CT notebook initially selects CT1.bmp; change `CT_PATH` to another supplied image for a separate experiment. The MRI notebook explicitly selects T1; switch `SEQUENCE` to T2 or FLAIR only when both matching representations are present.

Do not publish source DICOM metadata, raw medical images, original notebook outputs with identifiers, or the instructor's slide PDFs without appropriate authorization. Public example outputs in this repository use synthetic inputs only.
