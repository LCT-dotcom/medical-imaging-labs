# Changes from the submitted coursework notebooks

## Organization

- Three notebooks in the original lab order: DSA, CT, MRI.
- English headings, purpose, configuration, implementation, visualization, metrics and interpretation.
- Removed empty cells, repeated directory-discovery cells, repeated imports and superseded experiment variants.
- Reusable methods moved into `imaging_labs/`; all inputs configured before use.
- Replaced `/content/` and `/mnt/data/` paths with a configurable local data/output root.
- Original files are preserved outside the repository. Original outputs are not reused as evidence of the cleaned code.

## DSA

- Fixed mask/live paths being used before definition.
- Rejects shape mismatch instead of silently resizing a medical image.
- Retains signed float subtraction, post-subtraction equalization, smoothing, black-hat/CLAHE, multiscale enhancement and Frangi.
- Initializes NCC/ECC registration with phase-correlation translation and combines transforms before resampling original intensities.
- Excludes warp borders; registration failure is explicit instead of silently changing algorithms.
- Uses supported Frangi `sigmas` syntax and an accurately labeled result.

## CT

- Consolidates the original circle phantom and external-image experiments, with 1/10/180/360 projection views.
- Retains explicit FFT/filter/IFFT and BP/FBP comparisons.
- Corrects the iterative forward/transpose mismatch using a sparse, pixel-driven system matrix and its exact transpose. This is a changed numerical implementation, not a claim of identical original results.
- Uses the same detector interpolation weights for forward and transpose operations. Removes the extra `pi/(2*n_angles)` scaling of an already scaled inverse-Radon operation.
- Keeps CGLS unconstrained and records residuals after each update. No independently normalized reconstructions are used for RMSE/PSNR/SSIM.
- Distinguishes library Radon/FBP from the educational discrete iterative model.

## MRI

- Explicit sequence selection replaces choosing the first NIfTI file found.
- Sorts slices along the normal derived from ImageOrientationPatient; validates geometry and builds an LPS-to-RAS affine.
- Compares matching canonical grids directly or resamples using physical coordinates; removes SSIM-maximizing flip/rotation searches and proportional slice matching.
- Replaces format “winner” labels with measurements and their limits. Background-noise and sharpness metrics are labeled as proxies.
- Retains three-view visualization, slice montage, native/interpolated viewers and display exports. Adds a controlled same-slice JPEG comparison for the lab's format discussion.
- Requires explicit selection of a time point for 4D inputs; does not call a structural MRI volume an fMRI analysis.

## Verification boundaries

The reorganization and numerical fixes were prepared with coding-assistant support. They extend the original coursework and should not be represented as experiments performed at the original submission date. Dataset permissions and model/application validation are separate from code correctness.
