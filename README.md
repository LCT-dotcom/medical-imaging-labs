# Medical Imaging Labs — DSA, CT and MRI

**Lý Chính Tân · Biomedical Engineering · HCMUT, VNU-HCM**

This project studies how medical images are **processed, reconstructed and interpreted in software**. It turns three Medical Imaging coursework labs into reproducible Python notebooks with saved figures, tables and numerical results.

The three labs address different questions: **DSA** asks how to suppress stationary background and compensate for motion; **CT** asks how to reconstruct an image from projections; **MRI** asks how to load, align and compare stored volumes without confusing file format with image quality. The focus is classical image processing and numerical methods, rather than training a diagnostic AI model.

## Explore the three labs

| Lab | Main question | Open the executed notebook |
|---|---|---|
| 01 · DSA | How can subtraction reveal contrast-related changes, and why must images be aligned first? | [Subtraction, registration and vessel enhancement](notebooks/01_dsa_subtraction_registration.ipynb) |
| 02 · CT | How do projection sampling, filtering and iterative algorithms affect reconstruction? | [Projection and reconstruction](notebooks/02_ct_projection_reconstruction.ipynb) |
| 03 · MRI | How can DICOM and NIfTI be compared on the correct physical grid? | [Volume loading, geometry and analysis](notebooks/03_mri_dicom_nifti_analysis.ipynb) |

**The main notebooks contain saved outputs from the supplied coursework data.** Their configuration defaults to synthetic demo inputs so a new reader can run them without those files. Rerunning in demo mode replaces the saved coursework outputs with demo results. The `examples/` folder provides additional executed synthetic examples.

For the theory, equations and metric definitions, read **[Medical Imaging Foundations](docs/FOUNDATIONS.md)**.

## Lab 01 — Digital Subtraction Angiography

### What this lab does

The inputs are a grayscale **mask**, a **live image** and a rotated live image. Subtracting the mask from the live image suppresses structures common to both. When the patient or image moves, those structures no longer coincide and leave subtraction artifacts. The lab examines both the basic operation and the effect of alignment.

```text
Mask + live image → signed float subtraction → contrast display and smoothing
Mask + rotated live → rotation/translation initialization → ECC alignment
                    → subtraction on valid overlap → inspect residuals
```

The notebook then explores black-hat morphology, multiscale enhancement, CLAHE and Frangi filtering to make vessel-like dark structures more visible.

### What to learn from the results

- **Image arithmetic:** floating point preserves negative differences and avoids unsigned-integer underflow.
- **Noise versus detail:** Gaussian smoothing can suppress noise while blurring fine vessels.
- **Image registration:** NCC, phase correlation and ECC estimate alignment before subtraction.
- **Display versus measurement:** contrast enhancement changes visibility; it does not establish vessel segmentation accuracy.

In the recorded coursework run, the estimated correction was about **−30°**, with ECC correlation **0.98396**. The mean absolute mask/live difference over the same valid overlap decreased from **26.2763 to 4.1197** intensity units. This describes alignment behavior; it is not a vessel-detection accuracy score.

## Lab 02 — Computed Tomography

### What this lab does

Starting from CT1.bmp or a synthetic phantom, the notebook simulates projections with the Radon transform. It compares **1, 10, 180 and 360 projection directions over [0°, 180°)**, then examines how the image can be recovered.

```text
Known image → projections / sinogram → unfiltered backprojection
                                    → filtered backprojection
Known image → matched discrete forward model → SIRT / CGLS → residual curves
```

The Fourier-filter section explicitly shows FFT, ramp/windowed filtering and inverse FFT. A separate iterative experiment uses a sparse matrix and its exact transpose so the forward and backward operations are mathematically consistent.

### What to learn from the results

- **Projection and sinogram:** one projection summarizes the object along ray paths; multiple angles constrain reconstruction.
- **Backprojection:** spreading projections back into the image produces blur without filtering.
- **FBP:** ramp and windowed filters trade recovery of detail against suppression of high-frequency components.
- **Inverse problems:** SIRT updates an image from normalized residual corrections; CGLS solves a linear least-squares problem iteratively.
- **Evaluation:** RMSE, PSNR and SSIM use a shared intensity scale, while residual curves describe measurement consistency.

The recorded CT1 ramp-FBP example has **RMSE 0.01988 and SSIM 0.96314** at 128 × 128 pixels and 180 views. SIRT/CGLS use a separate 64 × 64 discrete model, so their scores must be compared within that experiment rather than ranked directly against FBP. CT1.bmp is used to simulate projections; this is not reconstruction from raw scanner measurements or a calibrated Hounsfield-unit experiment.

## Lab 03 — Magnetic Resonance Imaging

### What this lab does

The notebook loads a selected DICOM slice series and its corresponding NIfTI volume, inspects their geometry, and visualizes slices. The verified coursework example uses **BRAINIX T1**.

```text
Matched DICOM + NIfTI → inspect spacing and orientation → canonical reorientation
                     → compare on a common physical grid → tables and difference images
                     → slice viewers → controlled JPEG display comparison
```

DICOM slices are sorted using their spatial orientation and positions. Equivalent grids are compared directly; different grids are resampled using affine metadata. The notebook avoids choosing rotations solely to maximize a similarity score.

### What to learn from the results

- **MRI background:** T1, T2 and FLAIR describe relaxation/contrast mechanisms; the notebook works with already reconstructed images.
- **Medical storage:** DICOM and NIfTI organize pixels/voxels and metadata differently; file extension alone does not determine quality.
- **Spatial geometry:** voxel indices, voxel spacing and physical coordinates must be distinguished.
- **Interpolation:** extra display slices do not create new acquired detail.
- **Image comparison:** JPEG compression is compared with the same windowed 8-bit slice, rather than with an unrelated volume.

The supplied 512 × 512 × 22 T1 representations have **zero MAE/RMSE after correct reorientation**, indicating identical voxel values in this pair. The original array orientations differed. This is a consistency check for these files, not a universal claim about format conversion. The project does not perform raw k-space reconstruction, pulse-sequence simulation or fMRI activation analysis.

## Skills demonstrated

Image arithmetic, filtering and mathematical morphology; spatial registration; Fourier methods; linear algebra and inverse problems; medical-image metadata and coordinate systems; reference-based evaluation; and reproducible notebook organization.

Core tools: **NumPy, SciPy, matplotlib, pandas, OpenCV, scikit-image, pydicom and NiBabel**. [Implementation map and theory](docs/FOUNDATIONS.md#6-where-to-inspect-the-implementation) · [Recorded validation](docs/VALIDATION.md) · [Changes from the original coursework](CHANGES.md).

## Run

Use Python 3.12. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m jupyterlab
```

On macOS/Linux, activate with `source .venv/bin/activate`. Open the notebooks in order and choose **Restart Kernel and Run All Cells**. Each notebook also runs independently. The notebook imports helpers from `imaging_labs/`, so download or clone the entire repository, not just one notebook.

**Demo mode is the default:** deterministic synthetic images allow execution without downloading medical data. Outputs from this mode are not results on the original lab data. Set `MODE = 'local'` in the configuration cell after following [data setup](data/README.md). No package-install commands run inside a notebook.

Optional environment variables:

| Variable | Purpose |
|---|---|
| `IMAGING_LABS_MODE` | `demo` or `local` |
| `IMAGING_LABS_DATA` | Folder containing `dsa/`, `ct/`, `mri/` |
| `IMAGING_LABS_OUTPUT` | Output root; each run writes to a mode-specific folder |
| `IMAGING_LABS_WIDGETS` | `0` disables interactive MRI widgets for batch execution |

## Implementation choices

- DSA: preserves signed image arithmetic, separates enhancement from subtraction, and evaluates registration only within valid overlap.
- CT: makes filter construction visible and uses a matched sparse forward/transpose pair for iterative reconstruction. Quantitative metrics do not hide amplitude error through separate normalization.
- MRI: respects voxel-to-world geometry and explicit sequence selection; avoids ranking storage formats using unregistered images or heuristic sharpness scores.

## Validation

```powershell
python -m pytest tests -q
```

The tests cover subtraction underflow, image-shape validation, constant display windows, intensity-scale errors, the projector adjoint identity, iterative convergence, translation registration, DICOM slice sorting and NIfTI time-point selection. See [validation notes](docs/VALIDATION.md) for the executed configurations and limitations.

## Data and scope

These are educational experiments, not clinical validation. The original lab images, MRI archives and instructor slides are excluded from this repository. They were supplied for local verification; this does not establish redistribution rights. Synthetic examples are generated by code in `imaging_labs/demo.py`.

MRI widgets require a live Jupyter session; GitHub displays saved static notebook content. The supplied BRAINIX examples are structural MRI; this repository does not claim fMRI activation analysis or deployment on a medical device.

## Acknowledgment and references

Based on Medical Imaging Labs 1–3 (DSA, CT and MRI), taught by **Le Nhat Tan, MEng**, Department of Biomedical Engineering, Faculty of Applied Science, HCMUT–VNUHCM. The coursework materials remain with their respective authors.

- [OpenCV ECC registration](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)
- [scikit-image Radon reconstruction example](https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html)
- [NiBabel DICOM geometry](https://nipy.org/nibabel/dicom/dicom_orientation.html)
- [NiBabel coordinate systems](https://nipy.org/nibabel/coordinate_systems.html)
- [pydicom pixel data](https://pydicom.github.io/pydicom/stable/guides/user/working_with_pixel_data.html)

Author profile: [LCT-dotcom](https://github.com/LCT-dotcom).
