# Validation

Checked on September 12, 2026 with Python 3.12 on Windows.

## Executions

All three notebooks completed top-to-bottom in separate fresh IPython processes for each input mode: **six successful executions** with no cell errors. Static plots and tables were captured in executed notebook copies. Interactive widgets were disabled during batch execution; their Python construction is checked separately and full browser interaction is not part of this batch result.

| Notebook | Synthetic demo | Supplied coursework data |
|---|---|---|
| DSA | Passed | Passed: mask.jpg, live.jpg, rotated_live.jpg |
| CT | Passed | Passed: CT1.bmp plus circle phantom |
| MRI | Passed | Passed: matched BRAINIX T1 DICOM and NIfTI |

The test suite reports **16 passed**. It covers signed subtraction, shape validation, display handling, raw-scale metrics, adjoint identity, solver convergence/zero data, registration, DICOM sorting, equivalent axis orders, mixed-series rejection and explicit 4D time selection.

## Observed local-data results

- **DSA:** estimated rotation −30.0°, ECC correlation 0.98396, valid overlap 83.66%. Mean absolute mask/live difference on the same valid area decreased from 26.2763 to 4.1197 intensity units. This is an alignment descriptor, not a vessel-detection score.
- **CT1:** library ramp FBP, 128×128 input and 180 directions: RMSE 0.01988, SSIM 0.96314 on the original [0,1] intensity scale.
- **CT1 iterative example:** 64×64 input, 180 directions and 80 iterations. SIRT RMSE 0.03494, SSIM 0.96875; CGLS RMSE 0.00556, SSIM 0.99770. These two methods share the educational sparse projector; these values are not directly ranked against the separate library FBP experiment.
- **MRI T1:** matching 512×512×22 representations become identical after metadata-based canonical reorientation; MAE/RMSE are zero, overlap 100%. This is a format-consistency result for the supplied pair, not a general claim about DICOM and NIfTI.

## Public examples

`examples/` contains executed notebooks and representative figures using **synthetic demo inputs only**. They can be browsed on GitHub without access to the coursework data. Original medical-image outputs remain local and are not part of the publication archive.

## Environment

The verified environment used NumPy 2.5.3, SciPy 1.18.1, matplotlib 3.11.2, OpenCV 5.0.0, scikit-image 0.26.0, pydicom 3.0.2, NiBabel 5.4.2, nbformat 5.11.1 and IPython 9.17.1. `requirements.txt` specifies compatible ranges; these runs are not a cross-version compatibility guarantee.

## Limitations

The cleaned defaults were run against CT1 and MRI T1; the other supplied CT images and MRI sequences were not claimed as completed experiments. The explicit CT system is a teaching discretization, not a scanner model. MRI supports conventional single-frame, regular MONOCHROME2 series and does not replace a general DICOM converter or motion registration package.
