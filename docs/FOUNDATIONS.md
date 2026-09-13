# Medical Imaging Foundations

This guide explains the concepts used in the three notebooks and connects them to the actual implementation. Read it alongside the [executed notebooks](../README.md#explore-the-three-labs), where the images, tables and numerical outputs are saved.

## 1. Shared concepts

### Pixels, voxels and intensity

A grayscale image is a two-dimensional array of pixel values. A volume is a three-dimensional array of voxels. Array dimensions tell us how many samples exist; spacing and orientation tell us where those samples are in physical space. A 512 × 512 image does not, by itself, specify its resolution in millimeters.

Intensity has different meanings across modalities. In this repository, DSA uses grayscale image values, CT simulates projections from an input image, and MRI uses stored voxel values with applicable scaling. A grayscale BMP does not provide calibrated CT Hounsfield units merely because it depicts a CT slice.

### Processing, reconstruction and visualization

| Operation | Question it answers | Example here |
|---|---|---|
| Image processing | How can an existing image be aligned or enhanced? | DSA registration and subtraction |
| Reconstruction | What image is consistent with projection measurements? | CT FBP, SIRT and CGLS |
| Visualization | How should stored values be displayed? | MRI slice viewers and intensity windows |
| Evaluation | How different is an output from a defined reference? | RMSE, SSIM and matching-volume checks |

These operations should remain distinct. A display window can make an image look clearer without improving the underlying measurement.

### Why numerical scale matters

Unsigned 8-bit values cannot represent negative differences. The DSA code converts images to floating point before subtraction, so a value such as `10 - 250` remains `-240` rather than wrapping around.

For display, an intensity window maps a chosen range to black and white. Quantitative comparisons should use a defined, shared intensity scale. Independently normalizing each reconstruction can hide a gain error: an image multiplied by two may look identical after min-max normalization while remaining numerically incorrect.

## 2. Digital Subtraction Angiography

### Background cancellation

A mask image represents the background before contrast enhancement; the live image contains background plus contrast-related changes. This lab computes a simplified image-domain difference:

$$D(x,y)=L(x,y)-M(x,y).$$

Here, $L$ is the live image and $M$ is the mask. Stationary structures approximately cancel if the images are aligned and comparable in intensity. Motion, noise and exposure differences can leave residual structures. The notebook illustrates this principle with provided grayscale images; it is not a complete calibrated acquisition or logarithmic DSA pipeline.

### Gaussian smoothing and histogram equalization

Gaussian smoothing averages neighboring pixels with distance-dependent weights. It can reduce fine-scale noise, but it also blurs small structures. If the same linear Gaussian operator and boundary handling are used, smoothing before subtraction and smoothing the difference are approximately equivalent:

$$G*(L-M)=(G*L)-(G*M).$$

Histogram equalization redistributes display intensities using their cumulative distribution. Applying it after subtraction changes the visibility of residuals. Equalizing mask and live images separately before subtraction would apply different nonlinear mappings and change their relationship.

### Registration before subtraction

Registration estimates a spatial transformation that brings corresponding structures into alignment. The implemented sequence is:

1. **Normalized cross-correlation (NCC):** compare candidate rotations after centering and scaling the intensity patterns.
2. **Phase correlation:** estimate an initial translation from Fourier-domain information.
3. **Enhanced correlation coefficient (ECC):** refine the chosen motion model; the default is affine.
4. **Resampling:** combine the transformations and sample the original live-image intensities on the reference grid.

An affine transform can represent translation, rotation, scaling and shear. Border pixels created by warping are excluded from alignment statistics because they do not represent common acquired image content. See the [OpenCV registration API](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html).

### Vessel enhancement

| Method | Basic idea | Limitation |
|---|---|---|
| Black-hat morphology | Subtract the original image from its morphological closing to emphasize dark structures smaller than a chosen neighborhood | Depends on neighborhood size and intensity polarity |
| Multiscale black-hat | Combine responses from several neighborhood sizes | Can also enhance nonvascular structures |
| CLAHE | Apply contrast-limited histogram equalization in local regions | Improves display contrast but does not establish segmentation accuracy |
| Frangi filtering | Use second-derivative structure across scales to highlight elongated ridges | Vessel-like appearance is not proof of a blood vessel |

The notebook retains these as exploratory enhancement variants. It has no vessel ground-truth mask, so it does not report segmentation accuracy. Algorithm details are available in the [scikit-image filter reference](https://scikit-image.org/docs/stable/api/skimage.filters.html).

## 3. Computed Tomography

### Projection and sinogram

CT reconstruction estimates an internal image from projections acquired at multiple angles. In an ideal parallel-beam model, a projection value is an integral along a line through the object. Stacking projections at different angles produces a **sinogram**: one axis represents detector position and the other represents angle. The forward Radon transform simulates this process from a known input image. See the [scikit-image Radon tutorial](https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html).

The notebook compares **1, 10, 180 and 360 directions over [0°, 180°)**. These are view counts; 360 views does not mean a 360° acquisition. Sparse angular sampling leaves missing information and can produce streaks or poorly recovered structure.

### Backprojection and filtered backprojection

Unfiltered backprojection spreads each projection value back along its contributing ray paths and accumulates the contributions. This produces a blurred approximation because information is distributed across many pixels.

Filtered backprojection (FBP) filters each projection before backprojecting it. The notebook makes the steps explicit:

```text
Sinogram → FFT along detector axis → frequency filter → inverse FFT → backprojection
```

A ramp filter emphasizes higher spatial frequencies to counteract backprojection blur. Windowed variants—Shepp–Logan, cosine, Hamming and Hann—reduce the high-frequency response, trading sharpness against suppression of high-frequency components. Zero-padding reduces circular-convolution effects in the explicit Fourier demonstration. The manually constructed continuous filter is an approximation; the notebook also uses scikit-image's built-in FBP as a numerical reference.

### The discrete inverse problem

For iterative reconstruction, the lab represents the image by a vector $x$, projections by $b$, and the forward model by a sparse matrix $A$:

$$b=Ax.$$

Each image pixel contributes to neighboring detector bins through linear interpolation weights. The transpose $A^T$ redistributes measurement-domain corrections back to image pixels using those same weights. The test suite verifies the adjoint identity:

$$\langle Ax,y\rangle=\langle x,A^Ty\rangle.$$

This matters because a library backprojection is not automatically the exact transpose of a separately implemented forward projector. The iterative experiment uses an explicit matched pair; its geometry is distinct from the library Radon/FBP experiment.

### SIRT and CGLS

**Simultaneous Iterative Reconstruction Technique (SIRT)** repeatedly projects the current estimate, measures its residual, and distributes a normalized correction back to the image:

$$x_{k+1}=\max\left(0,\;x_k+\lambda C A^T R(b-Ax_k)\right).$$

$R$ and $C$ are diagonal reciprocal row/column-sum weights; $\lambda$ controls the update size. The implementation applies non-negativity after each update.

**Conjugate Gradient Least Squares (CGLS)** uses the forward and transpose operations to iteratively solve:

$$\min_x \|Ax-b\|_2^2.$$

Its search directions follow the least-squares structure. This implementation does not clip intermediate or final CGLS values, because clipping would change the unconstrained algorithm. Both methods record the relative projection residual after each update. Lower residual means greater agreement with the chosen measurements, not necessarily better clinical image quality.

## 4. Magnetic Resonance Imaging

### Physical background and contrast

MRI uses a magnetic field, radiofrequency excitation and spatial encoding to form images from magnetic-resonance signals. This physical background motivates the modality; the notebook begins with **already reconstructed images** and does not simulate pulse sequences or reconstruct raw k-space. An introductory explanation is available from [NIBIB](https://www.nibib.nih.gov/science-education/science-topics/magnetic-resonance-imaging-mri).

- **T1:** characterizes recovery of longitudinal magnetization after excitation.
- **T2:** characterizes decay of transverse coherence.
- **FLAIR:** uses inversion recovery with timing chosen to suppress cerebrospinal-fluid signal.

T1- and T2-weighted contrast also depends on acquisition parameters and tissue properties. A file labeled T1-weighted is not a quantitative T1 map. The checked coursework run uses the paired BRAINIX T1 representations; the presence of a sequence selector does not mean every available sequence was evaluated.

### DICOM and NIfTI

DICOM stores medical-image pixels together with acquisition and spatial metadata. A conventional series often consists of multiple slice files. NIfTI commonly stores a 3D volume or 4D array with a header and a voxel-to-world affine. These are **storage formats**, not imaging sequences or quality grades.

The loader applies DICOM rescale slope/intercept and checks that slices belong to one series with consistent, regular geometry. For a 4D NIfTI, the caller must select a time index explicitly. The provided structural examples do not support a claim of fMRI activation analysis.

### Voxel coordinates and physical coordinates

A voxel index is mapped into physical coordinates by an affine matrix:

$$\begin{bmatrix}x\\y\\z\\1\end{bmatrix}
=T\begin{bmatrix}i\\j\\k\\1\end{bmatrix}.$$

The DICOM loader uses orientation, pixel spacing and slice positions to construct this mapping. It sorts slice positions along their orientation-derived normal, rather than assuming the patient-coordinate z value always describes acquisition order. It converts DICOM LPS coordinates to the RAS convention used by NiBabel. [NiBabel geometry documentation](https://nipy.org/nibabel/dicom/dicom_orientation.html) explains the coordinate mapping.

Canonicalization reorders/flips array axes toward RAS without interpolation. It does not remove acquisition obliquity. If two matching acquisitions have different grids, resampling uses their physical coordinates; resizing slices or matching proportional indices is insufficient. Matching grids are compared directly to avoid introducing interpolation differences. See [NiBabel coordinate systems](https://nipy.org/nibabel/coordinate_systems.html).

### Display interpolation and JPEG export

Increasing the number of slices in a viewer interpolates between existing samples. It makes browsing smoother without creating newly acquired detail. The JPEG experiment compares a compressed image with the **same windowed 8-bit slice**, so geometry and display scaling are held fixed. A JPEG or display PNG does not preserve the original 3D voxel geometry and full intensity representation.

## 5. Reading the metrics

| Metric | Meaning in this project | Interpretation |
|---|---|---|
| MAE | Mean absolute pixel/voxel error against a stated reference | Lower means closer intensities on that scale |
| RMSE | Square root of mean squared error | Gives larger errors more weight; lower is better for fidelity |
| PSNR | Logarithmic ratio based on a declared data range and mean squared error | Higher indicates less error; exact equality produces infinity |
| SSIM | Local comparison of luminance, contrast and structure | Values closer to 1 generally indicate greater structural similarity |
| Relative projection residual | $\|b-Ax\|_2/\|b\|_2$ for nonzero measurements | Measures consistency with a specific forward model |
| Laplacian variance / gradient energy | Variation in second derivatives / strength of spatial gradients | Sharpness descriptors, not measured spatial resolution |
| Background MAD / foreground-to-noise ratio | Heuristic statistics from estimated foreground and background regions | Noise-related proxies, not calibrated MRI SNR |

Interpret scores only after checking the reference, grid, intensity scale and evaluation region. The CT FBP and iterative examples use different sizes/projectors; their scores are not a controlled ranking of reconstruction families. The zero DICOM/NIfTI error in the supplied T1 pair indicates matching stored information after reorientation, not that all conversions are lossless. [Recorded validation results](VALIDATION.md) specify the tested configurations.

## 6. Where to inspect the implementation

| File | Relevant implementation |
|---|---|
| [dsa.py](../imaging_labs/dsa.py) | Signed subtraction, NCC/phase/ECC alignment, overlap mask and enhancement variants |
| [ct.py](../imaging_labs/ct.py) | Sparse projection matrix, SIRT/CGLS and Fourier filter construction |
| [mri.py](../imaging_labs/mri.py) | DICOM/NIfTI loading, affine handling, consistency checks and viewers |
| [common.py](../imaging_labs/common.py) | Input validation, display windows and shared image metrics |
| [test_numerics.py](../tests/test_numerics.py) | Numerical and geometry regression checks |

Coursework foundation: Medical Imaging Labs 1–3, Le Nhat Tan, MEng, HCMUT–VNUHCM. The supplied teaching slides are acknowledged but not redistributed.
