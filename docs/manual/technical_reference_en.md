<div class="title-page" markdown="1">

# BICTool Technical Reference

<p class="sub">Algorithms, parameters and limits of the head-CT image-quality assessment<br>
Version {{VERSION}}</p>

<p class="sub">Developed for the German Federal Office for Radiation Protection (BfS)</p>

<div class="note" markdown="1">
**For medical physicists, QA staff and developers.** This reference explains how BICTool computes every
result. How to use the program is described in the *User Manual*. BICTool is decision support only; it is not
a medical device.
</div>

</div>

<div class="contents" markdown="1">

[TOC]

</div>

## 1. About this reference {#tr-intro}

Each chapter describes one processing step in the same order:

| Part | Content |
|---|---|
| Purpose | What the step is for and where its result is used |
| Method | The algorithm, step by step, with formulas |
| Parameters | The values BICTool uses; *provisional* marks values chosen on the development data that still need confirmation |
| Limits | What the step cannot do, and how that shows in the results |

**Conventions.** CT numbers are in Hounsfield units (HU). Spatial frequencies are in cycles per millimetre
(mm⁻¹). *px* is the in-plane pixel spacing in mm. "Slice" always means one axial image of the selected series.
Image coordinates are (row, column); on standard axial DICOM images the rows run from anterior to posterior.

**Provisional values.** BICTool was developed on a small set of head-CT studies from three scanner models,
rated by one radiologist (study verdict) and by the project team (descriptive findings). Every threshold
that was chosen on these data is marked *provisional* in this reference and in the program. Such values will
be revised when more rated data are available.

## 2. Processing pipeline {#tr-pipeline}

| Step | What happens | Result | Chapter |
|---|---|---|---|
| 1. Input | DICOM series found and ranked; pixel values converted to HU | CT volume | 3 |
| 2. Anatomy | AI segmentation of 16 brain structures, slice by slice | label masks per slice | 4 |
| 3. Brain mask | Union of the labels, largest 3D region | brain outline per slice | 5 |
| 4. Tissue | White matter, grey matter and CSF separated by CT number | tissue masks per slice | 6 |
| 5. ROIs | Falx and septum ROI pairs placed on suitable slices | ROI1/ROI2 per slice and pair | 7 |
| 6. Measurements | MTF on ROI1; power spectra of ROI1 and ROI2; derived features | feature vector per slice and pair | 8 |
| 7. Verdict | Study medians → logistic model with bootstrap interval; metal rule | study verdict with reasons | 9 |
| 8. Findings | Metal, brainstem band, basal-ganglia visibility, segmentation check | descriptive hints | 4.5, 10 |

Steps 2-4 run once per series and are cached. Step 5 runs on request (*Start auto ROIs*), step 6 on request
(*Run QA*), steps 7-8 after every QA run.

## 3. Input and series selection {#tr-input}

**Purpose.** Load the series the measurement is meant for: the axial reading series of a routine head CT.

**Method.**

1. All files below the chosen folder are read as DICOM headers; files that are clearly not DICOM are skipped.
   Files from more than one study are rejected.
2. The images are grouped into series (series UID, then coherent stacks within a series).
3. The series are ranked, best first:
    1. modality CT;
    2. axial stacking direction: the main axis of all slice positions (`ImagePositionPatient`) lies within
       60° of the patient's z-axis. The positions are used rather than the orientation tag, because some
       scanners write the axial orientation into their coronal and sagittal reformats. Tilted gantries are
       accepted;
    3. slice thickness closest to 4 mm (the standard reading series);
    4. most slices.
4. Pixel values are converted with `RescaleSlope` and `RescaleIntercept` to HU. Decimal values in the header
   are read with either a point or a comma as decimal separator (some scanners write commas).
5. Slices are sorted by their position along the stack direction.

Any matrix size and pixel spacing is accepted; BICTool never assumes 512 × 512 pixels.

## 4. Anatomical segmentation {#tr-segmentation}

### 4.1 Network {#tr-cnn}

**Purpose.** Find the anatomy that the ROIs and the findings are built on.

BICTool uses the head-CT segmentation network of the Mayo Clinic Radiology Informatics Lab (Cai et al. 2020),
a 2D U-Net with 17 output classes (background and 16 structures), with the published weights
*weights_mayoiNPH* (the set its authors recommend for routine use). The network was trained on adult head CT
of normal brains and of patients with normal-pressure hydrocephalus and Alzheimer's disease.

| Labels | Group |
|---|---|
| Brain parenchyma, frontal, parietal, occipital and temporal lobe, cerebellum, insular cortex, central sulcus | brain tissue |
| Caudate nucleus, lentiform nucleus, thalamus, internal capsule | deep grey and white matter |
| Ventricular system, subarachnoid space | CSF spaces |
| Dural folds (falx and tentorium), septum pellucidum | thin structures used for sharpness |

### 4.2 Pre- and post-processing {#tr-cnn-processing}

1. **Orientation.** The network expects the orientation of its training data (NIfTI volumes). The series is
   converted to NIfTI once, and the in-plane transform between a DICOM slice and its NIfTI counterpart is
   found by trying all 8 combinations of transpose and flips. Several reference slices are tried, ordered by
   image variance, so that a blank or artefact slice cannot decide; if no candidate matches, the
   segmentation stops with an error instead of guessing. The conversion does not resample or reorient the
   data.
2. **Size.** Each slice is padded or cropped (centred, per axis) to 512 × 512 pixels. It is never
   interpolated, so the geometry stays undistorted.
3. **Intensity.** The network gets plain HU values, as in its training.
4. **Output.** The class with the highest softmax probability per pixel; the result is cropped or padded
   back to the slice size and mapped back to the DICOM orientation. All measurements work on the original
   DICOM pixel grid.
5. **Cache.** The labels (and the softmax confidence, which is recorded but not used) are cached per series,
   so a study is segmented only once.

Speed: about 1-2 s per slice on a current office PC (CPU only).

### 4.3 Limits {#tr-cnn-limits}

- Trained on adult brains. Large lesions, post-operative changes, strong artefacts and children's heads can
  be segmented wrongly; the segmentation check (4.5) points to such studies.
- The network fills in structures from shape and position. A label is therefore no proof that the structure
  is visible in the image (this matters for the basal ganglia, 10.3).
- It has no separate falx class: the falx and the tentorium share the label *dural folds* (handled in 7.2).

### 4.4 Brain area per slice {#tr-brain-area}

The brain area of a slice is the number of pixels with any of the 16 labels. It is used for the slice
eligibility rule (7.1) and the segmentation check (4.5).

### 4.5 Segmentation check {#tr-seg-check}

**Purpose.** Tell the user, once per study, whether the segmentation should be looked at. BICTool is not
diagnostic, so the check gives no cause and no per-slice reason.

**Method (label fragmentation).**

1. Per slice and label, the connected pieces of at least 20 mm² are counted, *apart from the largest piece*.
   Smaller specks are ignored; the limit is in mm², so it does not depend on the pixel size.
2. Slice value = sum over all labels.
3. Study value = median over the slices with enough brain (≥ 20 % of the largest brain slice, as in 7.1).
4. Above **30** (provisional) the study is reported as *"Segmentation may be unreliable - check the
   overlays"*; otherwise *"Segmentation looks consistent"*.

Also reported (details, exports): the three most fragmented slices, and the number of "lobe islands" (a
piece of one lobe enclosed to ≥ 90 % by another lobe).

**Rationale.** A clean segmentation consists of a few large regions per slice. When the network struggles
(lesions, artefacts, noise) the lobes break into a patchwork of labels.

**Limits.** A segmentation that is wrong but compact (for example a large cavity labelled as one brain
region) is not detected. Scanners and kernels differ somewhat in their typical fragmentation. Label sizes and
the network's softmax confidence were examined as alternatives and rejected: label sizes follow age and
atrophy, and the network is often confidently wrong.

## 5. Brain mask {#tr-brain-mask}

**Purpose.** The brain outline that the tissue separation (6) works inside.

**Method.** With a segmentation: the union of all 16 labels per slice, then only the **largest connected 3D
region** of the whole volume (face connectivity). A brain is one connected object in 3D, even where a single
slice looks fragmented (craniotomy, artefact); isolated mislabelled patches outside the head are removed.

**Fallback** (slice not segmented yet): a HU skull strip:

1. keep pixels with −500 HU < value ≤ 100 HU (no air, no bone);
2. morphological opening (3 × 3) to cut thin bridges to the scalp;
3. keep the largest connected 2D component;
4. fill holes; closing (5 × 5).

The HU fallback fails on post-operative heads with air or fluid between brain and skull, which is why the
segmentation-based mask is preferred.

**Why a mask at all.** A raw whole-slice HU histogram of the head is nearly unimodal and dominated by skull,
scalp and background. Tissue classes can only be separated by CT number inside the brain.

## 6. White and grey matter separation {#tr-wmgm}

**Purpose.** Find homogeneous white matter for the noise ROI (ROI2) next to the falx (7.3). The separation
does not enter any measurement directly - it decides *where* ROI2 is placed.

All methods assume three tissue classes inside the brain mask, ordered by CT number: CSF and other low-density
tissue < white matter < grey matter (typical values about 10, 30 and 40 HU). The CT-number difference between
white and grey matter is only about 6-10 HU, close to the image noise, so no method works best on every scan.
The user chooses the method (box *WM/GM separation*), and the choice can move ROI2 and therefore change the
noise measurement.

### 6.1 Study tissue references {#tr-tissue-references}

The absolute CT numbers of brain tissue differ by up to about 17 HU between scanners and protocols. BICTool
therefore measures reference values in each study from the segmentation:

| Reference | Region | Rule |
|---|---|---|
| Grey matter | caudate and lentiform nucleus | eroded 1 px, −10 to 80 HU; only on slices where the basal-ganglia labels are plausible (10.3) |
| White matter | internal capsule | same slices, eroded 1 px, −10 to 80 HU |
| CSF | ventricles | eroded 2 px, −10 to 30 HU |

Each region needs at least 40 pixels. Each reference is the median of the per-region means over all slices.
Deep grey matter has practically the same CT number as cortical grey matter (literature: 1.5-3 HU denser).

From the references BICTool derives the study's **plausible brain-tissue range**:
lower limit = (CSF + white) / 2, upper limit = grey + 15 HU (above that: blood, calcification).

### 6.2 Adaptive thresholding (default) {#tr-adaptive}

1. Gaussian smoothing with σ = 1 mm (converted to pixels).
2. Pixels inside the brain mask and inside the plausible range are tissue; pixels below the lower limit are
   CSF.
3. **Local threshold:** the masked local mean HU over a 31 × 31 px window,
   *m(x) = Σ(HU·mask) / Σ(mask)* within the window, so pixels outside the brain never pull the mean towards
   skull or air values.
4. Tissue pixels below the local mean are white matter, the others grey matter.
5. Clean-up (6.6).

The plausible range is the study range (6.1) when it is available and the option is on; otherwise the fixed
range 15-50 HU. The local threshold follows slow intensity changes across the brain (beam hardening, the
cupping near the skull) that a single global threshold cannot.

### 6.3 Fuzzy C-Means {#tr-fcm}

1. The HU values of all brain-mask pixels are clustered into three classes with fuzzy c-means (Bezdek et
   al. 1984): fuzziness m = 2, 50 iterations, start centres at the 15/50/85 % quantiles. Each pixel gets a
   membership *u_k* in every class, *u_k ∝ |HU − c_k|^(−2/(m−1))*.
2. Spatial regularisation: each membership map is blended with its 3 × 3 neighbourhood mean,
   *u' = 0.7 u + 0.3 ū* (Chuang et al. 2006).
3. Each pixel takes the class with the largest blended membership. Clusters sorted by centre: CSF, white
   matter, grey matter.
4. Clean-up (6.6).

FCM adapts its class centres to every slice, but it always finds three classes, even on a slice that holds
mostly one tissue.

### 6.4 HU thresholding {#tr-hu-threshold}

Fixed bands on a 3 × 3 median-filtered slice: CSF 0-20 HU, white matter 20-32 HU, grey matter 32-45 HU, then
clean-up (6.6). Simple and transparent, but it fails when a scanner's CT numbers are shifted.

### 6.5 Anchored (prototype) {#tr-anchored}

Not validated; offered for comparison.

1. References from 6.1.
2. **Drift model.** Within a scan the tissue baseline drifts by about 5 HU with the distance *d* to the brain
   surface and with the slice level *z* - nearly the whole grey/white contrast. The drift is modelled as
   *D(d, z) = a₀ + a₁ e^(−d/3 mm) + a₂ e^(−d/10 mm) + a₃ z + a₄ z² + a₅ z e^(−d/10 mm)*.
3. The model is fitted by least squares to (smoothed HU − reference value of the pixel's current class) on up
   to 3000 brain pixels per slice of the whole series, in 5 rounds of classify → fit → reclassify, and pinned
   to zero drift where the references were measured.
4. Each slice is drift-corrected, smoothed (σ = 1 mm) and classified with thresholds midway between the
   references.

Known weakness: drift and tissue class are confounded; large areas (for example one hemisphere on high
slices) can come out as grey matter. There is no left-right or front-back drift term.

### 6.6 Clean-up (all methods) {#tr-tissue-cleanup}

Per class mask: opening (3 × 3), closing (5 × 5), then removal of connected regions smaller than 20 px. This
turns the speckle caused by noise near the class boundary into coherent regions.

## 7. Automatic ROI placement {#tr-rois}

Two ROI pairs are placed on every eligible slice:

| Pair | ROI1 (structured, sharpness) | ROI2 (homogeneous, noise) |
|---|---|---|
| Falx | on the falx cerebri (label *dural folds*) | white matter next to it |
| Septum | on the septum pellucidum | the ventricle next to it |

Both ROIs of a pair are squares of the same size, so their spectra have the same frequency bins.

### 7.1 Slice eligibility {#tr-eligibility}

Automatic ROIs are placed only on slices whose brain area (4.4) is at least **20 %** of the study's largest
brain slice (provisional). Manual ROIs are allowed on every slice.

**Rationale.** At the skull base the brain cross-section is small and lies between the petrous bones, whose
beam-hardening streaks raise the noise; there is usually no visible falx. At the vertex the slice cuts the
skull tangentially and holds little white matter, so ROI2 would land on sulci and cortex. The segmentation is
also least reliable on small cross-sections. On the development data, falx slices below 20 % had ROI2 noise
above twice the study median more than twice as often as the other slices; slices between 20 % and 40 % were
not noisier than the rest. Verdicts did not change for thresholds between 10 % and 30 %.

### 7.2 ROI1 on the falx or septum {#tr-roi1}

**Target size.** The MTF needs edge profiles of ±2.5 mm on both sides of the edge (8.1), so ROI1 grows from
18 px in 2 px steps until it holds them: *size ≥ ⌈2 × 5 mm / px⌉ + 2*, at most 48 px. At 0.4 mm pixels this
is 28 px ≈ 11 mm.

**Candidates.** For every connected piece of the structure label, up to 12 centres spread along its main axis
and the pixel nearest its centroid are tried - never a single heuristic's top pick.

**Growth.** Around each centre a square grows from 18 px in 2 px steps up to the target size. Every size must
lie inside the image and inside the brain, and may contain at most 2 % pixels of labels that do not belong
next to the structure. Allowed are, for the septum: septum and ventricles; for the falx: dural folds,
subarachnoid space, brain parenchyma, the lobes and the cerebellum.

**Line score.** The structure pixels in the square must form one long straight line, because the MTF
measures across an edge. Score in [0, 1]:

*line score = straightness × single-line fraction × span / pieces*

- straightness = 1 − minor/major spread (principal component analysis of the pixel positions);
- single-line fraction = share of 1 px cross-sections perpendicular to the line that cross exactly one run of
  pixels (a V- or Y-junction crosses two);
- span = length of the line across the square / square size (at most 1);
- pieces = number of separate pieces in the square.

Candidates below 0.5 are rejected (blobs and fragments at the skull base score 0.1-0.5, clean lines 0.6-0.9).

**Falx or tentorium.** The label *dural folds* contains both. A candidate counts as falx when its line scores
≥ 0.6, lies within 8 mm of the head's midline and runs within 15° of it. The midline direction is estimated
once per volume from the falx itself: every slice whose dural-fold pixels form one clearly elongated shape
(major/minor spread ≥ 4, ≥ 20 px) within 30° of the image vertical votes with its direction, weighted by
pixel count. This follows head tilt; the brain outline is not used, because its long axis turns at the skull
base and at the vertex.

**Visibility in the image.** The label can continue where the real line has faded. For each 1 px
cross-section of the line, the brightest structure pixel (smoothed, σ = 1 px) must exceed the median of the
tissue within 6 px beside it by at least 5 HU. The visible share of the line is a ranking criterion.

**Choice.** The best candidate is chosen in this order of priority:

1. falx rather than tentorium (dural folds only);
2. reached the target size;
3. visible share of the line (in steps of 0.1);
4. line score (in steps of 0.05);
5. larger size;
6. more ROI2-friendly space (ventricle, parenchyma) in a band one ROI-width wide around the square;
7. closer to the centroid of its structure piece.

### 7.3 ROI2 in homogeneous tissue {#tr-roi2}

ROI2 has the size of ROI1; it shrinks in 2 px steps down to 18 px only if nothing that size fits.

**Septum → ventricle.** Squares inside the ventricle label (≥ 98 % of pixels), without any septum pixel, with
their centre within 25 mm of ROI1. A square that does not touch ROI1 is preferred to a larger one that
touches it. Among valid squares the one closest to ROI1 wins. If the nearby ventricle cannot hold even an
18 px square, the white-matter search below is used.

**Falx → white matter.** Squares that

- consist of ≥ 95 % white or grey matter (from the selected separation method, 6),
- contain no pixel of the dural folds or the septum (separation methods label the dense falx as grey matter),
- do not touch ROI1 (at least 1 px gap).

Among these, the square with the lowest cost wins:
*cost = centre distance to ROI1 (mm) + 40 mm × grey-matter fraction*.
Ten per cent grey matter therefore counts as much as 4 mm more distance: pure white matter is preferred, but
not at any distance from the falx.

**Fallback.** If no square qualifies, the square near ROI1 with the fewest pixels outside white/grey matter
and outside the brain is used and marked *fallback*.

All positions of one size are scored at once with integral images, so the search is exhaustive and fast.

### 7.4 Manual ROIs {#tr-manual-rois}

Moved, drawn and deleted ROIs override the automatic ones on that slice; *Restore automatic* removes the
overrides. The measurement treats manual and automatic ROIs alike. For a manual ROI1 on a segmented slice the
structure label is still used for the MTF seed check (8.1).

## 8. Image-quality measurements {#tr-measurements}

BICTool follows the measurement design of Passand (2022), which was developed for thorax CT: sharpness at
an anatomical edge (MTF), noise in homogeneous tissue (NPS), and the power spectrum of the structured region
(PS) relative to the sharpness. The *physical recipe* used by default keeps every measurement step of that
method and corrects how the results are placed on the frequency axis (chapter 11 explains the differences).

### 8.1 Sharpness: MTF at the structure edge {#tr-mtf}

**Purpose.** How sharply the scanner reproduces the edge of the falx or septum.

**Method.**

1. **Upsampling.** ROI1 is interpolated 4× (two linear refinement passes), giving a sample spacing of px/4.
2. **Edge detection** (on an enhanced copy only: contrast stretch, Gaussian smoothing, sharpening). A
   Chan-Vese active contour (Chan & Vese 2001; region-based, 1000 iterations, smoothness weight α = 0.2,
   step size 0.45 / max|∂φ/∂t|) grows from a small seed box. The edge pixels are taken from the boundary of
   the grown region (Kirsch operator); the edge direction from the Sobel gradient.
3. **Edge profiles (ESF).** Across every edge pixel, 15 samples are taken along the gradient direction on the
   *unfiltered* upsampled image, over a total length of about 5 mm. The real sample spacing is
   *Δx = 2·round(5 mm / (px/4))·(px/4) / 14 ≈ 0.36 mm*.
4. **Profile conditioning.** Each ESF is upsampled 4× (cubic spline), its mean is removed, it is low-pass
   filtered (3rd-order Butterworth, zero phase) and checked for **bumps**: the line-spread function (LSF,
   derivative of the ESF) must have its peak in the middle 40-60 % of the profile, and its nearest zero
   crossings on both sides must be closer than 35 % of the profile length to the peak. Profiles failing this
   are discarded - they cross a thin line (a bump with two edges), not a single edge. Outside the zero
   crossings the ESF is set constant.
5. **MTF.** The ESF is padded with its end values to three times its length, LSF = first difference,
   *MTF(f) = |DFT(LSF)|(f) / |DFT(LSF)|(0)*, on the frequency axis *f_k = k / (n·Δx)*, kept up to the
   profiles' Nyquist frequency 1/(2Δx) ≈ 1.4 mm⁻¹.
6. **MTF area** = ∫₀^0.8 MTF(f) df per edge (trapezoidal rule on a 0.01 mm⁻¹ grid), averaged over all edges.
   Unit mm⁻¹; the maximum for a perfect system is 0.8 (MTF = 1 across the band). Edges with a non-finite MTF
   are dropped; if more than half are, the slice fails.

**Seeds.** The active contour must start *inside* the structure: a seed across the edge collapses, a seed in
flat tissue leaks or dies. Seeds are tried in this order until one gives usable edges:

1. up to 5 of the brightest pixels (smoothed, σ = 1 px) in the central area of ROI1 (Chebyshev radius 25 % of
   the ROI size), on the structure label when it is known, at least 3 px apart;
2. the original method's fixed seed at a quarter of the ROI size, as a last resort.

When the structure label is known, a seed's result counts only if **at least 30 %** of the region it grew lies
on the structure (label widened by 1 px, compared on the 4× grid). Otherwise a fallback seed beside a thin
falx can grow a region in the tissue next to it and measure that edge instead. If no seed qualifies, the
slice fails with *"MTF edge not on the structure"* and is left out of the verdict.

**What the MTF describes.** It is an *object* MTF: the structure's own thickness and partial-volume profile
are part of it. It describes "sharpness at this structure", not the system MTF of the scanner (compare the
task transfer function of AAPM TG-233 and the air-skin edge method of Sanders et al. 2016).

### 8.2 Power spectra: NPS and PS {#tr-nps}

The same estimator is applied to both ROIs: on ROI2 it gives the **noise power spectrum (NPS)**, on ROI1 the
**power spectrum (PS)** of structure plus noise.

For an N × N ROI *I* with pixel spacing px:

1. mean removal: *I₀ = I − mean(I)*;
2. radial Hann window *w(r) = ½(1 + cos(π r / (N/2)))* for r ≤ N/2, else 0;
3. *NPS₂D(u, v) = |DFT(I₀·w)|² · px² / (N² · mean(w²))* (unit HU²·mm²), on the exact DFT grid
   *u, v = k / (N·px)*;
4. **variance** *σ²_NPS = Σ NPS₂D · (1/(N·px))²* - the integral of the 2D spectrum, equal to the pixel
   variance for white noise;
5. **radial profile**: annuli one frequency bin (1/(N·px)) wide around the DC position, up to the Nyquist
   frequency 1/(2px).

No detrending is applied (the automatic ROI2 placement keeps anatomy out of ROI2). ROIs are about 11 mm wide,
so the first non-zero frequency bin is about 0.09 mm⁻¹ on every scanner.

### 8.3 Noise texture {#tr-texture}

From the radial NPS of ROI2 over 0-0.8 mm⁻¹ (AAPM TG-233):

- average frequency *f_av = ∫ f·NPS(f) df / ∫ NPS(f) df*;
- peak frequency *f_peak = argmax NPS(f)*.

A low f_av means coarse, blotchy noise (smooth kernels, strong iterative reconstruction); a high f_av means
fine-grained noise.

### 8.4 PS/MTF ratio {#tr-ratio}

The PS of ROI1 contains structure and noise; dividing by the MTF shows how much of it the scanner's sharpness
explains. Both curves are normalised at the lower band limit and compared on a fixed band
**0.1-0.8 mm⁻¹** in steps of 0.01 mm⁻¹:

*R(f) = [PS(f) / PS(f_lo)] / [MTF(f) / MTF(f_lo)]*

Features:

- **LF/HF** = ∫R over the lower half of the band ÷ ∫R over the upper half (split at 0.45 mm⁻¹);
- **slope** = mean dR/df (mm);
- **max location** = the frequency of the maximum of R (mm⁻¹).

On head CT, slope and max location carry almost no information: ROI1's PS is dominated by the bright line
and falls monotonically, so R drops from 1 to about 0 on almost every slice. The slope then equals
(R(f_hi) − R(f_lo)) / (f_hi − f_lo) ≈ −1/0.7 by construction, and the maximum sits at the band start. They are
reported for completeness but not used by the verdict.

### 8.5 Variance and contrast {#tr-contrast}

- **ROI1 variance** (*var_struct*): the sample variance of the ROI1 pixels (HU²).
- **RMS contrast** of each ROI: *C = √mean(((I − Ī)/Ī)²)*, i.e. standard deviation divided by the mean. No
  unit. Displayed only; not used by the verdict.

### 8.6 The feature band 0.1-0.8 mm⁻¹ {#tr-band}

No standard prescribes a band for patient-based spectral features, so the band is a design parameter. Its
limits follow from physical constraints:

- **Upper limit 0.8 mm⁻¹.**
    1. It is the Nyquist frequency of 0.625 mm pixels. Routine head CT is reconstructed at 0.39-0.49 mm, so
       every routine scan contains the whole band (sampling theorem).
    2. A standard adult head reconstruction carries hardly any signal or noise above it: Friedman et al.
       (2013) measured MTF = 0.1 at 0.62 mm⁻¹ and an NPS that falls to zero by 0.8 mm⁻¹. Above the band the
       ratio would divide by an MTF close to zero. Bhattarai et al. (2024) found the same for photon-counting
       CT with soft kernels.
    3. The edge profiles' own Nyquist frequency (≈ 1.4 mm⁻¹) is well above it.
- **Lower limit 0.1 mm⁻¹** - the first non-zero frequency bin of an 11 mm ROI.

Upper limits between 0.7 and 0.9 mm⁻¹ change the absolute values but not the ranking of slices and studies
(rank correlations ≥ 0.97 for MTF area and LF/HF). Values computed with different bands must never be mixed.

**Scope.** The band is meant for soft-tissue (brain) kernels. Sharp head kernels, bone kernels and
ultra-high-resolution reconstructions transfer real contrast above 0.8 mm⁻¹, which the features do not see.
For pixel spacings above 0.625 mm the band is shortened to what the spectra cover and the result is marked
*band truncated*.

### 8.7 Per-slice result {#tr-slice-result}

Per slice and ROI pair BICTool stores: MTF curve and area, PS and NPS curves and variances, ROI1 variance,
noise texture, PS/MTF ratio and its features, RMS contrasts, the seed used and, in the program, the
intermediate steps (*Results > Per slice > Show: Intermediate steps*). A slice whose MTF cannot be measured is
marked *failed* with the reason.

## 9. Study verdict {#tr-verdict}

### 9.1 Design {#tr-verdict-design}

Single slices turned out to be too noisy to classify on head CT: a slice-level classifier was at chance level.
Whole studies, described by the median of each feature over the falx slices, do carry the signal - mainly the
image noise around the falx, and it holds within each scanner. The verdict therefore judges the **study**:

| Element | Choice |
|---|---|
| Input | median over the study's falx slices of five features: MTF area, ROI2 NPS variance (*noise in homogeneous tissue*), ROI1 PS variance (*noise at the falx*), ROI1 variance, PS/MTF LF/HF |
| Model | logistic regression on standardised medians, L2 penalty (C = 1), class-balanced |
| Output | probability that a radiologist rates the study *acceptable*, with a 95 % interval |
| Decision | interval entirely above 0.5 → **Acceptable**; entirely below → **Not acceptable**; otherwise **Uncertain** |
| Minimum data | at least 5 usable falx slices, otherwise *not available* |

The septum pair does not enter the verdict (it carried no quality signal) but is used as a plausibility check
(9.5). Dose is deliberately not an input: it would act as a scanner label, and the project relates image
quality *to* dose, which would become circular.

### 9.2 Probability and interval {#tr-verdict-interval}

*P = 1 / (1 + exp(−(b₀ + Σ b_i z_i)))*, with *z_i* the standardised study medians.

The 95 % interval covers two sources of uncertainty at once (paired bootstrap, Efron & Tibshirani 1993):

- **model uncertainty**: the model was refitted 1000 times on resamples of the rated training studies, drawn
  with replacement within each class;
- **measurement uncertainty**: for each refit *b*, the study's falx slices are resampled with replacement and
  the medians recomputed.

Refit *b* is evaluated on slice resample *b*; the interval is the 2.5-97.5 percentile of the 1000
probabilities. The random seed is fixed, so the same study always gives the same result.

With the current, small training set most intervals are wide and *Uncertain* is a frequent and honest answer.

### 9.3 Reasons {#tr-verdict-reasons}

For every feature BICTool reports the study median with its 95 % interval (from the slice resamples) and the
range of the rated-good training studies. A feature is listed as a **reason** when its whole interval lies
outside the good range *and* it pushes the decision towards *not acceptable*; reasons are ordered by their
contribution *b_i (m_i − mean_i)* to the model.

### 9.4 Artefact indicator and outlier slices {#tr-artefact-indicator}

- **Noise consistency** (provisional): the scatter of ROI2 noise across the falx slices,
  *IQR / median*, with a bootstrap interval. Studies with motion or implant artefacts showed noise that
  jumps between slices, while plain noisy studies are noisy throughout. Reported as *above the good range*
  (the value exceeds the highest value of the good training studies) and *confirmed* (the whole interval
  does). Needs ≥ 10 slices; no probability is given.
- **Unusually noisy slices**: ROI2 noise above 3 × the study median. Descriptive.

### 9.5 Falx/septum plausibility {#tr-plausibility}

Scan properties should agree at both sites. The differences septum − falx of the medians of MTF area and
noise texture f_av are compared with the range seen in training. A difference outside that range suggests
that one of the ROI sets is wrong (*"Check the ROIs"*).

### 9.6 Metal rule {#tr-metal-rule}

Metal streaks disturb the measurements themselves: ROI2 avoids them, so a study with heavy streaks can look
better than it is. Therefore, after the model:

> If metal streaks reach the brain on any slice within the verdict's slice range (first to last falx slice),
> the verdict is **Not assessable - metal artefacts in the brain**, without a score.

Metal artefacts do not make a study unacceptable as such - with implants CT is often the only option, and
whether the images answer the clinical question is the reader's decision. The model's own result is kept and
shown in the details and exports. Streak detection: 10.1.

### 9.7 Indicative slice score {#tr-slice-score}

The *Per slice* view and the slice table show the model's probability for a single slice (the slice's values
instead of the study medians). It is labelled *indicative*: the model is fitted on study medians, and on
single slices it is not reliable.

### 9.8 Reliability {#tr-verdict-reliability}

The model (*head-ct-1*) is **provisional**. It was checked by patient-wise cross-validation (the whole
procedure, including the bootstrap, refitted without the held-out study) and once on locked test studies.
In both checks no confident verdict was wrong; the cross-validation was, as expected, more optimistic than
the test. The ratings come from one radiologist and are partly confounded with the scanner. A model trained on
more rated data, with a fresh test set, is planned. Details are in the model card (`docs/qa_method/` in the
source code).

## 10. Descriptive findings {#tr-findings}

These findings point the user to slices worth a look. They are provisional and - except metal streaks in the
brain (9.6) - do not change the verdict.

### 10.1 Metal and metal streaks {#tr-metal}

**Metal slices.** A slice holds metal when at least 3 connected pixels exceed **2800 HU**. Dense cortical bone
stays below about 2400 HU; metal reaches the scanner's ceiling (3071 HU or more).

**Streaks in the brain** (per metal slice with segmentation):

1. Brain tissue = labels without ventricles, subarachnoid space, septum and dural folds, eroded 2 mm (away
   from the bone edge).
2. Take the tissue within **30 mm** of the metal.
3. Share of it that differs by more than **30 HU** from the slice's brain-tissue median.
4. From **5 %** (provisional) the streaks reach the brain. With less than 20 mm² of tissue near the metal the
   slice is not judged.

Far from metal (> 60 mm) the same share is about 0-0.1 %. Small craniotomy screws without streaks give 0-0.2 %.

**Limits.** Few studies with metal were available. A large haemorrhage next to a clip can differ by more than
30 HU and be counted. Metal-artefact reduction lowers the share, as intended: only visible streaks count.

### 10.2 Dark band across the brainstem {#tr-brainstem}

**Purpose.** Beam hardening between the petrous bones shows as a dark band running left-right across the pons
(Barrett & Keat 2004); radiologists judge artefacts in this region.

**Method.**

1. Posterior-fossa slices: the cerebellum is ≥ 15 % of the labelled brain, in the skull-base half of the scan.
2. Brainstem = the *brain parenchyma* piece that touches the cerebellum (within 3 mm), at least 3 mm wide.
3. Per image row (anterior → posterior): mean HU of the brainstem, eroded 1 mm, 0-70 HU. Rows narrower than
   40 % of the widest row and the outer 3 mm at the front and back are left out (partial volume with the
   cisterns and the 4th ventricle); at least 10 rows are needed.
4. **Band depth** = the largest amount by which a row is darker than the 4 mm-smoothed row profile (HU).
5. Flag from **3.5 HU** (provisional).

**Evidence.** Against blinded slice ratings by a non-clinician, the band depth separated slices with and
without visible streaks (AUC about 0.75-0.80 after matching the slice level); the same measure taken
front-to-back did not (AUC ≈ 0.5), so it responds specifically to left-right bands. Brainstem noise, Radon
directionality, texture SD and a dark-line detector were tested and did not work.

### 10.3 Basal-ganglia visibility {#tr-basal-ganglia}

**Purpose.** The basal ganglia are a low-contrast region that matters clinically (an early infarct fades
their border). The CNN label only says *where* to look: it labels the nuclei from shape and position even
when their contrast is removed from the image. Visibility is therefore measured in the image.

**Anatomical block.** Deep grey matter per slice = lentiform + caudate + thalamus + internal capsule. Starting
at the slice with the most deep grey matter, the block extends up and down while the next slice keeps at
least 8 % of that area (provisional). Labels outside the block (for example near the vertex) are ignored.

**Measure per slice** (within the block):

1. **Border contrast** of each nucleus = (mean HU in a 2 px ring inside the label border − mean HU in a 2 px
   ring outside; 0-60 HU only) / pooled SD of the nucleus and the internal capsule (both eroded 1 px).
   Slice value = mean over the labelled lentiform and caudate nuclei.
2. **Plausibility** (provisional): nucleus − internal capsule must be 2-15 HU; otherwise the label lies on
   something else (metal streaks, blood) and the slice counts as *label not plausible*.
3. **Words** (provisional): < 0.25 not visible, 0.25-0.6 hardly visible, 0.6-0.85 visible, ≥ 0.85 clearly
   visible.

**Evidence and limits.** Against blinded ratings by a non-clinician, the border contrast agreed better than
the label area, the HU difference and a CNR (Spearman ≈ 0.6-0.7). Per study the agreement is weak, because
pathology dominates; no study score is given. Next to the ventricles the outer ring can contain CSF.

## 11. Legacy (Octave) recipe {#tr-legacy}

*Tools > Advanced > QA Recipe > Octave (legacy)* reproduces the original Octave implementation exactly, for
comparison with earlier results only. Applied to head CT it is not scanner-independent:

| Issue | Legacy | Physical (default) |
|---|---|---|
| MTF frequency axis | assumes 2× upsampling while the code upsamples 4×: frequencies too low by ≈ 1.87 | true axis of the profile samples |
| MTF normalisation | first harmonic | DC (TG-233) |
| PS/MTF range | from the first MTF frequency to the PS Nyquist frequency - depends on the pixel spacing, reaches into the mirrored half of the MTF, undefined below 0.333 mm pixels | fixed band 0.1-0.8 mm⁻¹ |
| NPS/PS axes | three small scaling errors (grid step, DC position, radial axis) | exact DFT grid |
| Active-contour step | sign error when the seed lies fully inside a bright structure | absolute step |
| Classifier | chest-CT SVM (slice level, majority vote) | head-CT study model (chapter 9) |

Physical MTF areas are about 1.8 × the legacy values; values of the two recipes must never be compared.

## 12. Parameter reference {#tr-parameters}

| Step | Parameter | Value |
|---|---|---|
| Series | preferred slice thickness; axial tolerance | 4 mm; 60° |
| Segmentation | network input | 512 × 512, HU, pad/crop |
| Segmentation check | min. piece; limit; lobe island | 20 mm²; 30 (provisional); 90 % enclosed |
| Brain mask (fallback) | air / bone cut-off | −500 / 100 HU |
| Tissue references | grey/white: erosion, HU range; CSF | 1 px, −10-80 HU; 2 px, −10-30 HU; ≥ 40 px |
| Adaptive | window; smoothing; fixed range | 31 px; 1 mm; 15-50 HU |
| Fuzzy C-Means | m; iterations; spatial weight | 2; 50; 0.3 |
| HU thresholding | CSF / WM / GM; median filter | 0-20 / 20-32 / 32-45 HU; 3 px |
| Clean-up | opening; closing; min. region | 3 × 3; 5 × 5; 20 px |
| Eligibility | min. brain area | 20 % of the largest slice (provisional) |
| ROI1 | size; growth; foreign labels | 18-48 px; 2 px; ≤ 2 % |
| ROI1 | line score min.; falx line score | 0.5; 0.6 |
| ROI1 | falx: midline distance; angle | ≤ 8 mm; ≤ 15° |
| ROI1 | visibility contrast; flank | 5 HU; 6 px |
| ROI2 | ventricle: inside; radius | ≥ 98 %; 25 mm |
| ROI2 | white matter: tissue share; grey-matter penalty | ≥ 95 %; 40 mm per 100 % |
| MTF | upsampling; profile; samples | 4×; ≈ 5 mm; 15 |
| MTF | active contour | 1000 iterations, α = 0.2 |
| MTF | bump removal | peak at 40-60 %; zero crossings < 35 % |
| MTF | seeds; on-structure share | 5 + legacy; ≥ 30 % |
| Spectra | window; band; ratio step | radial Hann; 0.1-0.8 mm⁻¹; 0.01 mm⁻¹ |
| Verdict | min. slices; artefact indicator; outlier | 5; ≥ 10 slices; 3 × median |
| Verdict | bootstrap; confidence; seed | 1000; 95 %; fixed |
| Metal | HU; min. pixels | 2800 HU; 3 |
| Metal streaks | reach; deviation; share | 30 mm; 30 HU; 5 % (provisional) |
| Brainstem band | cerebellum share; band depth | ≥ 15 %; 3.5 HU (provisional) |
| Basal ganglia | block; plausible difference; levels | 8 %; 2-15 HU; 0.25 / 0.6 / 0.85 (provisional) |

## 13. References {#tr-references}

- AAPM Task Group 233: Samei E, Bakalyar D, Boedeker KL, et al. Performance evaluation of computed tomography
  systems: summary of AAPM Task Group 233. *Med Phys.* 2019;46(11):e735-e756.
- Barrett JF, Keat N. Artifacts in CT: recognition and avoidance. *RadioGraphics.* 2004;24(6):1679-1691.
- Bezdek JC, Ehrlich R, Full W. FCM: the fuzzy c-means clustering algorithm. *Comput Geosci.*
  1984;10(2-3):191-203.
- Bhattarai M, Bache S, Abadi E, Samei E. A systematic task-based image quality assessment of photon-counting
  and energy integrating CT as a function of reconstruction kernel and phantom size. *Med Phys.*
  2024;51(2):1047-1060.
- Boedeker KL, Cooper VN, McNitt-Gray MF. Application of the noise power spectrum in modern diagnostic MDCT:
  part I. *Phys Med Biol.* 2007;52(14):4027-4046.
- Cai JC, Akkus Z, Philbrick KA, et al. Fully automated segmentation of head CT neuroanatomy using deep
  learning. *Radiol Artif Intell.* 2020;2(5):e190183.
- Chan TF, Vese LA. Active contours without edges. *IEEE Trans Image Process.* 2001;10(2):266-277.
- Chuang KS, Tzeng HL, Chen S, et al. Fuzzy c-means clustering with spatial information for image
  segmentation. *Comput Med Imaging Graph.* 2006;30(1):9-15.
- Efron B, Tibshirani RJ. *An Introduction to the Bootstrap.* Chapman & Hall; 1993.
- European Commission. *European Guidelines on Quality Criteria for Computed Tomography.* EUR 16262 EN; 1999.
- Friedman SN, Fung GSK, Siewerdsen JH, Tsui BMW. A simple approach to measure computed tomography (CT)
  modulation transfer function (MTF) and noise-power spectrum (NPS) using the American College of Radiology
  (ACR) accreditation phantom. *Med Phys.* 2013;40(5):051907.
- Passand Z. *Quality assessment of clinical thorax CT images.* Dissertation; 2022.
- Sanders J, Hurwitz L, Samei E. Patient-specific quantification of image quality: an automated method for
  measuring spatial resolution in clinical CT images. *Med Phys.* 2016;43(10):5330-5338.
- Shannon CE. Communication in the presence of noise. *Proc IRE.* 1949;37(1):10-21.
