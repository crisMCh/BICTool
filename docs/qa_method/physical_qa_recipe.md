# The physical QA recipe: MTF, PS/NPS and PS/MTF features on true frequency axes

Status: implemented (default QA recipe since 2026-10-02).
Code: `source/bictool/domain/metrics/physical_spectra.py`, selected in
`source/bictool/domain/results/study_quality.py` (`recipe="physical"`; the
original behaviour remains available as `recipe="octave"`).

## 1. Summary

BICTool's image-quality features (MTF area, NPS/PS variances, PS/MTF ratio
features) were ported 1:1 from the Octave implementation behind Passand
(2022). Applying that implementation to head CT from three scanner models
showed that it is not scanner-independent:

* the MTF frequency axis is mis-scaled by a factor of about 1.87, because the
  code upsamples the ROI 4x but assumes 2x;
* the PS/MTF ratio is evaluated from the lowest MTF frequency up to the
  Nyquist frequency of the PS, a range that depends on the pixel spacing, and
  it reaches into the mirrored half of the MTF spectrum;
* for pixel spacings below 0.333 mm the range exceeds the MTF axis altogether
  and the ratio features become undefined (56 % of all slices of one scanner
  in our dataset);
* the NPS/PS frequency axes carry three smaller scaling errors.

None of this was visible on the thorax CT the method was developed on
(pixel spacing 0.64-0.89 mm), but head CT is reconstructed at 0.3-0.6 mm.

The **physical recipe** keeps every measurement step of the original method
(same ROIs, same edge detection, same edge profiles, same periodogram) and
changes only how the results are placed on the frequency axis and where the
features are evaluated:

1. MTF on the true frequency axis of the edge profiles, normalised to DC,
   truncated at the profiles' Nyquist frequency.
2. PS/NPS on the exact DFT frequency grid, radially binned around the true
   DC position.
3. All frequency-domain features evaluated on one fixed band,
   **0.1-0.8 mm^-1**, for every scan. This band lies inside the measurable
   range of any head CT with pixel spacing up to 0.625 mm.

On 16 head CT volumes (3 scanner models, pixel spacing 0.316-0.446 mm), all
features are now defined on every slice, and their distributions overlap
across scanners instead of differing by an order of magnitude (Section 6).

Two limitations remain and are documented in Section 7. First, two of the
three PS/MTF features (slope, location of the maximum) carry almost no
information on head-CT ROIs. Second, the SVM classifier was trained on
chest-CT features from the Octave recipe, so it must be retrained before its
verdict means anything.

## 2. Background: the method of Passand (2022)

Passand, Z. *Quality assessment of clinical thorax CT images* (dissertation,
2022) - the basis of the Octave code. The parts relevant here:

* **Data.** Thorax CT from 31 patients and two scanners (GE Revolution HD,
  Toshiba Aquilion Prime). In-plane pixel spacing **0.637-0.891 mm**
  (Table 4.1). Images were rated by radiologists on the reproduction of the
  major fissure.
* **MTF** (Sec. 4.4). A manually selected ROI (about 18 x 18 px) containing
  the fissure is "linearly interpolated by a factor of 2", Gaussian-filtered
  and sharpened for edge detection only. A Chan-Vese active contour (seed at
  the ROI centre) separates object and background, and edge pixels are taken
  from the contour. For every edge pixel, a line profile perpendicular to the
  edge is sampled on the unfiltered image: "the total length of the line
  profile was 5 mm". Bump removal follows, then LSF = d ESF / dx and
  MTF = |DFT(LSF)| (Eq. 4.2) after zero padding. The MTF is averaged over
  edges, and the feature is the area under the MTF curve.
* **NPS** (Sec. 4.5) from a homogeneous ROI near the structure (Eq. 4.3,
  with detrending by a 2nd-order polynomial), radially averaged to 1D. The
  feature is the area under the curve.
* **PS** (Sec. 4.6). The same estimator on the structured (MTF) ROI.
  "Both curves were normalized such that the value at the lowest frequency
  was equal to one", and then Ratio1 = PS(f)/MTF(f) (Eq. 4.6).
* **Features** (Table 4.2). MTF area, NPS area, PS area, "PS/MTF LF/HF
  ratio", "PS/MTF slope" (average slope), "PS/MTF max loc" ("frequency with
  maximum amplitude"), and the variance of the fissure ROI. An SVM trained on
  these reached 93.5 % 4-fold cross-validation accuracy (Sec. 5.6). There was
  no independent test set.

BICTool's original domain code (`mtf.py`, `nps.py`, `ps_mtf_ratio.py`) is a
faithful port of that Octave code (`funcCalcMTFNew.m`,
`funcCalcDigitalNps.m`, `funcCalcNPScompleteROI.m`, `funcRatioNpsMtf.m`).
It is kept unchanged as the "octave" recipe.

## 3. Findings in the Octave implementation

### F1 - The MTF frequency axis is mis-scaled by ~1.87

`funcCalcMTFNew.m` upsamples the ROI with `interp2(myimg, ImageInterPFactor)`
where `ImageInterPFactor = 2`. In Octave and MATLAB, `interp2(Z, k)` refines
the grid *k times*, inserting one sample between neighbours each time. The
image is therefore upsampled by 2^k = **4**, not 2 (BICTool's
`interp2_octave` reproduces this exactly: N -> 2(2N-1)-1). The code then
divides the pixel spacing by `ImageInterPFactor` = 2, so every "mm" computed
on the upsampled grid is in fact 0.5 mm:

* `vecLen = round(vecLenMM / PixSpc)` with `vecLenMM = 5` gives profiles of
  2·vecLen upsampled pixels of PixSpc/4 each, i.e. **~5 mm long**. That
  matches the thesis ("total length 5 mm") and the code's own comment
  ("15px @ PixSpc 0.33 mm -> line length ~5 mm").
* The frequency axis, however, uses `PixSpcLP = (2*vecLenMM)/LPpoints` =
  10 mm / 15 = 0.667 mm ("factor 2 because length of the profile is
  2*vecLen"). It assumes the profile is 10 mm long.

The 15 profile samples are really 5 mm / 14 = **0.357 mm** apart. Every
frequency on the Octave MTF axis is therefore too low by 0.667/0.357 ≈ 1.87;
the exact factor is 1.84-1.90 because of rounding in `vecLen`.

Verification:

* Derivation from the code above.
* `tests/domain/metrics/test_physical_spectra.py`: a Gaussian LSF with a known
  analytical MTF is reproduced to within 0.01 on the corrected axis at pixel
  spacings 0.33, 0.44 and 0.58 mm.
* On 15 head-CT volumes the factor is 1.84-1.90 (Section 6).

Consequences:

* The MTF area is reported in the wrong units.
* The MTF is compared with the PS at the wrong frequencies inside the ratio:
  PS(f) is divided by MTF(1.87 f).

### F2 - The ratio uses the mirrored half of the MTF

The MTF is the magnitude of the DFT of a real-valued LSF, so it is symmetric
about the sampling Nyquist frequency 1/(2Δx). Only the first half carries
information. The Octave axis runs from 0 to 1/Δx; on its own (mislabelled)
scale that is 0-1.5 mm^-1, with the Nyquist frequency at 0.75.

* `funcCalcMTFNew`'s MTF area integrates the first half only, which is
  correct.
* `funcRatioNpsMtf` uses the full axis. Whenever the PS reaches beyond
  0.75 mm^-1 on that scale, the ratio divides by mirrored MTF values.

### F3 - The PS/MTF range depends on the pixel spacing and can be undefined

`funcRatioNpsMtf` evaluates the ratio on `fMin:0.01:fMax`, where

* `fMin` is the first non-zero MTF frequency, and
* `fMax = fNPS(end)` is the PS Nyquist frequency 1/(2·px), which depends on
  the pixel spacing.

LF/HF splits this range at its midpoint, and the max location is expressed as
a fraction of it. As a result:

* **The same image content gives different features at different pixel
  spacings**, because the range is 0.86 mm^-1 wide at px = 0.58 mm and
  1.5 mm^-1 wide at px = 0.333 mm.
* For **px < 0.333 mm**, fMax exceeds the end of the MTF axis (1.5 on its
  scale), `interp1` returns NaN, and LF/HF and slope become **undefined**.
  This happened on 119 of 212 slices of one scanner in our dataset.
* As the range approaches the end of the MTF axis, the ratio divides by MTF
  values near their minimum. Octave LF/HF reached 40-200 there, against
  10-20 elsewhere, and the max location sat at the very end of the range
  (median 0.955).
* The MTF is normalised by its first harmonic (`MTF(2)` in Octave indexing).
  AAPM TG-233 normalises the MTF/TTF by its DC component.

The thorax data explain why this went unnoticed. With px = 0.64-0.89 mm,
fMax = 0.56-0.78 mm^-1, close to or below the mirror point at 0.75. Every
scan therefore had a similar range, and the features were internally
consistent *for that dataset*. Head CT at 0.3-0.6 mm gives fMax = 0.83-1.67.

### F4 - NPS/PS frequency axes (`funcCalcDigitalNps`, `funcCalcNPScompleteROI`)

For an N x N ROI the DFT frequencies are k/(N·px). The Octave code has three
deviations from that:

1. The 2D frequency vector is `linspace(-0.5, 0.5, N)/px`, a step of
   1/((N-1)·px). The variance computed by integrating the 2D NPS
   (`var_NPS`) is therefore too large by (N/(N-1))². That is about 8 % at
   N = 26, and the error depends on N, which differs between scanners.
2. `radialAverage` is centred at pixel N/2 (1-based), one pixel off the DC
   position N/2 + 1 that `fftshift` produces for even N. It also uses
   2-pixel-wide overlapping annuli.
3. The 1D axis is `linspace(0, 1/(2px), N/2)`. That maps radius N/2-1 to the
   Nyquist frequency, a step of (N/2)/(N/2-1) ≈ 1.08 times too large.

### F5 - Further differences between the thesis and the code (not changed)

These are listed for completeness. They are *not* part of the physical
recipe (see Section 7).

* NPS detrending: the thesis subtracts a 2nd-order polynomial. In the code
  the detrended image `S` is computed but the line using it is commented out
  ("estefade nemishe", Persian for "not used").
* PS normalisation: the thesis normalises PS and MTF before forming the
  ratio (Sec. 4.6). The code does not, so the Octave slope carries the PS
  magnitude in HU²mm².
* Homogeneous sub-region selection by k-means for the NPS (Sec. 4.5.2) is not
  part of the live Octave pipeline (`funcCalcNPScompleteROI` uses the
  complete ROI).

## 4. Requirements for head CT

The method must give comparable numbers for any head CT, whatever the scanner.

* **Pixel spacing.** Routine head CT is reconstructed with a 200-250 mm
  display field of view on a 512² matrix, i.e. 0.39-0.49 mm. Larger matrices
  (768², 1024²) and smaller fields of view reach 0.25-0.35 mm. Our dataset
  covers 0.316-0.446 mm:

  | Scanner | Pixel spacing |
  |---|---|
  | Canon Aquilion Prime | 0.380-0.446 mm |
  | Siemens (ER) | 0.316-0.389 mm |
  | Siemens NAEOTOM Alpha (photon counting) | 0.383-0.440 mm |

* **Frequencies in physical units.** AAPM TG-233 expresses TTF/MTF and NPS on
  an absolute spatial-frequency axis in mm^-1. Resolution is summarised by
  f50 and f10, noise texture by the NPS peak and average frequencies. Noise
  and resolution are combined only on a common frequency axis (the
  detectability index d'). Patient-based measurements (Sanders et al. 2016,
  air-skin edge) follow the same convention.
* **Comparable features.** Any feature that integrates, splits or locates
  something along the frequency axis must do so on the same absolute band for
  every scan. Otherwise the pixel spacing, not image quality, determines the
  value.

## 5. The physical recipe

### 5.1 MTF

The edge detection, profiles, bump removal and LSF of `calc_mtf` are
unchanged. Given the per-edge |DFT(LSF)| of length n, `physical_mtf` does
three things:

* **Sample spacing.** Δx = 2·vecLen·(px/4)/(LP_POINTS-1). This is the real
  distance between profile samples, ≈ 0.357 mm.
* **Frequency axis.** f_k = k/(n·Δx), the exact DFT grid, kept only up to the
  Nyquist frequency 1/(2Δx) ≈ 1.4 mm^-1.
* **Normalisation.** MTF(f) = |DFT(LSF)|(f) / |DFT(LSF)|(0), i.e. DC
  normalisation (TG-233).

Edges whose MTF is not finite are dropped as before (at most 50 %, otherwise
the slice fails).

**Active-contour step size fixed (2026-10-20).** The Chan-Vese contour that
finds the edge used the step `dt = 0.45 / max(dphidt)` (Octave port, after
Lankton). When the seed lies fully inside a bright structure - the intended
case, a seed on the falx/septum - every point of the narrow band wants to join
the region, all `dphidt` are negative, `dt` becomes negative and the update
runs backwards: the region shrinks to nothing within ~10 iterations and the
seed "finds no edges". Only seeds beside the line survived, which then measured
the wrong edge. The physical recipe now uses the usual step
`0.45 / max(|dphidt|)` (`active_contour.region_seg`, test
`tests/domain/segmentation/test_active_contour.py`); the legacy octave recipe
keeps the original. Batch rerun of the 15 rated studies (2026-10-20,
`debugs/classifier/qa_2026-10-20b`): the fallback seed is no longer used
(24 before), 21 failed slices (18 before), 270 of 392 falx MTF areas unchanged
and the rest within ±0.04 (90th percentile), median 0.506 → 0.501; septum alike.
Study verdicts: no category changed, but the probabilities moved where newly
measured slices entered: in the acute-SDH study 5 upper slices with fresh blood
along the falx (they failed before) raise the falx variance and the PS/MTF
ratio, and P(acceptable) fell from 0.77 to 0.05 - the falx measures pathology
there, not image quality (a limitation, section 7). head-ct-1 was trained on
features from the old contour and is not refit.

**The measured edge must belong to the structure (added 2026-10-20).** The
active contour starts from a seed; seeds are tried in order (brightest points
on the CNN falx/septum label first, the Octave-era quarter-of-the-ROI point
last) until one gives usable edges. On a thin falx all seeds on the line can
fail (profiles across a thin line are bumps, which the bump removal discards),
and the last seed, lying in the tissue beside the line, then grew a region
there and measured *that* edge. In the 15 rated studies this happened in 20 of
the 24 ROI1s measured with the last seed.
When the structure is known (auto ROIs and any ROI on a segmented slice), a
seed's result now counts only if at least **30 %** of its active-contour region
lies on the structure (within 1 px, compared on the 4× grid). Over all 488
ROI1s of the rated studies: wrong last-seed results 0.00-0.29, correct ones
0.69-1.00, regular seeds median 0.98 (13 of 464 below 0.30, which then fall
through to the next seed). If no seed qualifies, the slice fails with
"MTF edge not on the structure" and is left out of the study verdict, like any
failed slice. Without a known structure (manual ROI on an unsegmented slice)
nothing changes.

**MTF area** = ∫₀^0.8 MTF(f) df per edge, in mm^-1, then averaged over edges.
The MTF is 1 at f = 0 and normally falls with frequency. A system that
transferred all contrast up to 0.8 mm^-1 (MTF = 1 across the band) would
therefore reach the maximum, 1 × 0.8 = 0.8 mm^-1. A measured area of 0.5
means the edge keeps on average 0.5/0.8 ≈ 62 % of its contrast across the
band. (Edge-enhancing kernels or noise can push the MTF slightly above 1 at
some frequencies, so 0.8 is a practical rather than a strict ceiling.)

### 5.2 PS and NPS (`power_spectrum`)

The estimator is the same as in the Octave port: mean removal, radial Hann
window, and |DFT|²·px²/(N²·mean(w²)). The corrections are:

* the 2D grid is `fftfreq(N, px)`;
* `var_nps` = Σ NPS · (1/(N·px))²;
* radial bins are 1 bin (1/(N·px)) wide around the true DC position, up to
  and including the Nyquist frequency.

White noise of variance σ² integrates to σ² within 5 % at every pixel spacing
tested (unit test).

### 5.3 PS/MTF ratio features on a fixed band

Following Passand Sec. 4.6, both curves are normalised at the lowest
frequency of the band, and

  R(f) = [PS(f)/PS(f_lo)] / [MTF(f)/MTF(f_lo)],  f ∈ [f_lo, f_hi], step 0.01 mm^-1

with the features

* **LF/HF** = ∫ R over the lower half of the band ÷ ∫ R over the upper half
  (split at the midpoint, 0.45 mm^-1).
* **Slope** = mean dR/df, in mm (per mm^-1).
* **Max location** = argmax_f R(f), in mm^-1. That is the thesis definition
  ("frequency with maximum amplitude"), now in physical units rather than a
  fraction of a varying range.

### 5.4 Choice of the band: 0.1-0.8 mm^-1

No standard prescribes a frequency band for patient-based PS/MTF features.
AAPM TG-233 requires absolute frequency units, but it reports f50, f10 and
the NPS peak and average frequencies rather than features integrated over a
fixed band. The band is therefore a design parameter of this recipe. Each of
its limits is set by a physical constraint, and the upper limit is checked
against a published measurement of a standard adult head protocol.

**Upper limit, 0.8 mm^-1.** It must satisfy three conditions.

1. **The image must contain the frequency on every scan (sampling
   theorem).** A digital image represents spatial frequencies only up to its
   Nyquist frequency 1/(2·px) (Shannon 1949). Above that, the PS of the
   image does not exist. 0.8 mm^-1 is the Nyquist frequency of a 0.625 mm
   pixel, i.e. 512² at a 320 mm display field of view. Head CT is
   reconstructed at 0.39-0.49 mm (200-250 mm field of view), for example
   0.49 mm in Friedman et al. (2013) and a 22 cm field of view (0.43 mm) in
   Solomon et al. (2012); our dataset is 0.316-0.446 mm. So the band is
   measurable on every routine head scan, with margin.
2. **A standard head reconstruction must still carry signal and noise
   there.** Friedman et al. (2013) measured a standard adult head protocol
   (Siemens SOMATOM Definition Flash, 120 kVp, 0.49 mm pixels, ACR
   phantom):

   * the axial MTF "had a value of 0.1 at 0.62 mm^-1";
   * the radial NPS peaked "at ~0.25 mm^-1 and dropped to 0 HU²mm³ by
     0.8 mm^-1".

   For standard head reconstructions, the frequencies above ~0.8 mm^-1
   therefore hold neither measurable noise nor transferred contrast. They
   would add only estimation noise to the features, and in the PS/MTF ratio
   they divide by an MTF close to zero. That is the cause of the Octave
   ratios of 40-200 (F3). The band ends where, according to this
   measurement, the image content of a standard head protocol ends.
3. **The MTF must be measurable there.** The profiles' own Nyquist
   frequency 1/(2Δx) is ≈ 1.4 mm^-1 (Section 5.1), well above the band.

**Lower limit, 0.1 mm^-1.** This limit follows from the ROI, not from the
scanner. The PS and NPS are computed on ROIs about 11 mm wide (20-34 px,
depending on the pixel spacing). Their first non-zero frequency bin is
1/(N·px) ≈ 0.09 mm^-1, and below it there is only the removed mean (DC).
Because the ROI has a physical size, the bins fall at the same absolute
frequencies on every scanner.

**Sensitivity of the choice.** Any upper limit between ~0.7 and ~0.8 mm^-1
satisfies conditions 1-3 for routine head CT. Within that range the exact
value changes the absolute feature values but not their comparability. The
band is one constant, `FEATURE_BAND_MM` in `physical_spectra.py`. Results
computed with different bands must not be compared.

**Photon-counting CT and newer scanners.** The sampling argument (condition
1) does not depend on detector technology. For the content argument
(condition 2), the 2013 energy-integrating measurement is confirmed for
photon-counting CT by Bhattarai et al. (2024, AAPM TG-233 metrology), who
compared the NAEOTOM Alpha with an energy-integrating Siemens Force:

* With the soft Br40 kernel, the TTF f50 was 0.35 mm^-1 on photon-counting
  and 0.37 mm^-1 on energy-integrating CT. With a soft kernel, resolution is
  limited by the kernel, not by the detector.
* The photon-counting advantage appears with sharp kernels (Br64: f50 =
  0.76 mm^-1) and in the ultra-high-resolution mode. There the detector
  resolves up to ~2.9 mm^-1 at 10 % MTF, and Rajendran et al. (2022) report
  in-plane resolution down to 125 µm.

For a bell-shaped MTF, f10 ≈ 1.8 × f50, so f50 = 0.35 mm^-1 corresponds to
f10 ≈ 0.64 mm^-1. That is an estimate, consistent with Friedman's 0.62.
Brain reconstructions use soft-tissue head kernels (Hr44 in our
photon-counting data), and for those the band holds on both detector types.

**Scope.** The band is defined for **soft-tissue (brain-window) head
reconstructions**. Sharp head kernels (e.g. Hr56-Hr72), bone kernels and
ultra-high-resolution reconstructions transfer real contrast above
0.8 mm^-1, and the features are blind to it. Such reconstructions would need
their own, wider band, and their results must not be mixed with soft-kernel
results. Our own object-MTF curves on the photon-counting falx ROIs still
show MTF ≈ 0.27 at 0.8 mm^-1. That is probably biased high by the
bump-removal step (Section 7, limitation 5), but it means our data alone
cannot show that nothing lies above 0.8 mm^-1 on these scans.

For pixel spacings above 0.625 mm, the band is shortened to what the PS
covers and `band_truncated` is set on the result. Such features are not
directly comparable with full-band ones. Where the band cannot be formed at
all, the slice fails with a clear error.

For orientation: on its thorax data the original method effectively
evaluated the PS between ~0.05 and 0.56-0.78 mm^-1 (F3). So 0.1-0.8 mm^-1 is
close to the range the method was designed and validated on, but now fixed
and in true units.

## 6. Validation

### 6.1 Unit tests (`tests/domain/metrics/test_physical_spectra.py`)

* Profile length is ~5 mm at px = 0.316-0.58 mm.
* A known Gaussian blur's MTF is reproduced to within 0.01, and the mirrored
  half is dropped.
* White-noise NPS integrates to the true variance within 5 % at
  px = 0.33/0.44/0.58 mm.
* The PS frequency axis is the DFT grid.
* **Pixel-spacing invariance.** The same analytical PS and MTF, sampled the
  way BICTool samples them at px = 0.32, 0.44 and 0.58 mm, give the same
  LF/HF, slope and max location within 5 %.
* Identical PS and MTF shapes give a flat ratio (LF/HF = 1, slope = 0).
* Truncated or uncovered bands are flagged or raised.

### 6.2 Head-CT dataset

The dataset is 16 volumes from 3 scanners, with automatic ROIs on the falx
(dural folds, 427 analysed slices) and the septum (81 slices), 508 analysed
slices in total. It was produced with `python -m
bictool.application.batch_qa`. Results are local only (`debugs/`, derived
from patient data). Per-scanner medians over all analysed slices (falx and
septum), with the interquartile range in brackets:

| Feature | Recipe | Canon (101 slices) | Siemens ER (212) | Siemens PCCT (195) |
|---|---|---|---|---|
| PS/MTF LF/HF | octave | 14.1 [10.1, 19.2] | 82.5 [40.7, 172], **undefined on 119** | 13.7 [10.2, 22.5] |
| PS/MTF LF/HF | physical | 10.8 [8.6, 14.3] | 18.6 [12.1, 29.9], undefined on 0 | 11.5 [8.4, 17.1] |
| PS/MTF max location | octave | 0.14 (fraction) | 0.955 (fraction) | 0.20 (fraction) |
| PS/MTF max location | physical | 0.10 mm^-1 | 0.10 mm^-1 | 0.13 mm^-1 |
| MTF area | octave | 0.294 | 0.262 | 0.282 |
| MTF area | physical | 0.525 mm^-1 | 0.482 mm^-1 | 0.515 mm^-1 |

On the same 508 slices, physical MTF area = 1.82 × Octave MTF area (median;
interquartile range 1.77-1.87), with a correlation of 0.97 (rank correlation
0.97). The ranking of slices is preserved. The factor is the axis error of F1
(1.84-1.90), slightly reduced because the Octave integral also included the
small MTF tail above 0.8 mm^-1. MTF areas from the thesis or from Octave-era
results (e.g. 0.31-0.34 for acceptable thorax images) are therefore on a
different scale and must not be compared with physical-recipe values.

Observations:

* All features are defined on every slice. The Siemens ER volumes, all at
  < 0.39 mm, no longer produce undefined or exploding ratios.
* The remaining scanner difference has a plausible physical reading. The ER
  scans use a soft head kernel (H30s), the PCCT scans a sharper one (Hr44).
  On the physical axis, ER shows the lowest MTF area and the highest LF/HF
  (energy concentrated at low frequencies). With the Octave recipe the ER
  values were dominated by the range artefact (F3) instead.
* With the Octave-trained SVM, verdicts change substantially between the
  recipes. That is expected, because the classifier's input scales changed
  (Section 7).

### 6.3 Sensitivity to the band limit

To check whether results depend on the exact upper limit (Section 5.4), the
band-dependent features were recomputed with upper limits of 0.7, 0.8 and
0.9 mm^-1. Everything else was identical: the same 508 slices, the same ROIs
and the same MTF seeds. No slice needed a truncated band.

**Agreement with the 0.8 band** (Spearman rank correlation; volume level =
mean per volume, 16 volumes):

| Feature | Slice level, falx | Slice level, septum | Volume level, falx | Volume level, septum |
|---|---|---|---|---|
| MTF area | 0.997 / 0.997 | 0.992 / 0.995 | 0.997 / 0.985 | 0.987 / 0.991 |
| PS/MTF LF/HF | 0.975 / 0.976 | 0.973 / 0.974 | 0.974 / 0.976 | 0.965 / 0.974 |
| PS/MTF slope | 0.934 / 0.897 | 0.911 / 0.889 | 0.818 / 0.900 | 0.908 / 0.938 |
| PS/MTF max location | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |

Each cell gives the band 0.1-0.7 / the band 0.1-0.9.

**Per-scanner medians, falx** (Canon / Siemens ER / Siemens PCCT):

| Feature | Band 0.1-0.7 | Band 0.1-0.8 | Band 0.1-0.9 |
|---|---|---|---|
| MTF area | 0.494 / 0.459 / 0.491 | 0.525 / 0.479 / 0.517 | 0.547 / 0.495 / 0.537 |
| PS/MTF LF/HF | 7.2 / 11.2 / 7.7 | 11.2 / 18.5 / 11.5 | 17.4 / 29.9 / 18.9 |
| PS/MTF slope | -1.63 / -1.65 / -1.63 | -1.41 / -1.42 / -1.42 | -1.24 / -1.25 / -1.25 |

Findings:

* **MTF area and LF/HF are robust to the choice of band.** The ranking of
  slices and of volumes is practically unchanged (ρ ≥ 0.97), and so is the
  scanner ordering: Siemens ER has the lowest MTF area and the highest LF/HF
  in every band. The absolute values do shift, LF/HF strongly, because it
  splits the band at its midpoint. Values computed with different bands must
  therefore never be mixed, but conclusions drawn within one band do not
  depend on whether that band ends at 0.7, 0.8 or 0.9 mm^-1.
* **The slope mostly measures the band width.** Its value is ≈ -1/(band
  width) in every band (-1.63 / -1.42 / -1.25 for widths 0.6 / 0.7 / 0.8),
  and its ranking is the least stable (ρ 0.82-0.94). This confirms
  limitation 2 (Section 7): on head-CT ROIs the slope carries little
  information beyond the band itself.
* **The max location is invariant**, because it sits at the band start on
  practically every slice (limitation 2).
* **Exploratory check against the volume labels.** The study folders are
  named "good" or "bad" (8/6 falx volumes, 7/6 septum volumes; the labelling
  criteria are not documented). Neither MTF area nor LF/HF separates the two
  groups in any band (AUC 0.35-0.56), so the band choice does not create or
  remove a group difference. For comparison, the band-independent NPS
  variance of the white-matter ROI2 next to the falx does separate them
  (AUC 0.90, Mann-Whitney p = 0.013). With 14 volumes and undocumented
  labels this is a hint for the retraining phase, not a result.

## 7. Limitations and open points

1. **The classifier must be retrained.** The SVM was trained on chest CT,
   with Octave-recipe features from a single anatomical structure (the major
   fissure). Its verdict on head CT was never valid. With physical-recipe
   features its inputs are on different scales again. BICTool therefore
   labels the verdict as untrusted. A head-CT training set with radiologist
   ratings, using physical-recipe features, is required.
2. **Slope and max location are nearly uninformative on head-CT ROIs.**
   * ROI1 contains a bright falx or septum line, and its PS is dominated by
     that structure: it falls monotonically across the band.
   * The normalised ratio therefore drops from 1 to ~0 on almost every slice.
   * So the slope ≈ -(1-R(f_hi))/0.7 ≈ -1.42 mm, and the maximum sits at the
     band start (0.1 mm^-1).
   * For the slope this holds by construction, not only empirically. The mean
     of the successive differences telescopes, mean(ΔR)/Δf =
     (R(f_hi) - R(f_lo)) / (f_hi - f_lo). The slope therefore depends only on
     the curve's two end values, never on its shape in between (a bump
     changes nothing). With R(f_lo) = 1 by normalisation and R(f_hi) ≈ 0,
     it is fixed at about -1/(band width).
     `debugs/ps_mtf_ratio_explained.png` (local) shows typical curves.
   * In the thorax data the fissure PS first rises, which made both features
     discriminative there.

   Candidate replacements for the retraining phase are noise-texture
   descriptors on the same band (TG-233 NPS peak and average frequency) or
   the log-slope of R(f). The current values are kept so the feature vector
   keeps its structure.
3. **NPS detrending is not applied** (F5). The thesis prescribes it, and
   removing slow background trends is common practice when estimating NPS
   outside uniform phantoms. The automatic ROI2 placement keeps anatomy out
   of ROI2, but a slow intensity trend inside ROI2 would still add
   low-frequency power.
4. **Spectral resolution.** ROI1 and ROI2 are ~11 mm (20-34 px). The PS/NPS
   therefore has only ~8 bins in the band, against the 64 x 64-pixel ROIs
   TG-233 recommends for phantom NPS. The estimates are coarse, but
   consistent between scanners because the ROI size is physical.
5. **Object MTF, not system MTF.** Like the original method, the MTF is
   measured on an anatomical edge (falx or septum), so it includes the
   structure's own profile (thickness, partial volume). It describes
   "sharpness at this structure", not the scanner's MTF. Compare TG-233's TTF
   and the air-skin method of Sanders et al. (2016), which uses an edge of
   known shape.

   Our measured object-MTF f10 (0.85-1.22 mm^-1 on falx edges, Hr44/H30s and
   Canon head kernels) is higher than the phantom f10 of 0.62 mm^-1 reported
   for a standard head protocol by Friedman et al. (2013). Possible reasons
   are sharper kernels, and the bump-removal step of the original method,
   which replaces the ESF tails by constants and so narrows the LSF. The band
   limit therefore relies on the published phantom values (Section 5.4), not
   on our object-MTF measurement.
6. **ROI1 size.** The automatic ROI1 target size was derived from ±5 mm
   profiles. The profiles are really ±2.5 mm (F1), so ROI1 could be smaller.
   That is unchanged for now, since it also changes the PS frequency bins.
7. **Pathology on the measured structure.** ROI1 sits on the falx; when the
   falx itself is abnormal - e.g. an acute subdural haematoma along it - the
   falx variance, PS and PS/MTF ratio reflect the pathology, not the image
   quality, and the study verdict leans towards "not acceptable" (acute-SDH
   study: P(acceptable) 0.05 after the 2026-10-20 contour fix). Not handled
   yet; candidate for head-ct-2: flag ROI1s with blood-density HU on the falx.

## 8. Implementation map

| What | Where |
|---|---|
| Recipe selection (`"physical"` default, `"octave"`) | `domain/results/study_quality.py` (`RECIPES`, `evaluate_slice_quality(recipe=...)`, `SliceQualityResult.recipe`) |
| Switching in the app | Tools > Advanced > QA Recipe (session-only; a "Legacy QA recipe (Octave)" note shows in the Run Metrics box while active; switching clears existing QA results; locked during a QA batch) |
| Switching in the batch tool | `python -m bictool.application.batch_qa ... --recipe octave` |
| Profile sample spacing, physical MTF, MTF area | `domain/metrics/physical_spectra.py` (`mtf_sample_spacing_mm`, `physical_mtf`, `mtf_area_in_band`) |
| PS/NPS on the DFT grid | `physical_spectra.power_spectrum` |
| Ratio features on the band | `physical_spectra.ratio_ps_mtf_in_band` (`FEATURE_BAND_MM = (0.1, 0.8)`), `NpsMtfRatioResult.band` / `.band_truncated` |
| Octave recipe (unchanged ports) | `domain/metrics/mtf.py`, `nps.py`, `ps_mtf_ratio.py` |
| Recipe written to exports | `integration/results_export.py` ("QA recipe" rows) |
| Tests | `tests/domain/metrics/test_physical_spectra.py`, `tests/domain/results/test_study_quality.py` |
| Dataset run | `python -m bictool.application.batch_qa DATASET_DIR ... --out OUT_DIR` |
| Intermediate steps (Results > Per slice > Show: Intermediate steps) | `mtf.calc_mtf_with_steps` / `MtfSteps`, `study_quality.SliceSteps` (`keep_steps=True`, GUI only), `presentation/viewmodels/slice_steps.py`, `presentation/views/slice_steps_panel.py` |

## 9. Intermediate steps in the app (Results > Per slice > Show: Intermediate steps)

*Added 2026-10-08.* The switch "Show: Curves | Intermediate steps (advanced)" above the plots replaces the
four curves of the shown slice by how they were measured; the tab always opens on the curves. The steps are kept during the QA run in the app (`evaluate_slice_quality(..., keep_steps=True)`;
the numbers are identical with and without - tested) and cost no measurable time (postop study, 33 ROI
pairs: 14.9 s with vs 15-16 s without; a few MB of memory; drawing a slice ≈ 60-80 ms). The batch tool does
not keep them.

| Tab | What it shows | What to look for |
|---|---|---|
| MTF steps | ROI1 upsampled 4x as the MTF sees it, with the active-contour outline, the detected edge pixels (grey = dropped later), a sample of the sampled line profiles and the contour's start box; the edge profiles (ESF, raw HU) and their derivative (LSF, after mean removal, low-pass and bump removal); the MTF of every edge with the mean | The outline should follow the structure (falx), not noise or a neighbouring structure; profiles should be steps (dark → bright). A thin structure gives a *bump* (both sides inside one 5 mm profile) and an LSF with a positive and a negative lobe - not a clean edge MTF (this is what the septum shows) |
| MTF spread | Edge pixels coloured by their MTF area (dark = blurry, bright = sharp), the histogram of the per-edge areas with the slice's mean, and how many edges survive each step (found → full profile inside the ROI → bump removal → finite MTF) | A wide or two-peaked histogram, or few surviving edges, means the slice's MTF area rests on a mix of edges |
| ROIs & 2D spectra | ROI1 and ROI2 (W80/L40), the 2D PS of ROI1 and the 2D NPS of ROI2 (log scale, DC in the centre) | Noise is a round blob; lines or a stretched shape mean directional structure (an edge in the ROI, or streaks) |
| PS/MTF & features | The PS/MTF ratio on the feature band with its low half and high half shaded (LF/HF = area left / area right), the maximum, and the feature vector | Which part of the band drives the LF/HF ratio |

The legacy (Octave) recipe shows the same tabs; its MTF axis is shown up to its middle (the second half
mirrors the first), without the band shading.

## References

* Passand, Z. (2022). *Quality assessment of clinical thorax CT images.*
  Dissertation. Sections 4.4-4.8, 5.2-5.6, Table 4.1, Table 4.2.
* Samei, E., Bakalyar, D., Boedeker, K. L., et al. (2019). Performance
  evaluation of computed tomography systems: Summary of AAPM Task Group 233.
  *Medical Physics* 46(11), e735-e756. https://doi.org/10.1002/mp.13763.
  Full report: AAPM Report No. 233,
  https://www.aapm.org/pubs/reports/RPT_233.pdf. TTF from edge (ESF bins of
  1/10 pixel, DC normalisation, f50/f10), NPS (radial 1D, f_P, f_A),
  detectability index.
* Sanders, J., Hurwitz, L., Samei, E. (2016). Patient-specific quantification
  of image quality: An automated method for measuring spatial resolution in
  clinical CT images. *Medical Physics* 43(10), 5330-5338.
  https://doi.org/10.1118/1.4961984
* Samei, E., Flynn, M. J., Reimann, D. A. (1998). A method for measuring the
  presampled MTF of digital radiographic systems using an edge test device.
  *Medical Physics* 25(1), 102-113. Edge method, ESF sampling interval and
  frequency axis, aliasing.
* Friedman, S. N., Fung, G. S. K., Siewerdsen, J. H., Tsui, B. M. W.
  (2013). A simple approach to measure computed tomography (CT) modulation
  transfer function (MTF) and noise-power spectrum (NPS) using the American
  College of Radiology (ACR) accreditation phantom. *Medical Physics* 40(5),
  051907. https://doi.org/10.1118/1.4800795. Standard adult head protocol,
  0.49 mm pixels: MTF 0.1 at 0.62 mm^-1, NPS peak ~0.25 mm^-1, NPS 0 by
  0.8 mm^-1.
* Solomon, J. B., Christianson, O., Samei, E. (2012). Quantitative comparison
  of noise texture across CT scanners from different manufacturers. *Medical
  Physics* 39(10), 6048-6055. https://doi.org/10.1118/1.4752209. NPS compared
  across vendors and kernels in absolute mm^-1, 22 cm field of view.
* Bhattarai, M., Bache, S., Abadi, E., Samei, E. (2024). A systematic
  task-based image quality assessment of photon-counting and energy
  integrating CT as a function of reconstruction kernel and phantom size.
  *Medical Physics* 51(2), 1047-1060. https://doi.org/10.1002/mp.16619.
  NAEOTOM Alpha vs SOMATOM Force, TG-233 metrology: TTF f50 for Br40
  0.35 mm^-1 (photon-counting) vs 0.37 mm^-1 (energy-integrating), and for
  Br64 0.76 mm^-1.
* Rajendran, K., Petersilka, M., Henning, A., et al. (2022). First clinical
  photon-counting detector CT system: technical evaluation. *Radiology*
  303(1), 130-138. https://doi.org/10.1148/radiol.212579. Maximum in-plane
  resolution 125 µm (high-resolution mode) and 208 µm (standard).
* Shannon, C. E. (1949). Communication in the presence of noise.
  *Proceedings of the IRE* 37(1), 10-21. Sampling theorem: no frequency
  content above 1/(2·sampling interval).
* Christianson, O., Winslow, J., Frush, D. P., Samei, E. (2015). Automated
  technique to measure noise in clinical CT examinations. *AJR* 205(1),
  W93-W99. Patient-based noise measurement.
* Boedeker, K. L., Cooper, V. N., McNitt-Gray, M. F. (2007). Application of
  the noise power spectrum in modern diagnostic MDCT: part I. Measurement of
  noise power spectra and noise equivalent quanta. *Physics in Medicine and
  Biology* 52(14), 4027-4046. The NPS estimator and normalisation used by
  TG-233.
