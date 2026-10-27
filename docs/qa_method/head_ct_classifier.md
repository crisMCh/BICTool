# Image-quality classifier for head CT: analysis and decisions

Status: analysis complete, design decided (2026-10-06). Implementation order:
(1) app restructuring to ROI pairs per slice (D11) - done, (2) training and
model card - done, (3) study-verdict calculation - done, (4) one-time test - done (6.3),
(5) exports and app display - done (6.4). Model card: `head_ct_model_card.md`.
Companion to `physical_qa_recipe.md` (how the features are measured).
This document records why the legacy classifier is replaced, what the
available data allows, what was tested, and what was decided.

## 1. Summary

* The legacy classifier is an SVM trained on chest CT (Passand 2022). It is
  not valid for head CT, its output (a "decision value") is not a confidence
  level, and its slice-level design does not work on our head-CT data.
* We have 15 usable head-CT studies rated by one radiologist (good / bad /
  bad with artefacts) from three scanners. They were split into 11 training
  and 4 locked test studies.
* On the training studies, a classifier on **single slices** performs at
  chance level for every feature set tried. A classifier on **whole
  studies**, using the median of each feature over the study's falx slices,
  separates good from not-acceptable studies (patient-wise
  cross-validation: 11/11 correct). The signal is image noise around the
  falx, and it holds within each scanner.
* Septum-based features do not separate good from bad studies. They agree
  with the falx on scan properties (sharpness, noise texture), which makes
  them useful as a plausibility check.
* The two kinds of "bad" differ. Plain "bad" studies are noisy everywhere.
  "Bad with artefacts" studies are inconsistent: the noise level jumps
  between slices. A noise-consistency measure separates the two artefact
  studies from all others, but n = 2.
* Decision: replace the legacy slice-level SVM with a **study-level model
  that reports a probability with a 95 % interval** (Acceptable / Not
  acceptable / Uncertain), **reasons** (features outside the range of
  rated-good studies), and a **provisional artefact indicator**. Every
  statement carries a confidence level. The legacy classifier remains
  available with the legacy ("octave") QA recipe.

## 2. The legacy classifier

Source: Passand (2022), Sections 4.8 and 5.6; ported 1:1 from the Octave
code (`domain/results/classification.py`, model `_svm_model.npz`).

* **Model.** RBF-kernel support vector machine (libsvm) on 7 features of
  **one slice**: MTF area, NPS of the homogeneous ROI, NPS of the structured
  ROI, variance of the structured ROI, PS/MTF LF/HF ratio, PS/MTF slope,
  PS/MTF max location (thesis Table 4.2).
* **Training data.** 31 thorax CT images (2 scanners, pixel spacing
  0.64-0.89 mm), rated by radiologists on the reproduction of the major
  fissure. Reported accuracy: 93.5 % in 4-fold cross-validation. There was
  no independent test set.
* **Output.** A decision value, the signed distance from the SVM's
  separating boundary. Its sign gives the slice verdict (positive =
  acceptable). The study verdict is the majority vote of the slice verdicts
  (a tie is "inconclusive").

**Why it cannot be used for head CT:**

1. **Wrong anatomy and task.** It was trained on chest CT and the lung
   fissure.
2. **Wrong feature scale.** It was trained on Octave-recipe features, whose
   frequency axes are mis-scaled and whose ratio range depends on the pixel
   spacing (`physical_qa_recipe.md`, F1-F4). Physical-recipe features are
   on different scales.
3. **The decision value is not a confidence level.**
   * It is a distance in the SVM's internal units. It depends on the model's
     settings, is not calibrated against outcomes, and has no interval.
   * +0.02 and +2.0 both mean "acceptable", and how much more likely the
     second one is cannot be read from them. The model was not trained to
     output probabilities.
   * On our head CT the decision values clustered between -0.5 and +0.05,
     i.e. close to the boundary. That is consistent with a model that has
     no information.
4. **Slice-level design.** On head CT, single slices are too noisy to
   classify (Section 5.1).

## 3. Data

### 3.1 Studies and ratings

All ratings are from **one radiologist**, encoded in the study folder
names:

| Rating | Meaning | Studies |
|---|---|---|
| good | acceptable | 9 (Canon 1, Siemens ER 3, Siemens photon-counting 5) |
| bad | not acceptable | 4 (Canon 2, Siemens ER 1, photon-counting 1) |
| bad with artefacts | not acceptable | 3 (Canon motion, Siemens ER motion, Siemens ER implants) |

Decisions on individual studies:

* **Canon "pathology"** is counted as **good** (decision by the user,
  2026-10-05). It is the only Canon study that is not rated bad.
* **Siemens ER "stroke"** is **unrated** and therefore left out. Its CTDIvol
  header value is also invalid (0.0).
* **Canon "bad motion artefacts"** is left out. It has only 5 slices and no
  usable falx ROI, so it would contribute no data.

For training, "bad" and "bad with artefacts" form the class **not
acceptable**. That gives **15 usable studies: 9 good, 6 not acceptable.**

Features come from the automatic ROIs and the physical QA recipe
(`physical_qa_recipe.md`), computed with
`python -m bictool.application.batch_qa`. All results derived from patient
data stay local (`debugs/`), and so does the ratings file, because it names
patient studies.

### 3.2 Ratings are confounded with the scanner

| Scanner | good | not acceptable |
|---|---|---|
| Canon Aquilion Prime | 1 | 2 (+1 without data) |
| Siemens (ER) | 3 | 3 |
| Siemens NAEOTOM Alpha (photon counting) | 5 | 1 |

Anything that identifies the scanner can look like a quality predictor:
kernel sharpness, MTF area, pixel spacing, dose. For that reason every
analysis is checked within each scanner (Section 5.2), and the
cross-validation includes a "scanner-only" baseline (Section 5.1).

### 3.3 Dose

| Scanner | CTDIvol | DLP |
|---|---|---|
| Canon | 42.8-47.2 mGy | not in the image headers |
| Siemens ER | 49.3-51.1 mGy (one study: invalid 0.0) | not in the image headers |
| Photon counting | **not in the exported headers**; no dose report (RDSR) in the export | - |

DLP is only stored in the dose report (RDSR), which is not part of these
exports.

**Decision: train without dose** (user, 2026-10-05). The reasons:

1. **Dose barely varies.** All scans are standard-dose head protocols
   (43-51 mGy). Within a scanner the spread is a few mGy, so dose cannot
   explain good versus bad here.
2. **It would act as a scanner label.** Used as a feature it would mainly
   identify the scanner, which makes the confounding of Section 3.2 worse.
   It is also missing for all photon-counting studies.
3. **It would make the analysis circular.** The project relates image
   quality to dose. If dose is an input to the quality verdict, that
   relationship becomes circular. Dose belongs in the analysis of the
   results (exports, quality-versus-dose evaluation), not in the
   classifier.

### 3.4 Train/test split

* **Method.** Stratified random split with seed 2026 (`train_classifier.py
  split`). The strata are scanner × class. Every stratum with at least 2
  studies gives one study, chosen at random, to the test set. Every stratum
  therefore keeps at least one training study, and the test set spans
  scanners and both classes.
* **Test set (4 studies, locked until the final evaluation):** one bad
  Canon study, one bad and one good Siemens ER study, one good
  photon-counting study (2 good, 2 not acceptable).
* **Training set (11 studies):** 7 good, 4 not acceptable (one bad Canon
  study, two Siemens ER studies with artefacts - motion and implant - and
  one bad photon-counting study).
* **Limitation.** The test set contains no artefact study.

## 4. Candidate features

All features come from the physical QA recipe (band 0.1-0.8 mm^-1). ROI1 is
the structured ROI (falx or septum), ROI2 the homogeneous ROI next to it
(white matter next to the falx, ventricle next to the septum).

| Name | Meaning |
|---|---|
| mtf_area | Area under the MTF over 0-0.8 mm^-1 (sharpness at the structure) |
| nps_homo | Noise variance of ROI2 (noise level in homogeneous tissue) |
| nps_struct | Power of ROI1 (structure + noise) |
| var_struct | Pixel variance of ROI1 |
| ratio_lf_hf | PS/MTF ratio, coarse half ÷ fine half of the band |
| slope_avg, loc_max | PS/MTF slope and max location; ~constant on head CT (`physical_qa_recipe.md`, Sec. 7 limitation 2) |
| NPS f_av, NPS f_peak | Noise texture of ROI2: average and peak NPS frequency (AAPM TG-233, Eq. 4-5), over 0-0.8 mm^-1. Added 2026-10-05 as candidates to replace slope and max location |

Three feature sets were compared:

* **original7:** the thesis features;
* **core5:** without slope and max location;
* **core5_texture:** core5 plus f_av and f_peak.

## 5. Analysis on the training studies

The test set was not used in any of the following.

### 5.1 Slice-level classifier, as in the legacy design: chance level

Setup:

* RBF SVM, min-max scaling, class-balanced, default settings.
* One sample per slice, falx and septum slices pooled.
* Leave-one-patient-out cross-validation: all slices of a study are held
  out together. Slices of one patient are not independent, and per-slice
  folds would overstate accuracy.

Results on 349 slices from 11 studies:

| Feature set | Studies correct | Slice balanced accuracy | Slice AUC |
|---|---|---|---|
| original7 | 4 / 11 | 0.45 | 0.42 |
| core5 | 5 / 11 | 0.48 | 0.49 |
| core5_texture | 6 / 11 | 0.50 | 0.49 |
| Scanner name only (baseline) | 6 / 11 | 0.52 | 0.36 |

No feature set beats the scanner-only baseline. A slice-level classifier
does not work on this data.

### 5.2 Study-level analysis: strong signal at the falx

Each study is represented by the median of each feature over its falx
slices (11 training studies, 7 good).

Single-feature AUC (good versus not acceptable; lower value = better
quality):

| Feature | Falx | Septum |
|---|---|---|
| var_struct | **1.00** | 0.54 |
| nps_homo | **0.93** | 0.62 |
| ratio_lf_hf | **0.89** | 0.50 |
| nps_struct | 0.75 | 0.50 |
| NPS f_peak | 0.75 | 0.58 |
| NPS f_av | 0.39 | 0.33 |
| mtf_area | 0.32 | 0.38 |

Leave-one-patient-out logistic regression (standardised features,
class-balanced) at study level:

| Model | Falx (11 studies) | Septum (10 studies) |
|---|---|---|
| nps_homo only | AUC 0.89, 8 / 11 | AUC 0.38, 6 / 10 |
| core5 | **AUC 1.00, 11 / 11** | AUC 0.00, 0 / 10 |
| core5_texture | AUC 1.00, 11 / 11 | AUC 0.00, 1 / 10 |

An AUC of 0.00 at the septum is not an inverse signal. It is a known
artefact of leave-one-out cross-validation when there is no signal at all:
holding one study out shifts the class balance of the training folds.

**Within-scanner check** (falx medians; the quality signal must hold inside
each scanner, otherwise it would only be scanner recognition):

| Scanner | Rating | var_struct | nps_homo [HU²mm²] | LF/HF |
|---|---|---|---|---|
| Canon | bad | 157.4 | 26.3 | 11.6 |
| Canon | good | 85.6 | 25.0 | 13.3 |
| Siemens ER | bad, motion artefacts | 146.9 | 32.2 | 28.8 |
| Siemens ER | bad, implant artefacts | 122.1 | 41.2 | 21.4 |
| Siemens ER | good | 100.1 | 22.9 | 10.5 |
| Siemens ER | good | 65.0 | 18.0 | 19.3 |
| Photon counting | bad | 148.3 | 25.0 | 16.3 |
| Photon counting | good | 108.0 | 20.6 | 11.2 |
| Photon counting | good | 78.7 | 25.8 | 10.7 |
| Photon counting | good | 48.1 | 18.9 | 8.7 |
| Photon counting | good | 121.1 | 21.3 | 11.0 |

Within every scanner the not-acceptable studies have the higher ROI1
variance. Not-acceptable images are noisier around the falx, which is
physically plausible. The margin is thin: the noisiest good study is at
121.1, the least noisy bad one at 122.1. The 11/11 is therefore promising,
but partly lucky.

### 5.3 Why the septum does not work, and what it is good for

Per study, the septum offers far fewer slices (~4-8 against ~25-35 for the
falx). Its ROI2 lies in the ventricle (CSF) rather than in white matter. And
its ROI1 variance is dominated by anatomy: a bright line between two dark
ventricles.

Agreement between falx and septum (Spearman ρ over the 10 training studies
that have both):

| Feature | ρ | p |
|---|---|---|
| mtf_area | **0.77** | 0.01 |
| NPS f_av | **0.71** | 0.02 |
| nps_homo | 0.60 | 0.07 |
| ratio_lf_hf | 0.41 | 0.24 |
| nps_struct | 0.28 | 0.43 |
| NPS f_peak | -0.18 | 0.61 |
| var_struct | -0.02 | 0.96 |

Properties of the scan (sharpness, noise texture) agree between the two
sites, which supports the validity of both measurements. Properties of the
local anatomy (ROI1 variance) do not.

Combining the two sites gives no gain:

* Cross-structure features (septum noise - falx noise; the mean noise of
  both sites) reach AUC 0.83-0.92, no better than the falx alone (0.93).
* Adding septum features to the falx model also gives no gain: 9/10 either
  way on the 10 studies that have both.

### 5.4 Artefacts: noise consistency across slices

Two indicators were compared per study:

* **falx/septum disagreement**, the difference between their medians;
* **scatter across falx slices**, the interquartile range divided by the
  median.

| Indicator | Artefact studies (2) | All other training studies (9) | Separates? |
|---|---|---|---|
| MTF area difference, falx vs septum | 0.034, 0.008 | 0.003-0.028 | no |
| Noise ratio, septum / falx | 0.42, 0.55 | 0.53-0.98 | partly |
| **Scatter of nps_homo across falx slices** | **0.99, 1.29** | **0.42-0.79** | **yes, clean gap** |
| Scatter of mtf_area across slices | 0.10, 0.12 | 0.07-0.13 | no |

Interpretation:

* **Plain "bad"** studies are noisy *throughout*: a high median noise, which
  the study-level model detects.
* **"Bad with artefacts"** studies are *inconsistent*: motion or metal
  corrupts some slices strongly and leaves others clean. In the motion
  study the noise rises and jumps in the upper half of the volume, which
  is consistent with patient movement during the scan.
* A good study can show isolated spikes, for example where the noise ROI
  touches a lesion. The interquartile range is insensitive to a few
  outliers, so such a study remains in the normal range (0.53).

Evidence: 2 artefact studies, with the threshold between 0.79 and 0.99.
That is too little to calibrate a probability. The test set contains no
artefact study.

Figures (local, derived from patient data):
`debugs/artefact_noise_scatter.png`, `debugs/ps_mtf_ratio_explained.png`.

### 5.5 Multiple comparisons and selection bias

Several variants (feature sets, slice versus study level, falx versus
septum, cross-structure features, artefact indicators) were examined on the
same 11 training studies. The best-looking results are therefore
optimistic. This is why:

* the design is fixed now, before any test-set evaluation;
* no further tuning is done on the training studies;
* the 4 locked test studies are evaluated exactly once, and that result is
  reported whatever it is.

## 6. Decisions

| # | Decision | Reason |
|---|---|---|
| D1 | Replace the legacy chest-CT SVM for the physical recipe; keep it for the legacy recipe | Chest-trained, Octave-scaled features, no confidence (Section 2) |
| D2 | Train without dose | Little variation, acts as a scanner label, circular (Section 3.3) |
| D3 | Classify **studies**, not slices; features = median over the study's falx slices | Slice level is at chance (5.1); study level carries the signal (5.2) |
| D4 | Verdict from the **falx** only, also when the user views the septum | The septum carries no quality signal in this data (5.2, 5.3) |
| D5 | Keep septum measurements; use falx/septum agreement on sharpness and noise texture as a plausibility check | They agree on scan properties (5.3) |
| D6 | **Logistic regression** on core5 (mtf_area, nps_homo, nps_struct, var_struct, ratio_lf_hf) | Best study-level result; 6 parameters suit 11 studies better than a flexible SVM; each coefficient is interpretable |
| D7 | Output a **probability with a 95 % interval**, from a bootstrap over training studies (model uncertainty) and over the study's slices (measurement uncertainty). Verdict **Acceptable** / **Not acceptable** / **Uncertain** (interval contains 0.5) | Every statement carries a confidence level; with 11 training studies "uncertain" must be a possible answer |
| D8 | **Reasons** = features whose 95 % interval lies outside the range of the rated-good training studies, ordered by their contribution to the logistic decision | The explanation matches how the model decides, and stays checkable |
| D9 | **Artefact indicator** = nps_homo scatter across falx slices with a bootstrap interval, compared with the good range; labelled *provisional (2 artefact studies)*; no probability. Outlier slices listed descriptively | Clear separation, but too few cases to calibrate (5.4) |
| D10 | **Model card** with patient-wise cross-validation and a one-time test, both with exact (Clopper-Pearson) 95 % intervals; data, rater and scope limits stated | Honest reporting of reliability |
| D11 | **The app measures falx and septum ROI pairs side by side** (both shown and movable at once; any single ROI can be deleted; a hand-drawn ROI can be saved as falx or septum ROI1/ROI2). This replaces the either/or structure choice. It is implemented **before** training the new model (user, 2026-10-06) | The verdict needs the falx, the plausibility check needs both, from the same run; the calculation side (`volume_qa`, batch tool) already works per structure |

### 6.1 Implementation of D11 (done 2026-10-06)

* **ROI state** lives in `presentation/viewmodels/roi_pairs.py`
  (`RoiPairStore`, Qt-free). There are four slots per slice: `falx.roi1`,
  `falx.roi2`, `septum.roi1`, `septum.roi2`. Each slot has an automatic ROI,
  an optional manual ROI (a saved drawing or a moved ROI, which wins over the
  automatic one) and an optional "deleted" mark.
* **Image (`RoiViewport`).** Both pairs are shown at once, labelled F1/F2
  (cyan, pink) and S1/S2 (green, orange), with ROI2 dotted. Clicking an ROI
  selects it, and Delete or Backspace removes it on that slice. Moving,
  deleting and drawing are locked after "Confirmed".
* **ROI box.** "Auto ROIs" has Falx and Septum checkboxes next to it. Four
  "Save drawn rectangle as" buttons (Falx/Septum ROI1/ROI2) take a drawn
  rectangle. The table has four rows; clicking a row selects that ROI. The
  box also has "Delete ROI", "Restore automatic" (drops the slice's manual
  ROIs and deletions) and "Confirmed".
* **QA (Run All).** One job per complete (slice, pair), with results stored
  per (slice, pair). On the real postop scan this gave 27 falx and 7 septum
  pairs, measured in one batch of 34 jobs taking 15.5 s.
* **Results.** A Falx/Septum switch on the per-slice tab drives the curves
  and the Run Metrics box. The study summary has one verdict line per pair
  and a Structure column. The study export lists both pairs, with a
  "Structure" column and per-structure summaries.
* **Sidebar.** It scrolls vertically instead of squeezing boxes, which had
  clipped table rows.

### 6.2 Training and verdict calculation (done 2026-10-06)

* **Verdict calculation:** `domain/results/head_ct_verdict.py` (NumPy and
  SciPy only; the app needs no scikit-learn). `evaluate_study` takes a
  study's per-slice falx values (and optionally the septum values) and
  returns the verdict with probability and interval, the per-feature
  medians with intervals and the good range, the reasons, the artefact
  indicator, outlier slices, plausibility checks and indicative per-slice
  probabilities.
* **Training:** `application/train_classifier.py fit-study` writes the model
  to `domain/results/_head_ct_model.npz` and generates the model card. The
  card's cross-validation runs the *complete* procedure per fold, including
  the bootstrap, then the app's own verdict code on the held-out study.
* **Model:** logistic regression with L2 penalty C = 1 (scikit-learn
  default) on standardised study medians, class-balanced. Some
  regularisation is required: the training classes are perfectly
  separable, and on separable data the unpenalised maximum-likelihood
  estimate does not exist; the coefficients grow without bound (Albert &
  Anderson 1984). C was not tuned, because tuning it on 11 studies would
  be one more selection on the same data (5.5).
* **Uncertainty:** 1000 bootstrap refits, resampling the training studies
  with replacement *within each class*, so that every refit sees both
  classes (stratified bootstrap; Efron & Tibshirani 1993). For a study, refit
  b is paired with resample b of its falx slices. The 95 % interval is the
  2.5-97.5 percentile of the 1000 probabilities. The seed is fixed (2026),
  so a study always gets the same result.
* **Minimum data:** no verdict below 5 usable falx slices, because a bootstrap
  over 1-4 slices cannot express the uncertainty of their median. No artefact
  indicator below 10 slices. Training studies have 20-31 falx slices.

Results (leave one patient out, complete procedure; details in the model card):

| Measure | Result | Exact 95 % CI |
|---|---|---|
| Point estimate correct (P > 0.5) | 11 / 11 | 72-100 % |
| Verdict *Uncertain* (interval contains 0.5) | 8 / 11 | - |
| Confident verdicts correct | 3 / 3 | 29-100 % |
| Scanner-only baseline | 6 / 11 | 23-83 % |

Two findings change what the app can promise:

1. **Most verdicts will be *Uncertain* for now.** The point estimates
   separate the classes, but once both sources of uncertainty are included
   most intervals contain 0.5. Splitting the two sources shows that both
   matter. The model refits matter most for the not-acceptable studies,
   because each fold has only 3-4 of them. The slice sampling matters most
   for studies whose slices vary. This is the honest state of an
   11-study model. More rated studies narrow the model part directly.
2. **The artefact indicator cannot be confirmed statistically.** The two
   artefact studies have the highest values (0.99 and 1.29, against at most
   0.79 for good studies). But the interquartile range over 25-27 slices is
   itself uncertain, and neither interval lies entirely above the good
   range (lower bounds 0.44 and 0.64). The app therefore reports two levels:
   *above the good range* (the value) and *confirmed* (the whole interval).
   With the current data, only the first level can occur for real artefact
   studies. The threshold was not changed to make the indicator fire, which
   would have been tuning on the same two studies.

### 6.3 One-time test of head-ct-1 (2026-10-06)

The 4 test studies were evaluated once with the frozen model (details in
the model card, Section 7):

| Measure | Result | Exact 95 % CI |
|---|---|---|
| Point estimate correct (P > 0.5) | 2 / 4 | 7-93 % |
| AUC of the probabilities | 0.75 | - |
| Verdict *Uncertain* | 3 / 4 | - |
| Confident verdicts correct | 1 / 1 | 3-100 % |

* Both not-acceptable test studies got a point estimate above 0.5 (0.84 and
  0.57), but their intervals contain 0.5. The verdict was therefore
  *Uncertain*, not a wrong *Acceptable*. The one confident verdict (a good
  study, *Acceptable*, 0.99 [0.86, 1.00]) is correct.
* For the Canon bad study, the noise in homogeneous tissue lies outside the
  good range and was listed as a reason. The explanation flagged what the
  probability missed.
* With 4 studies the test cannot confirm the model's discrimination
  (2/4, 95 % CI 7-93 %). It shows that the cross-validation result (11/11)
  was optimistic, as expected from 5.5. It also shows that the interval
  design works as intended: no confident wrong verdict.
* Conclusion: head-ct-1 is a **provisional** model. The app shows its output
  as decision support with the interval, never as a bare verdict. The next
  version (head-ct-2) is trained on all 15 studies plus new rated data, and
  gets a fresh held-out test set from the new data.

### 6.4 App display and exports (done 2026-10-06)

Applies to results of the **physical** recipe. Under the legacy (octave)
recipe the chest-CT SVM and its per-slice majority vote are unchanged.

* **Study summary tab.** The headline is the verdict in plain words with
  the probability ("(62 % likely acceptable)"),
  coloured green (Acceptable), red (Not acceptable) or amber (Uncertain,
  with the side of the probability: "leaning acceptable / not
  acceptable"), followed by one plain sentence on what it means (2026-10-14:
  users need not know the model; the probability, its 95 % interval and
  the decision rule moved to the top of "Details", explained in plain
  words). Below it: the reasons, each with the study median, its interval and the
  good range; the artefact indicator in its two levels (above the good
  range / confirmed); slices with noise above 3x the median; the septum
  plausibility check; the septum note; failed slices; and the provisional
  note. The verdict always comes from the falx, also when the septum is
  shown (D4). Without model file, or with fewer than 5 falx slices, the
  headline says "not available" and why.
* **Slice table and Per slice tab.** The decision value and verdict
  columns are replaced by the model's **indicative slice score**
  (P(acceptable) of the study model on that one slice) and a "noise
  outlier" note. Septum slices have no score. The score is labelled
  indicative everywhere because the model is fitted on study medians; on
  single slices the classifier is at chance level (5.1).
* **Study export (CSV and Excel).** The summary has a verdict block (verdict,
  P, 95 % CI, slices used, model version), a feature table (median, 95 %
  CI, good-range minimum and maximum, outside / reason), the artefact
  indicator, outlier slices, the plausibility checks and the provisional
  note. The per-structure averages no longer contain the legacy majority
  vote. The slice table keeps the chest-SVM columns, renamed "(legacy
  chest SVM, untrusted)", and adds the slice score and outlier flag. The
  single-slice export marks the SVM verdict the same way.
* **Code.** `domain/results/head_ct_verdict.py` (`evaluate_study_results`),
  `integration/model_files.py` (reads the model file),
  `presentation/viewmodels/head_ct_verdict_text.py` (the wording),
  `integration/results_export.py`.
* **Check on real data.** The postop photon-counting scan, measured end to
  end in the app (27 falx and 7 septum pairs), gives P(acceptable) 0.66,
  95 % CI 0.22-0.89, verdict Uncertain - the same as the batch tool and
  the test result for this study in the model card.

### 6.5 No automatic ROIs on slices with too little brain (2026-10-06)

**Problem (user observation).** The auto-ROI search also places ROIs near
the ends of the brain. There the measurement is unreliable:

* At the skull base the slice holds a small brain cross-section between the
  petrous bones. Their beam-hardening streaks raise the noise (Barrett &
  Keat 2004), and there is usually no visible falx line.
* At the vertex the slice cuts the skull tangentially, so bone partly fills
  the pixels. The slice also holds little white matter, so ROI2 lands on
  sulci and cortex instead of homogeneous tissue.
* The segmentation is least reliable on small cross-sections.

**Rule.** Automatic ROIs only on slices whose brain area (union of the
segmentation labels on that slice) is at least **20 %** of the study's
largest brain slice. Manual ROIs are allowed on every slice. The app lists
the skipped slices in the Auto ROIs status line, and the dataset export
notes them. Code: `domain/roi_selection/slice_eligibility.py`.

**Choice of 20 %.** No published value exists for this. The threshold was
chosen with a sensitivity analysis on the 15 rated studies. Dropping
slices simulates the rule exactly, because ROIs are chosen per slice.

| Minimum brain area | Falx slices dropped | Max change of P(acceptable) | Verdict changes (15 studies) |
|---|---|---|---|
| 10 % | 6 / 402 (1 %) | 0.06 | 0 |
| 20 % | 19 / 402 (5 %) | 0.10 | 0 |
| 30 % | 39 / 402 (10 %) | 0.10 | 0 |
| 40 % | 66 / 402 (16 %) | 0.11 | 2 |
| 50 % | 94 / 402 (23 %) | 0.17 | 0 |

* Falx slices below 20 % have more than twice the study's median ROI2
  noise in 26 % of cases, against 10 % for the rest. Their noise also
  scatters in both directions (0.5x to 4.7x the median), which marks an
  unreliable measurement rather than a merely noisy one.
* The slices removed by thresholds between 20 % and 40 % are not
  noisier than the rest (median 0.95-1.05x). Above 20 % the rule would
  therefore remove normal slices.
* 20 % is the lowest threshold that removes the unreliable group, and the
  verdicts are the same anywhere from 10 % to 30 %, so the exact value is
  not critical. Caveat: the threshold was chosen on the same studies that
  the model was trained on (5.5).
* Definition of brain area: the union of the labels *per slice*, rather
  than the volume's largest connected 3D region used for the brain-mask
  overlay. This way a partly segmented volume also works. On all 636
  segmented slices of the 15 studies both definitions give identical
  eligibility.
* Check on real data: the postop photon-counting scan skips slices 1-5 and
  37-43. The falx keeps 26 of 27 slices, and the verdict changes from
  P = 0.66 [0.22, 0.89] to 0.68 [0.16, 0.90], still Uncertain.

**Correction to an earlier statement.** The study medians are less robust
than assumed. Dropping 2-3 slices changed individual feature medians of
some studies by 10-48 % (e.g. ROI1 noise of one artefact study, 82 to 122),
because the per-slice values spread widely. The verdicts stayed the same.
The slice resampling in the verdict's interval (D7) reflects part of this
uncertainty.

**Model.** head-ct-1 was trained on slices before this rule. At 20 % the
probabilities of the 15 studies change by at most 0.10, with no verdict
change, so head-ct-1 stays in use. The next version (head-ct-2) is trained
with the rule in place.

Figures (local, derived from patient data): `debugs/brain_threshold.png`
(noise against brain area; probability against threshold),
`debugs/brain_threshold_slices.png` (examples of dropped slices with their
ROIs).

### 6.6 Metal rule (2026-10-21, revised 2026-10-22)

Metal streaks that reach the brain within the verdict's slice range make the study **Not assessable**,
whatever the model says (user decision; "Not acceptable" on 2026-10-21, changed because with implants CT is
often the only option and the reader decides whether the images are adequate). The model's verdict and score
stay in the details and exports. Reason: streaks disturb the measurements themselves - ROI2 avoids them, so
the implants study (rated not acceptable) scored P = 0.55. Measure, evidence (2 studies with metal) and limits:
`artefact_flags.md`, "Metal rule". The rule sits outside the model: training and the LOPO evaluation
(`train_classifier.py`) are unchanged.

## 7. Legacy versus new

| | Legacy (octave recipe) | New (physical recipe) |
|---|---|---|
| Training data | 31 thorax images, chest radiologists | 11 head-CT studies, 1 radiologist (provisional) |
| Unit of decision | slice, then majority vote | study (falx medians) |
| Model | RBF SVM, 7 features | logistic regression, 5 features |
| Output | decision value (uncalibrated distance) | probability + 95 % interval; Acceptable / Not acceptable / Uncertain |
| Explanation | none | features outside the good-reference range, with intervals |
| Artefacts | none | noise-consistency indicator (provisional) |
| Reported reliability | 93.5 % 4-fold cross-validation (thorax), no test set | patient-wise cross-validation + locked test set, exact 95 % intervals |

## 8. Limitations and data needs

Limitations:

* One reader; no inter-reader variability is known.
* 15 studies, of which 11 are for training and 4 for the test. The test
  result will have a very wide confidence interval; for example, 4/4
  correct still gives a 95 % CI of 40-100 %.
* Ratings are confounded with the scanner (Section 3.2). There is only one
  good Canon study.
* Only soft head kernels (Canon FC26, Siemens H30s, Hr44), as in the
  physical recipe's scope.
* Only 2 artefact studies, so the artefact indicator is provisional.
* Only 2 studies with metal, so the metal rule's streak threshold is provisional (6.6).

What would improve the model most:

1. More rated studies, ideally ≥ 30-50, each rated by ≥ 2 readers. The
   rating should follow defined image criteria, e.g. the European
   Guidelines on Quality Criteria for CT (EUR 16262) for the brain: sharp
   reproduction of the grey/white matter border, basal ganglia, ventricular
   system and CSF spaces. This overlaps with the human-observer study in
   WP4.
2. "Good" Canon studies and more artefact studies.
3. Photon-counting exports that include the dose report (RDSR), for the
   quality-versus-dose analysis.

## References

* Passand, Z. (2022). *Quality assessment of clinical thorax CT images.*
  Dissertation. Sections 4.8, 5.6, Table 4.2.
* Albert, A., Anderson, J. A. (1984). On the existence of maximum
  likelihood estimates in logistic regression models. *Biometrika* 71(1),
  1-10.
* Barrett, J. F., Keat, N. (2004). Artifacts in CT: recognition and
  avoidance. *RadioGraphics* 24(6), 1679-1691.
* Clopper, C. J., Pearson, E. S. (1934). The use of confidence or fiducial
  limits illustrated in the case of the binomial. *Biometrika* 26(4),
  404-413.
* Efron, B., Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.*
  Chapman & Hall.
* European Commission (1999). *European Guidelines on Quality Criteria for
  Computed Tomography*, EUR 16262 EN.
* Mitchell, M., et al. (2019). Model cards for model reporting. *Proceedings
  of FAT* '19*, 220-229.
* Samei, E., et al. (2019). Performance evaluation of computed tomography
  systems: Summary of AAPM Task Group 233. *Medical Physics* 46(11),
  e735-e756. NPS f_A and f_P.
