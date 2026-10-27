# Possible-artefact flags: metal and posterior-fossa bands

*Status: implemented 2026-10-08 (`domain/artefacts/artefact_flags.py`). Descriptive hints, not part of the
study verdict (`head_ct_classifier.md`) - except metal streaks in the brain, which make the verdict Not
assessable (metal rule, 2026-10-21/22). The band and streak thresholds are provisional.*

## Why

WP2 (§3.5 and conclusion) names the brainstem as the region radiologists use to judge artefacts: it sits
between the dense petrous bones, where beam hardening produces streaks and dark bands. The study verdict
already has a noise-consistency artefact indicator (motion, implants); it does not look at the posterior
fossa. These flags point the user to slices worth a look.

## Metal

A slice is flagged when at least 3 connected pixels exceed **2800 HU**.

* Dense cortical bone reached at most 2330 HU in our data (photon-counting CT; 1870–2330 HU across the 16
  studies). Metal reached 3071 HU (scanner ceiling) and 4224 HU in the two studies containing metal.
  2800 HU leaves a margin on both sides; a single hot pixel is ignored.
* Metal is a *source* of streaks, not proof of a bad image: one of the two studies with metal was rated good.

## Metal rule: streaks in the brain (2026-10-21)

**Problem.** The implants study (rated not acceptable, heavy streaks) was reported "Uncertain, leaning
acceptable" (P = 0.55). Its noise ROI (ROI2) avoids the streaks since the WM/GM method change of 2026-10-09, so
the measurements look better than the image.

**Decision (user).** 2026-10-21 first "Not acceptable"; changed 2026-10-22 to **Not assessable**: metal
artefacts do not make a study unacceptable as such - for patients with implants CT is often the only option
(no MRI), and whether the images answer the clinical question is the reader's call. BICTool only says that its
measurements cannot judge the study.

**Why not "metal near the brain".** The second study with metal (tumor, rated acceptable) has small
craniotomy screws 6-14 mm from the brain; the implants are 4-25 mm away. Distance does not separate them.
Whether the streaks reach the brain does.

**Measure** (per metal slice with segmentation): brain tissue = CNN labels without ventricles, subarachnoid
space, septum and dural folds, eroded 2 mm (away from the bone edge). Take the tissue within **30 mm** of the
metal and count the share that differs by more than **30 HU** from the slice's brain-tissue median. From
**5 %** the streaks reach the brain. Fewer than 20 mm² of tissue near the metal: not judged.

| Study | Metal slices | Disturbed share near the metal |
|---|---|---|
| implants (rated not acceptable) | 14-22 | 4, 8, 60, 98, -, 15, 22, 0, 38 % (slice 18: too little tissue) |
| tumor, craniotomy screws (rated acceptable) | 27-30, 33 | 0, 0, 0, 0, 0.2 % |

The same share far from the metal (> 60 mm) is 0-0.1 % on all these slices.

**Rule** (`head_ct_verdict.apply_metal_rule`, after the model): if any slice within the verdict's slice range
(first to last falx slice, failed slices in between included) has streaks in the brain, the verdict is **Not
assessable - metal artefacts in the brain (slices ...)**, without a score and in neutral colour. The model's
own result is kept and shown in the details ("Without this rule the measurements would give: ... - not
reliable here") and exported. Result: implants study Not assessable (slices 15-17, 19-20, 22); tumor study
unchanged, Acceptable (P = 0.93). No other study of the 16 contains metal.

**Limits.** Two studies with metal; the 5 % threshold lies between 0.2 % and the weakest flagged slices. A
large haemorrhage next to a clip can differ by more than 30 HU from the brain and could be mistaken for
streaks. Iterative metal-artefact reduction (iMAR etc.) lowers the share, as intended: only streaks that
remain visible count.

## Dark band across the brainstem

**Where.** Posterior-fossa slices: the cerebellum is ≥ 15 % of the CNN-labelled brain, in the skull-base half of
the scan (the CNN puts a few cerebellum pixels on some high slices). The brainstem is the CNN's *Brain
Parenchyma* piece that touches the cerebellum (within 3 mm): on medulla and pons slices this outlines the
brainstem consistently across scanners (visual check, 16 studies); higher up the label also covers unrelated
areas, and pieces thinner than 3 mm are rejected.

**What.** Beam hardening between the petrous bones shows as a dark band running left-right across the pons
(Barrett & Keat 2004). Per image row (anterior→posterior) the mean HU of the brainstem (eroded 1 mm,
0–70 HU) is taken; rows narrower than 40 % of the widest row and the outer 3 mm at the front and back are left
out (partial volume with the prepontine cistern and the 4th ventricle). **Band depth** = the largest amount
by which a row is darker than the 4 mm-smoothed profile (HU). Flag from **3.5 HU** (provisional).

**Evidence (2026-10-08).** Blinded slice ratings (none / mild / strong / unsure) of 192 posterior-fossa slices
of 16 studies by a non-clinician (the project lead); blank cells were read as "none".

* Slice level alone predicts the ratings (AUC 0.86, optimistic) — streaks sit at the petrous level. Every
  measure was therefore also compared **level-matched** (each slice minus the median of the other studies'
  slices at a similar level).
* Band depth: median 2.0 HU (none, n = 110), 3.2 HU (mild, 17), 4.1 HU (strong, 14). Level-matched AUC
  0.75 for any streak, 0.80 for strong (p < 0.001, permutation); 0.78 / 0.82 without the two most doubtful
  rating units.
* Direction control: the same measure front-to-back (same region and noise) does not follow the ratings
  (AUC 0.48–0.51) — the agreement is specific to left-right bands.
* Per study the signal is weaker (AUC 0.69, 8 studies with a strong rating vs 8 without).
* Measures that did **not** work: brainstem noise (artefact index √(SD²brainstem − SD²thalamus)),
  directionality of the band-passed posterior fossa (Radon), texture SD, an oriented dark-line detector
  (its apparent success came from the cerebellum rim next to the skull).

**Confidence: medium.** Five measures were tried and the chosen one was refined once after looking at its false
alarms (the cistern tips) — principled but post hoc. The reference is a non-clinician's; the measure may
see bands a non-expert reader misses (two such slices looked plausible on review). Before the band depth can
enter a verdict (head-ct-2), the 3.5 HU threshold needs confirming on new data rated by a radiologist
(e.g. the WP4 reader study).

## In the app and exports

* Results → Study summary, under the verdict: "Artefacts (metal): on slices … - streaks reach the brain on
  slices …" (or "- no streaks measured in the brain") and "Artefacts (posterior fossa): possible dark band
  across the brainstem on slice(s) …" (both recipes); the per-slice band depths and streak shares are in the
  details. Study table and Per-slice view: "metal" or "metal streaks in the brain".
* With the metal rule the verdict headline reads "Study verdict: Not assessable - metal artefacts in the brain
  (slices …)".
* Study export: block "Possible artefacts (hints, not part of the verdict)" with metal slices and highest HU,
  streak slices and the share per metal slice, band slices and the band depth per posterior-fossa slice;
  verdict block row "Metal rule" with the model's own verdict. Dataset export (`batch_qa`): columns "Metal
  slices" and "Brainstem dark band slices (provisional)", once per study; "Metal rule: streaks in the brain on
  slices" and "Model verdict before the metal rule" on the falx row.

## Not pursued (tested 2026-10-08)

* **CSF attenuation consistency** (WP2 §3.1): no robust difference between good, bad and artefact studies
  once partial volume is controlled; intraventricular blood mimics the skull-base effect.
* **Basal ganglia from the CNN labels**: the CNN keeps labelling the lentiform nucleus when its image contrast
  is removed (12 of 16 studies) — a basal-ganglia visibility measure must measure contrast in the image and
  use the label only for the location.

## References

* Barrett JF, Keat N. Artifacts in CT: recognition and avoidance. RadioGraphics 2004;24:1679–1691.
