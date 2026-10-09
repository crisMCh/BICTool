# Possible-artefact flags: metal and posterior-fossa bands

*Status: implemented 2026-10-08 (`domain/artefacts/artefact_flags.py`). Descriptive hints only — not part of
the study verdict (`head_ct_classifier.md`). The band threshold is provisional.*

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

* Results → Study summary, under the verdict: "Artefacts (metal): on slices …" and "Artefacts (posterior
  fossa): possible dark band across the brainstem on slice(s) …" (both recipes); the per-slice band depths are
  in the details.
* Study export: block "Possible artefacts (hints, not part of the verdict)" with metal slices and highest HU,
  band slices and the band depth per posterior-fossa slice. Dataset export (`batch_qa`): columns "Metal
  slices" and "Brainstem dark band slices (provisional)", once per study.

## Not pursued (tested 2026-10-08)

* **CSF attenuation consistency** (WP2 §3.1): no robust difference between good, bad and artefact studies
  once partial volume is controlled; intraventricular blood mimics the skull-base effect.
* **Basal ganglia from the CNN labels**: the CNN keeps labelling the lentiform nucleus when its image contrast
  is removed (12 of 16 studies) — a basal-ganglia visibility measure must measure contrast in the image and
  use the label only for the location.

## References

* Barrett JF, Keat N. Artifacts in CT: recognition and avoidance. RadioGraphics 2004;24:1679–1691.
