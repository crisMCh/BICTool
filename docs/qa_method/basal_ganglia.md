# Basal-ganglia visibility

*Status: implemented 2026-10-09 (`domain/metrics/basal_ganglia.py`). Descriptive and **provisional** — not part
of the study verdict. Candidate feature for head-ct-2, to be pre-specified and tested on new rated data.*

## Why

WP2 (§3.3, §5.2) names the basal ganglia as a low-contrast but diagnostically important region: an early
stroke shows as a fading border between the basal ganglia and the white matter. WP2 asked to first determine
whether the structures are visible, then assess contrast with intensity profiles or statistics.

## The CNN label only says where to look

The anatomy CNN labels the lentiform and caudate nuclei, but it **fills them in from shape and position**:
with the lentiform contrast removed from the image, it still labelled the lentiform nucleus in 12 of 16
studies (the same test on the ventricles made their label disappear). A label is therefore no evidence of
visibility — visibility is measured in the image.

It also draws nuclei **far from where they are**: near the vertex and at the skull base (e.g. 6 vertex slices in
the motion-artefact study). The nuclei are one compact structure in the centre of the brain, so only the
**anatomical block** is measured (added 2026-10-20):

* per slice, the deep grey matter = CNN lentiform + caudate + thalamus + internal capsule (area);
* start at the slice with the most deep grey matter and extend up and down while the next slice keeps at least
  **8 %** of that (provisional); the first slice below it, or a slice without segmentation, ends the block;
* the thalamus is included because it lies at the same height and keeps the block together where the CNN
  under-labels the nuclei on one slice.

On the 15 rated studies the block is 8–13 slices (about 3–6 cm) per study. Excluded were only isolated labels
separated from the block (in 8 studies, 1–6 slices each, all near the vertex or skull base).

## Measure

Per slice of the anatomical block with labelled nuclei and internal capsule:

* **Border contrast** of each nucleus: mean HU 2 px inside minus 2 px outside the label border (0–60 HU only),
  divided by the pooled SD of the nucleus and the internal capsule (both eroded 1 px). Slice value = mean over
  the labelled lentiform and caudate nuclei.
* **Plausibility (provisional):** nucleus minus internal capsule must be 2–15 HU; otherwise the label sits on
  something else and the slice counts as not visible ("label not plausible").
* **Words (provisional cut points, between the rating medians):** < 0.25 not visible, 0.25–0.6 hardly
  visible, 0.6–0.85 visible, ≥ 0.85 clearly visible.

## Evidence (2026-10-09)

**Reference:** blinded slice ratings (clear / visible / hardly / not visible / unsure) of the lentiform and
caudate nuclei, 191 slices of 16 studies (157 usable), by the project lead — **not a clinician**; the
ratings reflect experience on this project. Four candidate measures were fixed in a script *before* the
ratings were read.

| Measure | Spearman | AUC visible+ vs not | AUC clear vs hardly | Beyond label area* |
|---|---|---|---|---|
| Label area (baseline) | 0.64 | 0.96 | 0.64 | — |
| HU difference to internal capsule | 0.37 | 0.76 | 0.52 | 0.26 |
| CNR (smoothed) | 0.49 | 0.81 | 0.72 | 0.40 |
| **Border contrast** | **0.63** | 0.89 | **0.85** | **0.57** |

\*partial rank correlation with the rating, controlling for the label area.

* The **label area** answers "is this slice at basal-ganglia level?" (anatomy). The **border contrast** adds
  "how clearly visible", the image-quality part: medians 0.07 / 0.43 / 0.75 / 0.88 for not visible /
  hardly / visible / clear.
* **Plausibility:** on slices rated visible or clear, the nucleus was 4.6–8.6 HU denser than the internal
  capsule (5–95 %, max 10.6). Labels with 22–40 HU sat on metal streaks (a study with implants) and next to
  blood (a study with an acute subdural haematoma). With the limit, the implemented measure reaches Spearman
  0.68, AUC 0.92 (visible+ vs not), 0.85 (clear vs hardly).
* **Per study it is weak** (Spearman 0.29 between the best-slice ratings and measures) — pathology dominates
  (e.g. a thalamic bleed: rated low, measured high). No study score is given.

## Limits

* One non-expert rater, 16 studies, three scanners; every cut point and the plausibility range are provisional.
* **Caudate next to the ventricle:** on upper slices the outer ring partly contains CSF, so the border measures
  caudate vs CSF — visibility can come out one level too high there. Starting the outer ring at 15 HU fixed
  this on the inspected study but lowered the overall agreement (Spearman 0.63, beyond-area 0.53), so the
  pre-specified 0–60 HU was kept.
* Labels scattered in small fragments (seen in a study with motion artefacts) are not caught by the
  plausibility limit; a label-shape check is an untested idea.
* Pathology (infarct, bleed) changes visibility for clinical, not technical, reasons.

## In the app and exports

* Results → Study summary: "Basal ganglia (provisional): visible on slices … (clearly on …)", and a line for
  labels without matching contrast; per-slice values in the details.
* Study table Note column and the Per-slice view: "basal ganglia visible" etc.
* Study export: block "Basal ganglia visibility (provisional, not part of the verdict)" with border contrast,
  HU differences and plausibility per slice; per-slice table column; dataset export (`batch_qa`): visible and
  not-plausible slices per study.
