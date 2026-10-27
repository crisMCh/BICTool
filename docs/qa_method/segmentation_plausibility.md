# Segmentation check: label fragmentation (provisional)

Added 2026-10-21. Code: `domain/segmentation/segmentation_plausibility.py`, wording in
`presentation/viewmodels/segmentation_plausibility_text.py`. Exploration data and pictures:
`debugs/seg_plausibility/` (gitignored).

## Purpose

The anatomy CNN (Mayo U-Net) was trained on normal-looking brains and Alzheimer's disease. When it struggles
(lesions, artefacts, noise), BICTool's automatic ROIs and masks, which are built on its labels, can be wrong.

BICTool is not a diagnostic tool. The check therefore says only **whether** the user should look at the
segmentation, once per study, not why and not slice by slice.

## Measure

**Label fragmentation:**
- **Per slice:** for every label, count the pieces of at least 20 mm² beyond its largest piece, and sum over
  the labels. Smaller specks are ignored; the threshold is in mm², so it does not depend on the pixel size.
- **Study value:** the median over the slices with enough brain (≥ 20 % of the largest brain slice, the
  automatic-ROI rule).
- **Decision:** above **30** the segmentation may be unreliable (provisional).

**Reported, but not used for the decision:**
- the 3 most fragmented slices (where to look);
- the number of "lobe islands", i.e. pieces of one lobe enclosed by another.

## Evidence (16 studies: 15 rated + 1 unrated stroke study)

| Study | Fragmentation | Check |
|---|---|---|
| motion artefacts | 39.5 | may be unreliable |
| acute SDH | 38.0 | may be unreliable |
| stroke (unrated) | 35.0 | may be unreliable |
| implants | 34.0 | may be unreliable |
| normal AS+ "19 yrs" | 25.0 | consistent |
| other AS+ and Canon studies | 20–21 | consistent |
| photon-counting studies | 9–21 | consistent |

In the four flagged studies the lobes visibly break into patchworks of labels
(`debugs/seg_plausibility/worst_slices.png`). Clean segmentations consist of a few large regions per slice.

## What else was tried and dropped

**Label sizes** (each label's share of the brain):
- The shares follow anatomy and age, e.g. subarachnoid space 4.5 % in young and 21 % in atrophic brains,
  ventricles 0.5–8.6 %.
- They do not separate badly from well segmented studies, and would flag older patients.

**Anatomical order and lobe islands** (frontal before parietal before occipital; a lobe piece inside another
lobe):
- These do catch real confusion on single slices.
- They also occur in normal studies, so they are noisy and only reported.

**CNN uncertainty** (softmax output, recorded with every segmentation since 2026-10-20):
- It is scanner-dependent: all AS+ studies are higher, normal ones included.
- The CNN is often confidently wrong: a large postoperative cavity labelled as brain, an infarct labelled as CSF.
- It is not used for the check.

**Lesion and artefact signs per slice** (CSF-like, blood-like, air, metal; 2026-10-20): dropped on 2026-10-21.
They were too specific, leaned towards diagnosis, and a reason per slice can confuse users.

## Where it shows

- "Anatomical" segmentation box, after the volume segmentation:
  - "Segmentation looks consistent." (grey);
  - "⚠ Segmentation may be unreliable in this study – check the overlays." (amber).
- Study summary: one line. Under Details: the value, the limit, the most fragmented slices and the lobe islands.
- Exports:
  - study summary block;
  - two columns in the dataset export (fragmentation; may be unreliable yes/no).
- Batch tool: computed for every study.

## Limits

- **Small evidence base:** 16 studies, and "unreliable" was judged by eye.
- **Missed failure mode:** a segmentation that is wrong but compact. For example, the cavity in the
  thalamic-bleed study is labelled as one large brain region; fragmentation is 20 there, so it is not flagged.
- **Scanner tendency:** AS+ studies tend to be more fragmented; normal 19 yrs reaches 25. More studies per
  scanner would allow a check of whether the limit should depend on the scanner or kernel.
