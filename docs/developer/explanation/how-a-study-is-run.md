# How BICTool runs a study

This page explains the steps between a study folder and a result, and where the API and the plug-ins come in.
The algorithms themselves are described in the *Technical Reference*.

## One pipeline, three ways in

The program, the Python API and the batch tool run the same steps (in `bictool/application/pipeline.py`), so
a series gives the same numbers whichever way it is run. Tests check this, and so did comparisons on real
studies.

```text
 study folder
     │
 1. Series ......... list_series / open_study: CT, axial, closest to 4 mm
     │
 2. Segmentation ... Study.segment: anatomy CNN, 16 structures per slice (1-2 s per slice)
     │
 3. WM/GM .......... white matter, grey matter, CSF inside the brain mask  ◄── plug-in WM/GM methods
     │
 4. ROIs ........... Study.place_rois: falx pair and septum pair per slice
     │                (you may change them: RoiPairStore)
 5. Measurement .... Study.measure: per ROI pair MTF, NPS, contrast        ──► bictool_pair_metrics
     │                                                                      ──► bictool_slice_metrics
 6. Study level .... hints (artefacts, basal ganglia, segmentation check)   ──► bictool_study_metrics
     │                verdict (physical recipe)                             ──► bictool_artefact_hints
     │
 StudyResult ....... JSON, the program's Results page and exports           ──► export formats
```

## The steps

1. **Series.** A study folder holds several series. BICTool chooses CT, axial, closest to 4 mm; you can choose
   another with `series=`.
2. **Segmentation.** A U-Net segments 16 anatomical structures on every slice (falx, septum, ventricles, basal
   ganglia, lobes, ...). It works on an internally re-oriented copy of the series; all masks are mapped back to
   the DICOM pixel grid before anything else uses them.
3. **WM/GM.** Inside the brain mask, a WM/GM method separates white matter, grey matter and CSF. BICTool uses
   it for one thing: the falx pair's ROI2 goes into white matter.
4. **ROIs.** On every slice with at least 20 % of the largest brain area, BICTool places two square ROI pairs:
   ROI1 on the falx or the septum (for sharpness), ROI2 next to it in homogeneous tissue (for noise).
5. **Measurement.** For each pair: the MTF at the structure's edges, the noise power spectrum of both ROIs, the
   PS/MTF ratio and the contrast - the values of a [`Measurement`](../reference/results.md#measurement).
6. **Study level.** The verdict model combines the falx medians into the head-CT verdict (physical recipe,
   from 5 falx slices on); the hints describe possible artefacts, the basal-ganglia visibility and the
   segmentation's plausibility.

## Where plug-ins come in

Plug-ins **add** to steps 3, 5, 6 and the export; they never replace a step or change the verdict:

| Plug-in kind | Step | What BICTool does with it |
|---|---|---|
| WM/GM method | 3 | Offers it as a method; if selected, its white matter decides where the falx ROI2 goes. |
| Pair metrics | 5 | Shows and exports the values next to its own. |
| Slice metrics, own ROIs | after 5 | Shows values and ROIs, exports them. |
| Study metrics, artefact hints | 6 | Shows them on the Study summary, exports them. |
| Export formats | after 6 | Offers them in the Export dialog, API and batch tool. |

Why plug-ins cannot replace the ROI placement, the segmentation or the verdict, and how they are kept apart
from BICTool's own results, is explained in [The plug-in system](plugin-system.md).

## In the program

The program runs the same steps, but in the background while you work: the segmentation starts when a study is
opened, the WM/GM masks are computed for the slice you look at, the ROIs appear as soon as their slices are
segmented, and Run QA measures the confirmed ROIs one pair after the other. The plug-ins' slice, study and hint
hooks run in one more background job when the last pair is measured.
