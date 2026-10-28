<div class="title-page" markdown="1">

# BICTool User Manual

<p class="sub">Objective, automated image-quality assessment of head CT<br>
Version {{VERSION}}</p>

<p class="sub">Developed for the German Federal Office for Radiation Protection (BfS)</p>

<div class="note" markdown="1">
**Decision support only.** BICTool is not a medical device and is not meant for diagnosis. Its study
verdict comes from a provisional model and must always be checked by a qualified person.
</div>

</div>

<div class="contents" markdown="1">

[TOC]

</div>

## 1. What BICTool does {#what-it-does}

BICTool measures the image quality of non-contrast head CT studies directly in the patient images. No
phantom scan is needed. For every suitable slice it:

1. finds the anatomy with an AI segmentation (falx, septum pellucidum, ventricles, lobes, basal ganglia and
   more) and separates white matter, grey matter and CSF by their CT numbers;
2. places two pairs of measurement regions (ROIs): one on a thin structure (the falx cerebri or the septum
   pellucidum) to measure **sharpness**, and one in homogeneous tissue next to it to measure **noise**;
3. computes sharpness (MTF), noise (noise power spectrum, NPS) and the balance of structure and noise;
4. combines the slices into a **study verdict** - *Acceptable*, *Not acceptable*, *Uncertain* or *Not
   assessable* - and points out slices and findings worth a look (possible artefacts, basal-ganglia
   visibility, a check of the segmentation).

All results can be exported as an Excel report or as CSV tables. Exports never contain patient names or IDs.

## 2. Installation and start {#installation}

### 2.1 Requirements {#requirements}

| Item | Requirement |
|---|---|
| Operating system | Windows 10 (version 1809 or newer) or Windows 11, 64-bit |
| Memory | 8 GB RAM recommended; BICTool itself uses about 1.5 GB |
| Disk space | about 3 GB for the unpacked program |
| Rights | no administrator rights needed |

### 2.2 Installation {#install}

BICTool comes as a ZIP file (`BICTool-{{VERSION}}-win64.zip`).

1. Unpack the ZIP file into a folder of your choice, for example `C:\Users\<name>\BICTool`. Unpack the whole
   archive - do not run the program from inside the ZIP.
2. Start `BICTool.exe` in the unpacked folder (a desktop shortcut can be created with right-click > *Send to >
   Desktop*).
3. The first time, Windows may show *"Windows protected your PC"*, because the program is not signed. Click
   *More info* and then *Run anyway*.

The folder must stay together: `BICTool.exe` needs the folders `_internal` (program libraries) and `models`
(the segmentation network). To remove BICTool, delete the folder.

### 2.3 Checking the installation {#self-test}

The program can check itself on one DICOM series without the user interface. Open a command prompt in the
BICTool folder and run:

```
start /wait "" BICTool.exe --self-test "D:\path\to\a\head CT series"
```

It segments the series, measures it and computes the study verdict, which takes about two minutes. The
result (*SELF-TEST PASSED* or *SELF-TEST FAILED* with the reason) is written to the log file (section 8.2);
read it before starting BICTool again, because every start begins a new log.

## 3. The main window {#main-window}

<figure markdown="1">
![Main window with a study loaded](images/01_viewer.png)
<figcaption>Figure 1 - Main window: image viewer (left) and the sidebar with the three work steps (right).</figcaption>
</figure>

| Area | Content |
|---|---|
| Menu bar | *File* (open), *Study* (close, reload), *View* (DICOM tags), *Tools* (segmentation, method settings, custom window), *Settings* (trace log, text size), *Export*, *Help* (manuals, about) |
| Viewer / Results | The two pages of the window, switched with the buttons at the top left |
| Viewer | The CT slice with overlays and ROIs; grey-level window preset (*Brain*, *Subdural*, *Bone*, *Stroke*, *Soft tissue*); slice slider. The window only changes the display, never a measurement |
| Sidebar | Series, then the three steps: *1. Segmentation*, *2. ROI Selection*, *3. Run QA metrics* |

Hover over any control to see what it does. A disabled control tells you in its tooltip why it is disabled
(for example *"Open a study first"*).

**Help window.** Press **F1** to open the help for the control under the mouse pointer, or click the
**? Help** button at the top of the window. The help window contains this manual (English or German) and the
*Technical Reference*, which explains every algorithm and parameter. It has a table of contents, a full-text
search and *Back* / *Forward* buttons, and it can stay open next to BICTool. *Help > Help Contents* opens it
as well; the PDF versions are in the *Help* menu and in the folder `manual`.

## 4. Working with BICTool {#working}

### 4.1 Opening a study (input) {#open-study}

Choose *File > Open DICOM Folder (Series)…* and select the folder that contains the CT series. BICTool reads
all DICOM files in it, groups them into series and selects the axial head series with about 4 mm slices. If
the study has several series, choose another one in the series list of the sidebar.

**Suitable input:**

- non-contrast CT of the head, axial slices, DICOM format;
- soft-tissue (brain) reconstruction kernels, as used for routine head CT (for example Siemens Hr44/H30s,
  Canon FC26). Sharp bone kernels are outside the scope of the measurement;
- any matrix size and pixel spacing.

*File > Open DICOM File (Single Slice)…* opens one image only; the study verdict needs several slices.

### 4.2 Step 1 - Segmentation {#segmentation}

The anatomical segmentation starts automatically when a study is opened and takes about 1-2 seconds per
slice. The progress is shown in the *Anatomical* box.

<figure markdown="1">
![Anatomical segmentation overlay](images/02_segmentation.png){ .image }
<figcaption>Figure 2 - Anatomical segmentation shown on the image (Show Segments).</figcaption>
</figure>

<figure markdown="1">
![Step 1 in the sidebar](images/02a_step1_box.png){ .box }
<figcaption>Figure 3 - Step 1 in the sidebar: anatomical segmentation and WM/GM separation.</figcaption>
</figure>

- **Show Segments** draws the structures on the image; **Select Segments…** chooses which structures are
  shown and how opaque.
- After the whole series is segmented, a line in the box says whether the segmentation **looks consistent**
  or **may be unreliable**. In the second case, look at the overlays on a few slices: lesions, artefacts or
  strong noise can make the AI break the anatomy into scattered pieces, and then the automatic ROIs can be
  wrong. The check does not say why.

**WM/GM separation.** The ROIs for noise are placed in white matter. White and grey matter are separated by
their CT numbers with one of these methods (box *WM/GM separation*, list *Method*):

| Method | In short |
|---|---|
| Adaptive Thresholding (default) | Local threshold, lightly smoothed, HU range taken from this study's basal ganglia. Works on most scans |
| Fuzzy C-Means | Clusters the brain into three classes (CSF, white, grey matter) |
| HU Thresholding | Fixed HU ranges per tissue |
| Anchored (prototype) | Uses the basal ganglia as references and corrects HU drift; not validated yet |

<figure markdown="1">
![White and grey matter overlay](images/03_wmgm.png){ .image }
<figcaption>Figure 4 - White matter (WM) and grey matter (GM) found by the selected method.</figcaption>
</figure>

<div class="note" markdown="1">
**Choose the method that fits the scan best.** No method works best on every scan. Tick *WM* and *GM*,
look at a few slices and try another method if the white matter does not look right. The noise ROIs are
placed in the white matter the method finds, so the choice affects the measurement.
</div>

*CSF* shows the cerebrospinal fluid and *Brain Mask* the brain outline the separation works inside.

Technical details: *Technical Reference*, [chapters 4-6](tr:tr-segmentation) (segmentation, brain mask, [WM/GM methods](tr:tr-wmgm)).

### 4.3 Step 2 - ROI selection {#roi-selection}

Click **Start auto ROIs**. BICTool places two ROI pairs on every suitable slice:

| Pair | ROI1 (sharpness) | ROI2 (noise) |
|---|---|---|
| Falx | on the falx cerebri | white matter next to it |
| Septum | on the septum pellucidum | the ventricle next to it |

Slices with too little brain (near the skull base and the vertex) get no automatic ROIs.

<figure markdown="1">
![Automatic ROIs](images/04_auto_rois.png){ .image }
<figcaption>Figure 5 - Automatic ROIs on one slice: F1 on the falx, F2 white matter, S1 on the septum, S2 ventricle.</figcaption>
</figure>

<figure markdown="1">
![ROI box](images/04a_roi_box.png){ .box }
<figcaption>Figure 6 - Step 2 in the sidebar; ROI details show position and size.</figcaption>
</figure>

**Check the ROIs on a few slices** (slider or mouse wheel):

- drag an ROI to move it;
- right-click an ROI to delete it or to save it as falx, white matter, septum or ventricle ROI;
- drag on the image to draw your own rectangle, then save it with a right-click;
- **Delete ROI** removes the selected ROI (or press *Del*); **Restore automatic** brings back the automatic
  ROIs of this slice.

When the ROIs are right, tick **ROIs checked - lock them for QA** in the *Confirm ROIs* box. The ROIs are
then locked; untick to edit them again.

<figure markdown="1">
![Confirmed ROIs](images/05_confirmed.png){ .box }
<figcaption>Figure 7 - ROIs confirmed and locked.</figcaption>
</figure>

Technical details: *Technical Reference*, [chapter 7](tr:tr-rois) (how the ROIs are placed).

### 4.4 Step 3 - QA run {#qa-run}

Click **▶ Run QA**. BICTool measures every confirmed ROI pair on every slice (about a second per pair)
and opens the *Results* page when it is done. A running measurement can be stopped; finished slices are kept.

<figure markdown="1">
![Run QA](images/05a_run_box.png){ .box }
<figcaption>Figure 8 - Step 3: Run QA, with the measurements of the shown slice.</figcaption>
</figure>

Technical details: *Technical Reference*, [chapter 8](tr:tr-measurements) (MTF, NPS, PS/MTF).

### 4.5 Results per slice {#results-per-slice}

<figure markdown="1">
![Results of one slice](images/06_results_curves.png)
<figcaption>Figure 9 - Results page, tab Per slice: the four curves of one slice.</figcaption>
</figure>

The *Per slice* tab shows one slice at a time (slider, arrows or mouse wheel); *Falx* / *Septum* switches
between the two ROI pairs. The four curves:

| Curve | Meaning |
|---|---|
| MTF | Sharpness measured at the edge of the structure: the higher the curve, the sharper |
| NPS | Noise power spectrum in homogeneous tissue: how strong the noise is and how fine or coarse |
| PS | Power spectrum at the structure: image detail plus noise |
| PS/MTF | Power spectrum divided by sharpness: high values at low frequencies mean structure, not noise |

The line *"This slice resembles: …"* is only a hint: the model judges whole studies, not single slices.

*Show: Intermediate steps (advanced)* replaces the curves with the steps of the measurement - the edge used
for the MTF, the edge profiles, both ROIs, the 2D spectra and the feature values. It is meant for checking
a single measurement in detail.

<figure markdown="1">
![Intermediate steps](images/07_results_steps.png)
<figcaption>Figure 10 - Intermediate steps of one measurement.</figcaption>
</figure>

### 4.6 Study summary and verdict {#study-summary}

<figure markdown="1">
![Study summary](images/08_study_summary.png)
<figcaption>Figure 11 - Study summary: the verdict, the findings and every measured slice in one table.</figcaption>
</figure>

The **study verdict** is the result for the whole study. It compares five measurements (medians over the
falx slices) with studies that a radiologist rated:

| Verdict | Meaning |
|---|---|
| **Acceptable** (green) | The measurements clearly match the studies rated acceptable |
| **Not acceptable** (red) | The measurements clearly match the studies rated not acceptable |
| **Uncertain, leaning acceptable / not acceptable** | No clear answer; the measurements point slightly to one side. Check the images yourself |
| **Not assessable - metal artefacts in the brain** | Metal streaks reach the brain. They disturb the measurements, so BICTool cannot judge the study. Whether the images are good enough is for the reader to decide - with metal implants CT is often the only option |
| *Not available* | Fewer than 5 falx slices could be measured |

The percentage (*"90 % likely acceptable"*) is the model's estimate. It is repeated many times with slightly
different selections of the rated studies and of this study's slices; the verdict is *Acceptable* or *Not
acceptable* only if the middle 95 % of these repetitions all lie on the same side of 50 %. **Details** shows every measurement with its interval
and the range of good studies.

<figure markdown="1">
![Verdict details](images/09_summary_details.png)
<figcaption>Figure 12 - Details behind the verdict.</figcaption>
</figure>

Below the verdict, plain lines name what to look at:

| Line | Meaning |
|---|---|
| *Sharpness / Noise … is higher or lower than in good studies* | The measurements that push the verdict towards *not acceptable* |
| *Artefacts (noise across slices)* | The noise changes strongly from slice to slice (motion, implants) |
| *Artefacts (metal)* | Metal on these slices; whether its streaks reach the brain |
| *Artefacts (posterior fossa)* | A possible dark band across the brainstem (beam hardening between the petrous bones) |
| *Basal ganglia (provisional)* | On which slices the basal ganglia are visible against the internal capsule |
| *Segmentation (provisional)* | Whether the anatomical segmentation looks consistent |
| *Unusually noisy slices* | Slices with noise far above the rest of the study |
| *Check the ROIs* | The falx and septum pairs disagree - look at the ROIs |

The findings marked *provisional* are hints where to look; they do not change the verdict (except metal
streaks in the brain). The table below lists every measured slice; click a row to open that slice.

Technical details: *Technical Reference*, [chapter 9](tr:tr-verdict) (verdict model) and [chapter 10](tr:tr-findings) (findings).

### 4.7 Export (output) {#export}

Choose *Export > Export Results…* (enabled after a QA run).

<figure markdown="1">
![Export dialog](images/10_export_dialog.png){ .box }
<figcaption>Figure 13 - Export: scope and formats; the file name is chosen in the next step.</figcaption>
</figure>

1. **Scope:** the *current slice* or the *whole study*.
2. **Formats** (any combination):
    - *Report - Excel workbook (.xlsx)*: a readable report. Sheet *Summary*: study, verdict and findings as
       shown in the program, measurements with units, technical details below. Sheet *Per slice*: one row per
     slice and ROI pair, filterable.
    - *All numbers - CSV tables (.csv)*: the complete tables for analysis in other programs (plain text).
    - *Charts - JPG images* (current slice only): the four curves.
3. Click **Save as…** and choose the file name and folder. BICTool suggests a name such as
   *BICTool study 2026-03-12 Head native 4.0 Hr40*; all files of one export get this name (for example
   *… - summary.csv*, *… - per slice.csv*, *… - MTF.jpg*). The last folder is remembered.

Exports contain only de-identified fields: no patient name, ID or birth date.

## 5. The measurements {#measurements}

| Measurement | Unit | What it says |
|---|---|---|
| Sharpness (MTF area) | 1/mm | Area under the modulation transfer function at the edge of the falx or septum. Higher = sharper |
| Noise (NPS variance) | HU²·mm² | Strength of the noise in homogeneous tissue (ROI2), from the noise power spectrum |
| Noise texture (f_av, f_peak) | 1/mm | Mean and peak frequency of the noise: fine or coarse grain |
| Structure-to-noise (PS/MTF ratio, LF/HF) | - | Low- versus high-frequency content of the structure spectrum, corrected for sharpness |
| Contrast (RMS) | - | Standard deviation divided by the mean of ROI1: how much the structure stands out |

The study verdict uses five study-level values: MTF area, noise in homogeneous tissue, noise at the falx,
variance at the falx and the PS/MTF ratio. How every value is computed - formulas, parameters, limits and
references - is described in the *Technical Reference* (help window, F1).

Technical details: *Technical Reference*, [chapter 8](tr:tr-measurements) and the [parameter list](tr:tr-parameters).

## 6. Settings and parameters {#settings}

### 6.1 WM/GM separation methods {#wmgm-methods}

Each method has its own settings (*Tools > … Settings* or the ⚙ button next to the method). *Restore
Defaults* returns to the tested values; change them only if the overlay is clearly wrong on a scan.

| Method | Parameters |
|---|---|
| Adaptive Thresholding | Local window size (px); plausible HU range (low, high) - used when the study's own range is switched off or not available; minimum region size (px); smoothing (mm, 0 = off); use the study's own HU range |
| Fuzzy C-Means | Fuzziness m; maximum iterations; spatial blending weight; minimum region size (px) |
| HU Thresholding | Skull-strip cutoffs for air and bone (HU); HU ranges of CSF, white and grey matter; smoothing (median filter size, px); minimum region size (px) |

Technical details: *Technical Reference*, [chapter 6](tr:tr-wmgm).

### 6.2 Viewing {#viewing}

- **Window preset** (viewer toolbar): *Brain* (default), *Subdural*, *Bone*, *Stroke*, *Soft tissue*.
- **Tools > Custom Window…**: your own window width and level (HU). Viewing only.
- **View > DICOM Tags…**: the DICOM header of the current image.

### 6.3 Program settings {#program-settings}

- **Settings > Text Size…**: size of all text in BICTool (80-200 %, default 125 %). Applies after a restart.

<figure markdown="1">
![Text size](images/11_text_size.png){ .box }
<figcaption>Figure 14 - Text size.</figcaption>
</figure>

- **Settings > Show Trace Log**: a technical log at the bottom of the window, for troubleshooting.
- Window layout, opened sections, text size and the last export folder are remembered per Windows user
  (file `%APPDATA%\BfS\BICTool.ini`).
- *Tools > Advanced > QA Recipe* contains a legacy calculation that only reproduces results of the earlier
  Octave version. Keep the default *Physical*.

## 7. Limitations {#limitations}

- **Decision support only.** The verdict comes from a provisional model trained on studies rated by one
  radiologist; it has not yet been confirmed on independent data. Always look at the images.
- The verdict is meant for routine non-contrast head CT with soft-tissue kernels. Other protocols, contrast
  studies, children and sharp kernels are outside its scope.
- Pathology on or next to the falx (bleeding, tumour, surgery) can disturb the sharpness measurement. Check
  the ROIs on such slices.
- The AI segmentation was trained on adult brains. Large lesions, strong artefacts or unusual anatomy can
  make it wrong; the segmentation check points to such studies.
- The provisional findings (brainstem band, basal ganglia, segmentation check, metal streaks) use preliminary
  limits and will be refined with more data.

## 8. Troubleshooting {#troubleshooting}

### 8.1 Problems and solutions {#problems}

| Problem | Cause and solution |
|---|---|
| *"Windows protected your PC"* at the first start | The program is not signed. Click *More info > Run anyway* |
| The program does not start, or closes at once | Check that the whole ZIP was unpacked and the folders `_internal` and `models` are next to `BICTool.exe`. Look at the log file (8.2) |
| *"Anatomy segmentation weights not found"* | The file `models\weights_mayoiNPH.hdf5` is missing. Unpack the ZIP again |
| Segmentation is very slow | About 1-2 s per slice is normal; it runs once per study. The speed depends on the processor. Close other programs that use much memory or processor time |
| No series or the wrong series is selected | Choose the series in the sidebar's series list. Only DICOM files are read |
| *Start auto ROIs* is disabled | The segmentation is not finished yet - wait until the *Anatomical* box shows that the volume is segmented |
| Few or no automatic ROIs | The falx or septum is not visible or not segmented on those slices, or the slices have too little brain. Draw ROIs by hand |
| ROI2 lies in grey matter or CSF | Choose another WM/GM method (4.2) and restart the auto ROIs, or move ROI2 by hand |
| *Run QA* is disabled | Tick *ROIs checked - lock them for QA* first |
| Some slices *failed* | The sharpness edge could not be measured there (for example no clear falx). The other slices are used; the verdict needs at least 5 falx slices |
| *"Segmentation may be unreliable"* | Look at the overlays; correct the ROIs by hand where needed |
| Export fails | The folder is not writable or a file with that name is open in Excel. Close it or choose another folder |
| Text too small or too large | *Settings > Text Size…* and restart |

### 8.2 Log file {#log-file}

BICTool writes a log of the last session to

```
%LOCALAPPDATA%\BICTool\logs\bictool.log
```

(paste this into the Windows Explorer address bar). When you report a problem, please send this file. It
contains no patient names or IDs, but it can contain folder names - check them before sending the file outside
your organisation. *Settings > Show Trace Log* shows the same information inside the program.

## 9. License and acknowledgements {#license}

BICTool is free software under the GNU General Public License version 3 or later (`LICENSE.txt`). The
third-party components and their licenses are listed in `THIRD_PARTY_NOTICES.txt` and the folder `licenses`.

The anatomical segmentation uses the network and weights of the Mayo Clinic Radiology Informatics Lab
(github.com/jasonccai/HeadCTSegmentation):
Cai JC, Akkus Z, Philbrick KA, et al. *Fully Automated Segmentation of Neuroanatomy on Head CT Using Deep
Learning.* Radiol Artif Intell. 2020;2(5):e190183.
