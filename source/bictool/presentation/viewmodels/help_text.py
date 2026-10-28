"""User-facing help: one-line instructions per workflow step and the hover
help of every control. Kept in one place (not scattered over the View) so
the wording stays consistent and can be reused for the user manual.

`TOOLTIPS` keys are the main window's attribute names; `"name:key"` means
item `key` of a dict attribute (e.g. one of the "Save drawn rectangle as"
buttons). A control that is disabled also shows why (added by the View).
"""

from __future__ import annotations

STEP_HINTS = {
    "segmentation": "Runs automatically when a study is opened. Tick an overlay to check the result on the image.",
    "roi": (
        "Place the ROIs automatically and check them on a few slices: drag to move, right-click to delete or "
        "save as falx, white matter, septum or ventricle. Draw your own with a drag on the image. Then confirm."
    ),
    "confirm": "Confirmed ROIs are locked and used for the QA run. Untick to edit them again.",
    "qa": "Measures sharpness, noise and noise texture on every confirmed ROI pair. Results open in the Results tab.",
}

TAB_TOOLTIPS = {
    "per_slice": "Curves and measurements of one slice at a time - scroll through the slices with the mouse wheel.",
    "summary": "The study verdict, and every measured slice in one table.",
}

TOOLTIPS = {
    # Boxes and sub-boxes (hover the title or an empty spot inside)
    "_series_box": "Which image series of the study is analysed.",
    "_step1_section": (
        "Step 1 - finds the anatomy (falx, septum, ventricles, ...) and separates white from grey matter. "
        "Both are needed to place the ROIs."
    ),
    "_anatomical_box": "AI segmentation of 16 brain structures (U-Net). Runs automatically after loading a study.",
    "_wmgm_box": (
        "Separates white matter, grey matter and CSF by their CT numbers. "
        "The noise ROI (ROI2) next to the falx is placed in white matter."
    ),
    "_roi_section": (
        "Step 2 - the measurement regions: ROI1 on a thin structure (falx or septum) for sharpness, "
        "ROI2 in homogeneous tissue next to it for noise."
    ),
    "_confirm_box": "Locks the ROIs, so the QA run measures exactly what you checked.",
    "_metrics_box": (
        "Step 3 - runs the image-quality measurements. On the Results page this box shows the measurements "
        "of the shown slice, or the study averages."
    ),
    "_trace_box": "Technical log for troubleshooting (Settings > Show Trace Log).",
    # Page switch, viewer
    "_viewer_tab_btn": "The image viewer: segmentation overlays, ROIs and the QA run.",
    "_results_tab_btn": "The QA results: per slice, and the study summary with its verdict.",
    "_help_btn": (
        "Opens the help: the user manual and the technical reference (algorithms, parameters). "
        "F1 opens the help for the control under the mouse."
    ),
    "_window_preset_combo": "Grey-level window for viewing only - does not change any measurement.",
    "_slice_slider": "Move through the slices (or use the mouse wheel over the image).",
    "_series_combo": "The image series being analysed. BICTool picks the axial ~4 mm head series; choose another here.",
    # 1. Segmentation
    "_segmentation_status_label": "Progress of the automatic anatomy segmentation.",
    "_cancel_segment_volume_btn": "Stop the running segmentation (slices done so far are kept).",
    "_select_segments_btn": "Choose which anatomical structures are drawn on the image, and how opaque.",
    "_show_segments_cb": "Show or hide the selected anatomical structures on the image.",
    "_brain_tissue_method_combo": (
        "How white and grey matter are separated (needed to place the noise ROI). "
        "Adaptive Thresholding works on most scans (lightly smoothed, HU range from this study's basal ganglia - "
        "see its settings); try another method if the overlay looks wrong. "
        "Anchored (prototype): uses the basal ganglia as gray/white references and corrects the HU drift near "
        "the skull and between slices - not validated yet; it can mislabel large areas as gray matter."
    ),
    "_brain_tissue_settings_btn": "Fine-tune the selected separation method.",
    "_white_matter_cb": "Show the white matter found on this slice.",
    "_gray_matter_cb": "Show the grey matter found on this slice.",
    "_csf_cb": "Show the cerebrospinal fluid (CSF) found on this slice.",
    "_brain_mask_cb": "Show the brain outline the separation works inside (skull removed).",
    # 2. ROI selection
    "_auto_rois_btn": (
        "Places two ROI pairs on every suitable slice: at the falx (ROI1) with white matter next to it (ROI2), "
        "and at the septum pellucidum with the ventricle next to it. Slices with too little brain are skipped."
    ),
    "_roi_details_btn": "Show or hide the position and size of every ROI on this slice.",
    "_roi_table": "The ROIs on this slice. Click a row to select that ROI on the image.",
    "_delete_roi_btn": "Remove the selected ROI, or the rectangle you drew, from this slice (or press Del).",
    "_restore_auto_btn": "Undo your moves and deletions on this slice and bring the automatic ROIs back.",
    "_confirm_rois_cb": "Lock all ROIs and enable the QA run. Untick to edit them again.",
    # 3. QA run
    "_run_all_btn": "Measure every confirmed ROI pair on every slice. Takes about a second per pair.",
    "_cancel_run_btn": "Stop the QA run (finished slices are kept).",
    "_metrics_table": "The measurements of the shown slice - or, on the Study summary tab, the study averages.",
    # Results
    "_results_prev_btn": "Previous slice with a result.",
    "_results_next_btn": "Next slice with a result.",
    "_results_slider": "Move through the slices with results (or use the mouse wheel).",
    "_results_pair_buttons:falx": "Show the falx ROI pair's result on this slice.",
    "_results_pair_buttons:septum": "Show the septum ROI pair's result on this slice.",
    "_mtf_chart_view": "Sharpness (MTF) measured at the edge of the structure: the higher the curve, the sharper.",
    "_nps_chart_view": "Noise power spectrum in homogeneous tissue: how strong the noise is and how fine or coarse.",
    "_ps_chart_view": "Power spectrum at the structure: image detail plus noise.",
    "_ratio_chart_view": "Power spectrum divided by sharpness: high values at low frequencies mean structure, not noise.",
    "_curves_view_btn": "The four result curves of this slice (MTF, NPS, PS, PS/MTF).",
    "_steps_view_btn": (
        "How this slice was measured, instead of the four curves: the edge the MTF used, the edge "
        "profiles, the MTF of every edge, both ROIs, the 2D spectra, the PS/MTF band and the feature values."
    ),
    "_study_table": "Every measured slice and ROI pair. Click a row to open that slice.",
    "_verdict_details_btn": "Show the measurement details behind the verdict (intervals, reference ranges, checks).",
    "_study_provisional_label": "",  # set from the model note
    # Menus
    "_open_folder_action": "Open a folder with a DICOM series (the usual way to load a study).",
    "_open_file_action": "Open a single DICOM image.",
    "_close_study_action": "Close the study and clear all results.",
    "_reload_series_action": "Load the selected series again from disk.",
    "_view_tags_action": "Show the DICOM header of the current image.",
    "_segment_slice_action": "Segment only the current slice again.",
    "_segment_volume_action": "Segment the whole series again.",
    "_custom_window_action": "Set your own grey-level window (viewing only).",
    "_export_action": "Save the results as Excel or CSV (no patient names or IDs are written).",
    "_show_trace_action": "Show the technical log at the bottom of the window (for troubleshooting).",
}
