"""View layer: GUI widgets/windows.

Renders application state and forwards user interaction to the ViewModel layer.
GUI toolkit: PySide6 (see `bictool.presentation` docstring for the decision rationale).

`main_window.MainWindowView` is the first real view: a structural port of
`design/bictool_console_mockup.html` (single fixed window, no page navigation -
see that file's docstring/comments for the layout rationale). It now has real
end-to-end wiring for a first bare-functionality test: local DICOM import of
whole series/volumes (`integration.dicom_file_loader.load_dicom_volume`),
slice navigation via a slider or mouse-wheel scroll over the viewport, manual
ROI selection by mouse drag (`roi_viewport.RoiViewport` + `viewmodels.roi_selection`,
re-done per slice), and a real `Worker`/`Scheduler` dispatch of
`evaluate_slice_quality` on the currently displayed slice. Still out of
scope: automatic/anatomical segmentation (FR-02, the Segmentation checkboxes
are inert), export, and chart rendering.
"""
