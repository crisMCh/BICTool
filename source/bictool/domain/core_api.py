"""Core API for the Domain Layer.

The single entry point the (not yet implemented) Application Layer is meant
to call into: given already-cropped ROI arrays for one slice, run every
quality metric and return a structured result. A flat namespace of Domain
entry points, not a class hierarchy - there is currently exactly one
capability to expose. DICOM loading, ROI selection, and result export are
Integration/Presentation-layer concerns and are not part of this API; see
`bictool.domain.results.study_quality` for what is and is not in scope.
"""

from bictool.domain.results.study_quality import SliceQualityResult, evaluate_slice_quality

__all__ = ["evaluate_slice_quality", "SliceQualityResult"]
