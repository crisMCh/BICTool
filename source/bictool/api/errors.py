"""The errors the API raises - all derive from `BictoolError`, and each also
from the matching built-in error (so `except ValueError` keeps working)."""

from __future__ import annotations


class BictoolError(Exception):
    """Base class of every error the BICTool API raises on purpose."""


class SeriesError(BictoolError, ValueError):
    """The folder holds no usable DICOM series, several studies, or not the requested series."""


class ModelNotAvailable(BictoolError, FileNotFoundError):
    """The anatomy CNN can't be loaded: no weights configured, file missing, or TensorFlow not installed."""


class NotSegmented(BictoolError, RuntimeError):
    """ROIs need the anatomy segmentation - run `Study.segment` (or `set_segmentation`) first."""


class InvalidSegmentation(BictoolError, ValueError):
    """A segmentation (or segmentation cache) does not fit the series."""


class InvalidRois(BictoolError, ValueError):
    """ROIs that can't be measured: unknown slice, outside the image, not square, or placed on another series."""


class Cancelled(BictoolError):
    """`cancel()` returned True - the step stopped between two slices; finished slices are kept."""
