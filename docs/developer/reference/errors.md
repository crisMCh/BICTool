# Reference: Errors

Every error the API raises on purpose derives from `api.BictoolError` and from the matching built-in error, so
`except ValueError` keeps working. Catch `api.BictoolError` for "this study could not be analysed"; any other
exception is a programming error.

| Error | Also a | Raised when |
|---|---|---|
| [`BictoolError`](#bictoolerror) | `Exception` | Base class of all below. |
| [`SeriesError`](#serieserror) | `ValueError` | The study folder can't be used. |
| [`ModelNotAvailable`](#modelnotavailable) | `FileNotFoundError` | The anatomy CNN can't be loaded. |
| [`NotSegmented`](#notsegmented) | `RuntimeError` | ROIs are requested before the segmentation. |
| [`InvalidSegmentation`](#invalidsegmentation) | `ValueError` | A segmentation or cache does not fit the series. |
| [`InvalidRois`](#invalidrois) | `ValueError` | ROIs can't be measured. |
| [`PluginFailed`](#pluginfailed) | `RuntimeError` | A plug-in export format is missing or failed. |
| [`Cancelled`](#cancelled) | - | `cancel()` returned `True`. |

Plug-in hooks that fail during a study do **not** raise: their error is recorded in `PluginValues.error` or
`PluginHints.error` and the study goes on.

### `BictoolError`

*Since 0.1.* Base class.

### `SeriesError`

*Since 0.1.* The folder does not exist, holds no usable DICOM series, holds several studies ("Several studies
under ... - point at one study's folder"; the patients are not named), or the series number does not exist.
Raised by `list_series`, `open_study`, `analyse_study`.

### `ModelNotAvailable`

*Since 0.1.* No weights configured, the weights file is missing, or TensorFlow is not installed. Raised by
`load_anatomy_model`, `Study.segment`, `analyse_study`.

### `NotSegmented`

*Since 0.1.* `Study.place_rois` or `Study.measure` before `Study.segment` / `set_segmentation`.

### `InvalidSegmentation`

*Since 0.1.* `Study.set_segmentation` with an unknown slice or masks of the wrong shape; a segmentation cache of
another series (`cache_file`).

### `InvalidRois`

*Since 0.1.* Raised by `Study.measure` before anything is measured: ROIs placed on another series, or an ROI not
in whole pixels, outside the image, empty or not square. The message names up to five slices and slots.

### `PluginFailed`

*Since 0.3.* `write_export` with a format that is not available, or whose plug-in raised; the message contains
the plug-in's error.

### `Cancelled`

*Since 0.1.* The `cancel` callback returned `True`; the step stopped between two slices. Finished slices are
kept (a new `segment` call continues).
