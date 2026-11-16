"""Domain / Business Layer: core image-analysis logic.

Organized into sub-modules matching the architecture diagram:
`preprocessing`, `segmentation`, `metrics`, `results`.

Translated from a legacy Octave implementation (`reference/octave/`), not
imported wholesale - see `reference/octave/TRANSLATION_STATUS.md` for what's
ported vs pending. The pipeline that strings these functions together, and
the public API, is `bictool.api`.
"""
