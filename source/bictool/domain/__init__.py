"""Domain / Business Layer: core image-analysis logic.

Organized into sub-modules matching the architecture diagram:
`preprocessing`, `segmentation`, `metrics`, `results`.

Translated from a legacy Octave implementation (`reference/octave/`), not
imported wholesale - see `reference/octave/TRANSLATION_STATUS.md` for what's
ported vs pending. `core_api.py` exposes the Core API the Application Layer
is meant to call into.
"""
