"""Presentation Layer (MVVM): see `views` (View) and `viewmodels` (ViewModel) subpackages.

GUI framework: PySide6 (decided 2026-09-10 - see architecture/layers_description.md,
"User Interface Framework"). LGPL licensing (compatible with FR-08's plugin API without
forcing plugin authors into GPL) and active Qt6 maintenance were prioritized over legacy
Windows 7/8 support, which Qt6 does not support (requires Windows 10 1809+). Fallback if
that turns out to be a hard requirement: downgrade to PySide2 (Qt5, also LGPL).
"""
