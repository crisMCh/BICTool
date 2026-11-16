BICTool plug-ins
================

Plug-ins add values (metrics, also from ROIs of their own), possible-artefact
hints or WM/GM methods to BICTool. Copy a plug-in -
a .py file or a folder with an __init__.py - into this folder and restart
BICTool. Settings > Plug-ins lists the plug-ins found, their status, and
switches them on or off.

- Plug-in values are descriptive: they are shown on the Results page and
  exported, but never change the study verdict.
- Plug-ins run inside BICTool with full access to the images. Only use
  plug-ins from people you trust.
- A plug-in here can use Python's standard library, NumPy, SciPy and BICTool
  - nothing else. Packages installed into a Python on this computer (e.g.
  PyTorch, scikit-image) are not visible to BICTool.exe.

examples/ holds three example plug-ins (not loaded from there): roi_stats.py
(extra metrics), wm_box.py (a metric with its own ROI) and otsu_wmgm.py (a
WM/GM method). To try one, copy it into
this folder. How to write plug-ins: the BICTool Developer Guide (manual folder, or
docs/developer in the BICTool source).
