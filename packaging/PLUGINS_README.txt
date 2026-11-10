BICTool plug-ins
================

Plug-ins add values (metrics) or WM/GM methods to BICTool. Copy a plug-in -
a .py file or a folder with an __init__.py - into this folder and restart
BICTool. Settings > Plug-ins lists the plug-ins found, their status, and
switches them on or off.

- Plug-in values are descriptive: they are shown on the Results page and
  exported, but never change the study verdict.
- Plug-ins run inside BICTool with full access to the images. Only use
  plug-ins from people you trust.
- A plug-in here can use Python's standard library, NumPy, SciPy and BICTool.

examples/ holds two example plug-ins (not loaded from there): roi_stats.py
(extra metrics) and otsu_wmgm.py (a WM/GM method). To try one, copy it into
this folder. How to write plug-ins: docs/plugins.md in the BICTool source.
