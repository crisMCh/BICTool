# PyInstaller spec of the portable Windows build - run through build_portable.py.
# One folder (not one file): starts fast and the CNN weights sit next to it.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPECPATH).parent
sys.path.insert(0, str(ROOT / "source"))

datas = collect_data_files("bictool", includes=["**/*.npz", "**/*.png", "**/*.svg", "**/*.ico"])

# A Conda-based Python keeps the DLLs of its standard-library extensions
# (_ctypes, _ssl, pyexpat, _lzma, _bz2) in Library\bin, where PyInstaller does
# not look - without them scipy fails to import in the packaged app.
CONDA_BIN = Path(sys.base_prefix) / "Library" / "bin"
CONDA_DLLS = ["ffi-8.dll", "libcrypto-3-x64.dll", "libssl-3-x64.dll", "libexpat.dll", "liblzma.dll", "libbz2.dll"]
binaries = [(str(CONDA_BIN / name), ".") for name in CONDA_DLLS if (CONDA_BIN / name).is_file()]

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT / "source")],
    binaries=binaries,
    datas=datas,
    # Not used by the app: training, tests, plotting, other DICOM stacks.
    excludes=["sklearn", "matplotlib", "pytest", "IPython", "tkinter", "SimpleITK", "pynetdicom", "skimage"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BICTool",
    icon=str(ROOT / "source" / "bictool" / "presentation" / "resources" / "app_icon.ico"),
    console=False,
    version=str(ROOT / "packaging" / "build" / "version_info.txt"),
)
coll = COLLECT(exe, a.binaries, a.datas, name="BICTool")
