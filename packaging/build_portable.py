"""Builds the portable Windows package: packaging/dist/BICTool-<version>-win64.zip.

    .venv/Scripts/python.exe packaging/build_portable.py [--weights PATH] [--skip-pyinstaller]

Steps: PyInstaller (bictool.spec) -> one folder with BICTool.exe; then the CNN
anatomy weights (models/), LICENSE, third-party notices with the license texts
of every bundled Python distribution, the manuals and help pages (if built,
docs/manual/build_manual.py) and a short README are copied in, and the folder
is zipped.

The CNN weights are third-party (Cai et al., Mayo Clinic RIL) and their source
states no license - they are bundled for internal use while redistribution
permission is requested (2026-10-27). Remove them before any external release
unless permission has arrived (packaging/THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata as metadata
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging"
sys.path.insert(0, str(ROOT / "source"))

from bictool import __version__  # noqa: E402
from bictool.infrastructure.app_paths import HELP_INDEX_FILE  # noqa: E402
from bictool.infrastructure.config import BUNDLED_ANATOMY_WEIGHTS, load_config  # noqa: E402

BUILD = PACKAGING / "build"
DIST = PACKAGING / "dist"
APP = DIST / "BICTool"

_VERSION_INFO = """VSVersionInfo(
  ffi=FixedFileInfo(filevers=({v}, 0), prodvers=({v}, 0), mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1,
                    subtype=0x0, date=(0, 0)),
  kids=[StringFileInfo([StringTable('040904B0', [
    StringStruct('CompanyName', 'BICTool project for the German Federal Office for Radiation Protection (BfS)'),
    StringStruct('FileDescription', 'BICTool - image-quality assessment of head CT'),
    StringStruct('FileVersion', '{dotted}'),
    StringStruct('LegalCopyright', 'GPL-3.0-or-later'),
    StringStruct('ProductName', 'BICTool'),
    StringStruct('ProductVersion', '{dotted}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])]
)
"""


def _write_version_info() -> None:
    parts = [int(p) for p in __version__.split(".")] + [0] * 4
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "version_info.txt").write_text(
        _VERSION_INFO.format(v=", ".join(map(str, parts[:3])), dotted=__version__), encoding="utf-8"
    )


def _bundled_distributions() -> list[str]:
    """Python distributions whose top-level packages ended up in the build: the folders in
    _internal (extension modules, data) plus the pure-Python modules in PyInstaller's archive."""
    internal = APP / "_internal"
    names = {path.name.split(".")[0] for path in internal.iterdir()}
    archive_toc = BUILD / "pyinstaller" / "bictool" / "PYZ-00.toc"
    _archive, modules = ast.literal_eval(archive_toc.read_text(encoding="utf-8"))
    names |= {module.split(".")[0] for module, _source, _kind in modules}
    owners = metadata.packages_distributions()
    return sorted({dist for name in names for dist in owners.get(name, [])} - {"bictool"}, key=str.lower)


def _copy_license_texts() -> list[tuple[str, str, str]]:
    """Each bundled distribution's license files into licenses/<name>/; returns (name, version, license)."""
    target = APP / "licenses"
    rows = []
    for name in _bundled_distributions():
        dist = metadata.distribution(name)
        files = [f for f in dist.files or [] if any(key in f.name.upper() for key in ("LICENSE", "COPYING", "NOTICE"))]
        for file in files:
            source = Path(dist.locate_file(file))
            if source.is_file():
                (target / name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target / name / source.name)
        license_name = dist.metadata.get("License-Expression") or dist.metadata.get("License") or ""
        if not license_name or len(license_name) > 60:
            classifiers = [c.split("::")[-1].strip() for c in dist.metadata.get_all("Classifier") or [] if c.startswith("License")]
            license_name = ", ".join(classifiers) or "see licenses folder"
        rows.append((name, dist.version, license_name))
    return rows


def build(weights: Path, *, skip_pyinstaller: bool = False) -> Path:
    """`skip_pyinstaller`: reuse the program folder of the last build and only refresh the
    files around it (manuals, notices, README) and the ZIP."""
    if not skip_pyinstaller:
        _write_version_info()
        if APP.exists():
            shutil.rmtree(APP)
        subprocess.run(
            [sys.executable, "-m", "PyInstaller", str(PACKAGING / "bictool.spec"), "--noconfirm",
             "--distpath", str(DIST), "--workpath", str(BUILD / "pyinstaller")],
            check=True,
        )
    elif not (APP / "BICTool.exe").is_file():
        sys.exit("No previous build to reuse - run without --skip-pyinstaller")
    for folder in ("licenses", "manual", "help"):
        if (APP / folder).exists():
            shutil.rmtree(APP / folder)
    (APP / BUNDLED_ANATOMY_WEIGHTS).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, APP / BUNDLED_ANATOMY_WEIGHTS)
    shutil.copy2(ROOT / "LICENSE", APP / "LICENSE.txt")
    rows = _copy_license_texts()
    notices = (PACKAGING / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    table = "\n".join(f"| {name} | {version} | {license_name} |" for name, version, license_name in rows)
    (APP / "THIRD_PARTY_NOTICES.txt").write_text(notices.replace("{{PYTHON_PACKAGES}}", table), encoding="utf-8")
    shutil.copy2(PACKAGING / "README.txt", APP / "README.txt")
    manuals = ROOT / "docs" / "manual" / "build"
    for pdf in sorted(manuals.glob("*.pdf")) if manuals.is_dir() else []:
        (APP / "manual").mkdir(exist_ok=True)
        shutil.copy2(pdf, APP / "manual" / pdf.name)
    if (manuals / "help" / HELP_INDEX_FILE).is_file():
        shutil.copytree(manuals / "help", APP / "help")
    else:
        print("Warning: help pages not built (docs/manual/build_manual.py) - the help window will be empty")

    archive = DIST / f"BICTool-{__version__}-win64.zip"
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(APP.rglob("*")):
            zf.write(path, Path("BICTool") / path.relative_to(APP))
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, help="anatomy CNN weights (default: the configured path)")
    parser.add_argument("--skip-pyinstaller", action="store_true", help="reuse the last program build")
    args = parser.parse_args()
    weights = args.weights or Path(load_config().get("segmentation", {}).get("anatomy_weights_path", ""))
    if not weights.is_file():
        sys.exit(f"CNN weights not found: {weights} - pass --weights")
    archive = build(weights, skip_pyinstaller=args.skip_pyinstaller)
    print(f"Built {archive} ({archive.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
