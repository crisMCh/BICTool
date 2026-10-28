"""Text of Help > About BICTool (rich text for a message box). Qt-free."""

from __future__ import annotations

from pathlib import Path

from bictool import __version__

CNN_CITATION = (
    "Cai JC, Akkus Z, Philbrick KA, et al. Fully Automated Segmentation of Neuroanatomy on Head CT Using "
    "Deep Learning. Radiol Artif Intell. 2020;2(5):e190183."
)


def about_text(notices: Path) -> str:
    return (
        f"<h3>BICTool {__version__}</h3>"
        "<p>Objective, automated image-quality assessment of head CT.<br>"
        "Developed for the German Federal Office for Radiation Protection (BfS).</p>"
        "<p><b>Decision support only</b> - not a medical device and not for diagnosis. "
        "The study verdict comes from a provisional model.</p>"
        "<p>Free software under the GNU General Public License v3 or later.</p>"
        "<p>Anatomy segmentation: network and weights of the Mayo Clinic Radiology Informatics Lab "
        f"(github.com/jasonccai/HeadCTSegmentation).<br>{CNN_CITATION}</p>"
        f"<p>Third-party components and licenses: <a href=\"{notices.as_uri()}\">{notices.name}</a></p>"
    )
