"""On-disk cache of a volume's anatomy segmentation (used by
`application/batch_qa.py`), so a rerun over a dataset skips the CNN.

Masks are mutually exclusive (one class per pixel), so each slice is stored
as one label map; plain arrays, no pickles. Since 2026-10-20 the CNN's
per-slice confidence (`SegmentationConfidence`) is stored too; older files
simply have none (`load_confidence` returns {}). Files are derived from patient
data - keep the cache folder out of version control.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from bictool.domain.segmentation.segmentation_confidence import SegmentationConfidence


def cache_path(cache_dir: Path, group: str, name: str, slice_paths: list[Path]) -> Path:
    """Per study and series - a different series choice never reuses a stale segmentation."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{group}__{name}").strip("_")
    series = hashlib.sha1("|".join(str(path) for path in slice_paths).encode("utf-8")).hexdigest()[:10]
    return Path(cache_dir) / f"{safe}__{series}.npz"


def save_segmentation(
    path: Path,
    masks_by_slice: dict[int, dict[str, np.ndarray]],
    confidence_by_slice: dict[int, SegmentationConfidence] | None = None,
) -> None:
    names = sorted({name for masks in masks_by_slice.values() for name in masks})
    indices = sorted(masks_by_slice)
    labels = []
    for index in indices:
        label_map = np.zeros(next(iter(masks_by_slice[index].values())).shape, dtype=np.uint8)
        for code, name in enumerate(names, start=1):
            mask = masks_by_slice[index].get(name)
            if mask is not None:
                label_map[mask] = code
        labels.append(label_map)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if confidence_by_slice:
        conf_indices = sorted(confidence_by_slice)
        extra = {
            "confidence_indices": np.array(conf_indices),
            "confidence": np.array(
                [[c.mean_entropy, c.low_fraction, c.labelled_pixels] for c in (confidence_by_slice[i] for i in conf_indices)],
                dtype=np.float64,
            ),
        }
    np.savez_compressed(path, indices=np.array(indices), names=np.array(names), labels=np.array(labels), **extra)


def load_segmentation(path: Path) -> dict[int, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        names = [str(name) for name in data["names"]]
        return {
            int(index): {name: label_map == code for code, name in enumerate(names, start=1)}
            for index, label_map in zip(data["indices"], data["labels"])
        }


def load_confidence(path: Path) -> dict[int, SegmentationConfidence]:
    """The stored CNN confidence per slice; {} for caches written before it was stored."""
    with np.load(path, allow_pickle=False) as data:
        if "confidence" not in data.files:
            return {}
        return {
            int(index): SegmentationConfidence(float(row[0]), float(row[1]), int(row[2]))
            for index, row in zip(data["confidence_indices"], data["confidence"])
        }
