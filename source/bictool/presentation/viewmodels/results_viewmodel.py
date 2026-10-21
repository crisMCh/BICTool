"""ResultsViewModel: what the Results page shows.

Built from the QA run's results (`QaRunViewModel`): the slices that have a
result (the Per-slice slider's positions), which ROI pair is shown
(Falx/Septum switch), the study-level head-CT verdict (physical recipe) or
the legacy majority vote (Octave recipe), the possible-artefact flags (metal,
posterior-fossa bands - hints, not part of the verdict), the study table's
rows, and the texts of one slice's view. Colours are a view decision - the ViewModel
gives tones ("good", "bad", "neutral"). QtCore only - no widgets.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from PySide6.QtCore import QObject, Signal

from bictool.domain.artefacts.artefact_flags import ArtefactFlags, evaluate_artefact_flags
from bictool.domain.metrics.basal_ganglia import BasalGangliaSlice, evaluate_basal_ganglia
from bictool.domain.results.head_ct_verdict import evaluate_study_results
from bictool.domain.results.study_quality import RECIPE_PHYSICAL
from bictool.domain.results.study_summary import ACCEPTABLE, UNACCEPTABLE, study_verdict
from bictool.presentation.viewmodels.artefact_text import artefact_basics, artefact_details, slice_artefact_notes
from bictool.presentation.viewmodels.basal_ganglia_text import (
    basal_ganglia_basics,
    basal_ganglia_details,
    basal_ganglia_note,
)
from bictool.presentation.viewmodels.head_ct_verdict_text import (
    OUTLIER_TEXT,
    PROVISIONAL_NOTE,
    SEPTUM_NOTE,
    TONE_BAD,
    TONE_GOOD,
    TONE_NEUTRAL,
    VerdictText,
    is_outlier,
    resembles,
    slice_hint,
    slice_score,
    verdict_text,
)
from bictool.presentation.viewmodels.roi_pairs import FALX, PAIR_LABELS, PAIRS, SEPTUM
from bictool.presentation.viewmodels.segmentation_plausibility_text import plausibility_basics, plausibility_details
from bictool.presentation.viewmodels.slice_steps import SliceStepsView, slice_steps_view

STUDY_TABLE_COLUMNS = [
    "Slice", "Structure", "MTF area [1/mm]", "NPS ROI2 [HU²mm²]", "PS/MTF LF/HF", "Decision", "Verdict",
]
# Physical recipe: the last two columns show the study model's per-slice view
# instead of the legacy chest-CT SVM (docs/qa_method/head_ct_classifier.md).
STUDY_TABLE_COLUMNS_PHYSICAL = [*STUDY_TABLE_COLUMNS[:5], "Resembles (hint)", "Note"]


@dataclass(frozen=True)
class TableRow:
    job: tuple[int, str]
    cells: list[str]
    last_tone: str          # tone of the last cell
    hint_tooltip: str = ""  # physical recipe: tooltip of the "Resembles" cell


@dataclass(frozen=True)
class SliceView:
    pair: str
    position: int           # slider position
    count: int              # slices with a result
    available_pairs: tuple[str, ...]
    verdict_text: str
    verdict_tooltip: str
    verdict_tone: str
    note_text: str          # outlier note (physical) or decision value (legacy)


class ResultsViewModel(QObject):
    changed = Signal()  # results were (re)built - redraw the Results page

    def __init__(
        self,
        qa,
        model: Callable[[], object],
        parent: QObject | None = None,
        *,
        volume: Callable[[], object] | None = None,
        segmentation_cache: Callable[[], dict] | None = None,
        segmentation_plausibility: Callable[[], object] | None = None,
    ) -> None:
        """`qa` = the QaRunViewModel; `model()` = the head-CT study model (or None);
        `volume()` / `segmentation_cache()` = the shown series and its anatomy
        results, for the artefact flags (no flags without them)."""
        super().__init__(parent)
        self._qa = qa
        self._model = model
        self._volume = volume
        self._segmentation_cache = segmentation_cache
        self._segmentation_plausibility = segmentation_plausibility  # () -> SegmentationPlausibility | None
        self.artefacts: ArtefactFlags | None = None
        self.basal_ganglia: dict[int, BasalGangliaSlice] | None = None  # slice -> visibility (provisional)
        self.slices: list[int] = []
        self.pair = FALX
        self.verdict = None  # HeadCtVerdict of the physical recipe, else None
        self.table_jobs: list[tuple[int, str]] = []
        # Rebuilt whenever the QA results are complete or dropped.
        qa.finished.connect(lambda _cancelled: self.refresh())
        qa.reset_done.connect(self.refresh)

    # ---- building ----

    def refresh(self) -> None:
        results = self._qa.results
        self.slices = sorted({index for index, _pair in results})
        self.verdict = (
            evaluate_study_results(self._model(), self._qa.results_by_pair(FALX), self._qa.results_by_pair(SEPTUM))
            if self.is_physical()
            else None
        )
        self.table_jobs = sorted(results, key=lambda job: (job[0], PAIRS.index(job[1])))
        self.artefacts, self.basal_ganglia = self._image_hints() if results else (None, None)
        self.changed.emit()

    def _image_hints(self) -> tuple[ArtefactFlags | None, dict[int, BasalGangliaSlice] | None]:
        """Possible artefacts and basal-ganglia visibility of the shown series."""
        volume = self._volume() if self._volume else None
        if volume is None:
            return None, None
        cache = self._segmentation_cache() if self._segmentation_cache else {}
        masks = {index: result.masks for index, result in cache.items()}
        artefacts = evaluate_artefact_flags(masks, volume.get_slice_hu, volume.num_slices, volume.pixel_spacing_mm)
        return artefacts, evaluate_basal_ganglia(masks, volume.get_slice_hu)

    @property
    def plausibility(self):
        """The study-level segmentation check of the shown series (None until it ran)."""
        return self._segmentation_plausibility() if self._segmentation_plausibility else None

    def plausibility_lines(self) -> list[str]:
        return plausibility_basics(self.plausibility)

    def artefact_lines(self) -> list[str]:
        """Plain lines about possible artefacts (metal, posterior-fossa bands)."""
        return artefact_basics(self.artefacts)

    def basal_ganglia_lines(self) -> list[str]:
        """Plain lines about the basal-ganglia visibility (provisional)."""
        return basal_ganglia_basics(self.basal_ganglia)

    def is_physical(self) -> bool:
        return self._qa.is_physical()

    def select_pair(self, pair: str) -> None:
        self.pair = pair

    # ---- the Per-slice tab ----

    def shown_pair(self, index: int) -> str | None:
        """The pair shown for slice `index`: the switch's choice if it has a
        result there, else the other one (None without any result)."""
        candidates = [self.pair] + [pair for pair in PAIRS if pair != self.pair]
        return next((pair for pair in candidates if (index, pair) in self._qa.results), None)

    def slice_view(self, index: int) -> SliceView | None:
        if index not in self.slices:
            return None
        pair = self.shown_pair(index)
        result = self._qa.results[(index, pair)]
        if getattr(result, "recipe", None) == RECIPE_PHYSICAL:
            # No per-slice verdict: the head-CT model decides per study; a falx
            # slice gets the model's hint, in words.
            score = slice_score(self.verdict, index) if pair == FALX else None
            text, tooltip = slice_hint(score)
            if score is None:
                text = "The septum is not used for the verdict." if pair == SEPTUM else ""
            outlier = pair == FALX and is_outlier(self.verdict, index)
            tone, note = TONE_NEUTRAL, OUTLIER_TEXT if outlier else ""
        else:
            acceptable = result.predicted_label == 1
            text, tooltip = ("Acceptable" if acceptable else "Unacceptable"), ""
            tone, note = (TONE_GOOD if acceptable else TONE_BAD), f"decision {result.decision_value:+.2f}"
        artefacts = slice_artefact_notes(self.artefacts, index)
        if artefacts:
            artefact_note = f"Possible artefact on this slice: {', '.join(artefacts)}"
            note = f"{note} · {artefact_note}" if note else artefact_note
        basal = basal_ganglia_note(self.basal_ganglia, index)
        if basal:
            basal = basal[0].upper() + basal[1:]
            note = f"{note} · {basal}" if note else basal
        return SliceView(
            pair=pair,
            position=self.slices.index(index),
            count=len(self.slices),
            available_pairs=tuple(p for p in PAIRS if (index, p) in self._qa.results),
            verdict_text=text,
            verdict_tooltip=tooltip,
            verdict_tone=tone,
            note_text=note,
        )

    def slice_steps(self, index: int) -> SliceStepsView | None:
        """The shown pair's intermediate steps on slice `index` (Advanced), if kept."""
        if index not in self.slices:
            return None
        volume = self._volume() if self._volume else None
        result = self._qa.results.get((index, self.shown_pair(index)))
        if volume is None or result is None:
            return None
        return slice_steps_view(result, volume.pixel_spacing_mm)

    # ---- the Study summary tab ----

    def table_columns(self) -> list[str]:
        return STUDY_TABLE_COLUMNS_PHYSICAL if self.is_physical() else STUDY_TABLE_COLUMNS

    def table_rows(self) -> list[TableRow]:
        physical = self.is_physical()
        rows = []
        for index, pair in self.table_jobs:
            result = self._qa.results[(index, pair)]
            cells = [
                str(index + 1),
                PAIR_LABELS[pair],
                f"{result.mtf_area:.3f}",
                f"{result.nps_roi2.var_nps:.3f}",
                f"{result.ratio.ratio_lf_hf:.2f}",
            ]
            if physical:
                score = slice_score(self.verdict, index) if pair == FALX else None
                outlier = pair == FALX and is_outlier(self.verdict, index)
                warnings = (["noise outlier"] if outlier else []) + slice_artefact_notes(self.artefacts, index)
                notes = warnings + [note for note in (basal_ganglia_note(self.basal_ganglia, index),) if note]
                cells += [resembles(score) or "—", "; ".join(notes)]
                # Red only for warnings; the basal-ganglia visibility is descriptive.
                tone = TONE_BAD if warnings else TONE_NEUTRAL
                rows.append(TableRow((index, pair), cells, tone, slice_hint(score)[1]))
            else:
                acceptable = result.predicted_label == 1
                cells += [f"{result.decision_value:+.2f}", "Acceptable" if acceptable else "Unacceptable"]
                rows.append(TableRow((index, pair), cells, TONE_GOOD if acceptable else TONE_BAD))
        return rows

    def head_ct_text(self, model_available: bool) -> tuple[VerdictText, str]:
        """The physical recipe's verdict text, and the details including the
        septum note, failed jobs and the provisional-model note."""
        text = verdict_text(self.verdict, len(self._qa.results_by_pair(FALX)), model_available=model_available)
        basics = list(text.basics)
        noise_line = next((i for i, line in enumerate(basics) if line.startswith("Artefacts (")), len(basics) - 1)
        basics[noise_line + 1:noise_line + 1] = (
            artefact_basics(self.artefacts) + basal_ganglia_basics(self.basal_ganglia) + plausibility_basics(self.plausibility)
        )
        text = replace(text, basics=tuple(basics))
        details = (
            list(text.details) + artefact_details(self.artefacts) + basal_ganglia_details(self.basal_ganglia)
            + plausibility_details(self.plausibility)
        )
        if self._qa.results_by_pair(SEPTUM):
            details.append(SEPTUM_NOTE)
        failed = sorted(self._qa.errors)
        if failed:
            details.append("Failed (not used): " + ", ".join(f"{index + 1} ({PAIR_LABELS[pair]})" for index, pair in failed))
        details.append(PROVISIONAL_NOTE)
        return text, "\n".join(details)

    def legacy_verdict(self) -> tuple[list[str], list[str]]:
        """Legacy recipe: one majority-vote line per pair (chest-CT SVM), with tones."""
        lines, tones = [], []
        for pair in PAIRS:
            verdict = study_verdict(self._qa.results_by_pair(pair).values())
            if verdict is None:
                continue
            failed = sum(1 for _index, p in self._qa.errors if p == pair)
            failed_text = f" ({failed} failed, not shown)" if failed else ""
            lines.append(
                f"{PAIR_LABELS[pair]}: {verdict.verdict} — "
                f"{verdict.acceptable} of {verdict.total} slice(s) acceptable{failed_text}"
            )
            tones.append({ACCEPTABLE: TONE_GOOD, UNACCEPTABLE: TONE_BAD}.get(verdict.verdict, TONE_NEUTRAL))
        return lines, tones
