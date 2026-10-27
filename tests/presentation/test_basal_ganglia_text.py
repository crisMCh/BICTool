"""Tests `viewmodels.basal_ganglia_text`."""

from bictool.domain.metrics.basal_ganglia import BasalGangliaSlice
from bictool.presentation.viewmodels.basal_ganglia_text import (
    BASAL_GANGLIA_NOTE,
    basal_ganglia_basics,
    basal_ganglia_details,
    basal_ganglia_note,
)


def _s(border, plausible=True):
    return BasalGangliaSlice(border=border, differences={"Lentiform Nucleus": 6.0}, plausible=plausible)


def test_visible_slices_with_the_clearly_visible_ones():
    slices = {15: _s(0.4), 16: _s(0.7), 17: _s(0.9), 18: _s(0.95), 19: _s(0.1), 22: _s(0.0, plausible=False)}
    basics = basal_ganglia_basics(slices)
    assert basics[0] == "Basal ganglia (provisional): visible on slices 17–19 (clearly on slices 18–19)."
    assert basics[1] == "Basal ganglia: CNN label without matching image contrast on slice 23 - not counted."
    details = basal_ganglia_details(slices)
    assert "16: 0.40 (hardly visible)" in details[0] and details[1] == BASAL_GANGLIA_NOTE


def test_nothing_visible_says_where_it_is_hardly_seen():
    assert basal_ganglia_basics({4: _s(0.3), 5: _s(0.1)}) == [
        "Basal ganglia (provisional): not clearly visible on any slice (hardly on slice 5)."
    ]
    assert basal_ganglia_basics({}) == [] and basal_ganglia_basics(None) == []


def test_slice_note():
    slices = {4: _s(0.7)}
    assert basal_ganglia_note(slices, 4) == "basal ganglia visible"
    assert basal_ganglia_note(slices, 5) == "" and basal_ganglia_note(None, 4) == ""
