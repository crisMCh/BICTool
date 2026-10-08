"""Tests the Qt-free ROI pair store."""

from bictool.presentation.viewmodels.roi_pairs import (
    FALX,
    ROI_KEYS,
    SEPTUM,
    RoiPairStore,
    key_label,
    roi_key,
)
from bictool.presentation.viewmodels.roi_selection import RoiRect

A = RoiRect(0, 0, 20, 20)
B = RoiRect(30, 30, 50, 50)
C = RoiRect(60, 60, 80, 80)


def test_keys_and_labels():
    assert ROI_KEYS == ("falx.roi1", "falx.roi2", "septum.roi1", "septum.roi2")
    assert key_label("septum.roi2") == "Septum ROI2"


def test_manual_roi_wins_over_the_automatic_one():
    store = RoiPairStore()
    store.set_auto(3, "falx.roi1", A)
    store.set_manual(3, "falx.roi1", B)

    assert store.effective(3)["falx.roi1"] == B
    assert store.auto(3, "falx.roi1") == A  # kept for "Restore automatic"


def test_delete_removes_manual_and_automatic_and_restore_brings_back_the_automatic():
    store = RoiPairStore()
    store.set_auto(3, "falx.roi1", A)
    store.set_manual(3, "falx.roi1", B)

    store.delete(3, "falx.roi1")
    assert "falx.roi1" not in store.effective(3)
    assert store.has_edits(3)

    store.restore_slice(3)
    assert store.effective(3)["falx.roi1"] == A
    assert not store.has_edits(3)


def test_saving_into_a_deleted_slot_brings_it_back():
    store = RoiPairStore()
    store.set_auto(3, "septum.roi2", A)
    store.delete(3, "septum.roi2")

    store.set_manual(3, "septum.roi2", C)

    assert store.effective(3)["septum.roi2"] == C


def test_complete_pairs_and_qa_jobs():
    store = RoiPairStore()
    for key in ROI_KEYS:
        store.set_auto(1, key, A)
    store.set_auto(2, roi_key(FALX, "roi1"), A)  # falx ROI2 missing -> incomplete
    store.set_manual(4, "septum.roi1", B)
    store.set_manual(4, "septum.roi2", C)

    assert store.complete_pairs(1) == [FALX, SEPTUM]
    assert store.complete_pairs(2) == []
    assert store.ready_jobs() == [(1, FALX), (1, SEPTUM), (4, SEPTUM)]


def test_clear_auto_of_one_pair_keeps_the_other_and_manual_rois():
    store = RoiPairStore()
    for key in ROI_KEYS:
        store.set_auto(1, key, A)
    store.set_manual(1, "falx.roi2", B)

    store.clear_auto(FALX)

    assert set(store.effective(1)) == {"falx.roi2", "septum.roi1", "septum.roi2"}
    assert store.effective(1)["falx.roi2"] == B
