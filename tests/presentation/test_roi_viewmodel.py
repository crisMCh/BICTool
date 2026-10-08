"""Tests `RoiViewModel` without widgets - synthetic label maps only."""

from types import SimpleNamespace

import numpy as np

from bictool.domain.roi_selection.structured_roi import SEPTUM_PELLUCIDUM
from bictool.presentation.viewmodels.roi_selection import RoiRect
from bictool.presentation.viewmodels.roi_viewmodel import RoiViewModel

SHAPE = (160, 160)


def _septum_masks():
    ventricle = np.zeros(SHAPE, dtype=bool)
    ventricle[40:120, 40:120] = True
    septum = np.zeros(SHAPE, dtype=bool)
    septum[45:115, 79:81] = True
    ventricle &= ~septum
    parenchyma = np.zeros(SHAPE, dtype=bool)
    parenchyma[10:150, 10:150] = True
    parenchyma &= ~(ventricle | septum)
    return {"Brain Parenchyma": parenchyma, "Ventricular System": ventricle, SEPTUM_PELLUCIDUM: septum}


def _vm(segmentation=None):
    image = np.full(SHAPE, 30.0)
    volume = SimpleNamespace(num_slices=1, pixel_spacing_mm=0.5, get_slice_hu=lambda i: image)
    seg = segmentation if segmentation is not None else {}
    vm = RoiViewModel(lambda job: None, lambda: volume, lambda: 0, lambda: seg, lambda: {}, lambda: None)
    changed = []
    vm.rois_changed.connect(lambda: changed.append(True))
    return vm, changed


def test_auto_rois_place_the_septum_pair_and_can_be_removed():
    vm, changed = _vm({0: SimpleNamespace(masks=_septum_masks())})

    vm.set_auto(True)
    rois = vm.effective(0)
    assert {"septum.roi1", "septum.roi2"} <= set(rois)  # ventricle ROI2 needs no WM/GM job
    assert vm.ready_jobs() == [(0, "septum")] and changed

    vm.set_auto(False)
    assert vm.effective(0) == {}


def test_manual_edits_delete_reassign_and_restore():
    vm, _changed = _vm()
    rect = RoiRect(10, 10, 30, 30)

    vm.save_manual(0, "falx.roi1", rect)
    assert vm.effective(0) == {"falx.roi1": rect} and vm.has_edits(0)
    vm.reassign(0, "falx.roi1", "septum.roi1")
    assert vm.effective(0) == {"septum.roi1": rect}
    vm.delete(0, "septum.roi1")
    assert vm.effective(0) == {}
    vm.restore_slice(0)
    assert vm.effective(0) == {} and not vm.has_edits(0)


def test_confirmed_rois_can_not_be_edited():
    vm, _changed = _vm()
    rect = RoiRect(10, 10, 30, 30)
    vm.save_manual(0, "falx.roi1", rect)
    confirmed = []
    vm.confirmed_changed.connect(confirmed.append)

    vm.set_confirmed(True)
    vm.delete(0, "falx.roi1")
    vm.save_manual(0, "falx.roi2", rect)
    vm.reassign(0, "falx.roi1", "septum.roi1")

    assert vm.effective(0) == {"falx.roi1": rect} and confirmed == [True]


def test_reset_forgets_everything():
    vm, _changed = _vm({0: SimpleNamespace(masks=_septum_masks())})
    vm.set_auto(True)
    vm.set_confirmed(True)

    vm.reset()

    assert vm.effective(0) == {} and not vm.auto_on and not vm.confirmed and vm.status_text == ""
