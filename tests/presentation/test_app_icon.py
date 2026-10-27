"""The app icon is shipped with the package and loads in all sizes."""

from PySide6.QtCore import QSize

from bictool.presentation.views.app_icon import ICON_SIZES, RESOURCES, app_icon, register_windows_app_id


def test_all_icon_sizes_are_shipped():
    for size in ICON_SIZES:
        assert (RESOURCES / f"app_icon_{size}.png").is_file()


def test_the_icon_loads_with_small_and_large_sizes():
    icon = app_icon()

    assert not icon.isNull()
    sizes = {(s.width(), s.height()) for s in icon.availableSizes()}
    assert (16, 16) in sizes and (256, 256) in sizes
    assert not icon.pixmap(QSize(32, 32)).isNull()


def test_registering_the_windows_app_id_never_fails():
    register_windows_app_id()  # no-op off Windows; must not raise on Windows
