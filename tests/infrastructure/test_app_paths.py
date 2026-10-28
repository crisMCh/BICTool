"""Where shipped files are found: next to BICTool.exe when packaged, in the repository from source."""

import sys

from bictool.infrastructure import app_paths


def test_from_source_the_manuals_are_the_built_ones_in_the_repository():
    assert not app_paths.is_packaged()
    assert app_paths.manual_path("en").parent == app_paths.app_folder() / "docs" / "manual" / "build"
    assert app_paths.manual_path("de").name == "BICTool_Benutzerhandbuch_DE.pdf"
    assert app_paths.notices_path().name == "THIRD_PARTY_NOTICES.md"
    assert app_paths.help_folder() == app_paths.app_folder() / "docs" / "manual" / "build" / "help"
    assert app_paths.technical_reference_path().parent == app_paths.manual_path("en").parent


def test_packaged_the_files_sit_next_to_the_exe(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "BICTool.exe"))

    assert app_paths.app_folder() == tmp_path
    assert app_paths.manual_path("en") == tmp_path / "manual" / "BICTool_User_Manual_EN.pdf"
    assert app_paths.notices_path() == tmp_path / "THIRD_PARTY_NOTICES.txt"
    assert app_paths.help_folder() == tmp_path / "help"
    assert app_paths.technical_reference_path() == tmp_path / "manual" / "BICTool_Technical_Reference_EN.pdf"
