"""Reading the help window's index (built by docs/manual/build_manual.py)."""

import json

from bictool.integration.help_files import load_help_documents


def test_reads_documents_and_their_sections(tmp_path):
    (tmp_path / "help_index.json").write_text(json.dumps({"documents": [{
        "key": "user_guide", "language": "de", "title": "Benutzerhandbuch", "file": "user_guide_de.html",
        "sections": [{"id": "export", "title": "4.7 Export", "level": 3, "text": "Ergebnisse speichern"}],
    }]}), encoding="utf-8")

    [document] = load_help_documents(tmp_path, "help_index.json")

    assert (document.key, document.language, document.title) == ("user_guide", "de", "Benutzerhandbuch")
    assert document.path == tmp_path / "user_guide_de.html"
    assert document.sections[0].id == "export" and document.sections[0].level == 3


def test_a_missing_or_broken_index_gives_no_documents(tmp_path):
    assert load_help_documents(tmp_path, "help_index.json") == []
    (tmp_path / "help_index.json").write_text("{not json", encoding="utf-8")
    assert load_help_documents(tmp_path, "help_index.json") == []
    (tmp_path / "help_index.json").write_text(json.dumps({"documents": [{"key": "x"}]}), encoding="utf-8")
    assert load_help_documents(tmp_path, "help_index.json") == []
