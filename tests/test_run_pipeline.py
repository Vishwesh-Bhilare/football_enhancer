import json
import subprocess

import pytest

import run_pipeline


def test_validate_selection_rejects_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(run_pipeline, "SELECTION_PATH", tmp_path / "selection.json")

    with pytest.raises(RuntimeError, match="was not created"):
        run_pipeline.validate_selection()


def test_validate_selection_accepts_selected_ids(monkeypatch, tmp_path):
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({"selected_ids": [1, 3]}))
    monkeypatch.setattr(run_pipeline, "SELECTION_PATH", selection_path)

    run_pipeline.validate_selection()


def test_run_command_reports_child_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(7, args[0])

    monkeypatch.setattr(run_pipeline.subprocess, "run", fail)

    with pytest.raises(RuntimeError, match="Render failed with exit code 7"):
        run_pipeline.run_command(["renderer"], "Render")
