"""Contracts for the published gender checkpoint."""

from pathlib import Path
from unittest.mock import patch

import pytest

from naampy._resources import HF_REPO, HF_REVISION, resolve_model


def test_local_override_avoids_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "gender_lstm.pt"
    model.write_bytes(b"weights")
    monkeypatch.setenv("NAAMPY_MODEL_DIR", str(tmp_path))

    with patch("huggingface_hub.hf_hub_download") as download:
        assert resolve_model(model.name) == str(model)
    download.assert_not_called()


def test_missing_model_uses_the_exact_pinned_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NAAMPY_MODEL_DIR", str(tmp_path))
    with patch(
        "huggingface_hub.hf_hub_download", return_value="/cache/gender.pt"
    ) as download:
        assert resolve_model("gender_lstm.pt") == "/cache/gender.pt"
    download.assert_called_once_with(HF_REPO, "gender_lstm.pt", revision=HF_REVISION)


def test_revision_is_an_immutable_commit() -> None:
    assert len(HF_REVISION) == 40
    assert set(HF_REVISION) <= set("0123456789abcdef")


@pytest.mark.live
def test_pinned_revision_contains_the_checkpoint() -> None:
    from huggingface_hub import list_repo_files

    assert "gender_lstm.pt" in list_repo_files(HF_REPO, revision=HF_REVISION)
