"""SatCLIP source cache: a failed fetch leaves nothing to reuse."""
import os
import subprocess

import pytest

from georvq.model import location


def test_failed_checkout_leaves_no_unpinned_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("GEORVQ_SATCLIP_DIR", raising=False)
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path))

    def fake_run(cmd, check):
        if cmd[1] == "clone":
            os.makedirs(os.path.join(cmd[-1], "satclip"))
            return
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(location.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError):
        location._satclip_src()
    parent = tmp_path / ".cache" / "georvq"
    assert not (parent / "satclip").exists()
    assert list(parent.iterdir()) == []


def test_pinned_clone_is_moved_into_place(tmp_path, monkeypatch):
    monkeypatch.delenv("GEORVQ_SATCLIP_DIR", raising=False)
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path))
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd[1] if cmd[1] == "clone" else cmd[3])
        if cmd[1] == "clone":
            os.makedirs(os.path.join(cmd[-1], "satclip"))

    monkeypatch.setattr(location.subprocess, "run", fake_run)
    src = location._satclip_src()
    assert src == str(tmp_path / ".cache" / "georvq" / "satclip" / "satclip")
    assert os.path.isdir(src) and calls == ["clone", "checkout"]
