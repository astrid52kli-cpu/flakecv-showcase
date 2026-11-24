from flakecv.config import allow_writes


def test_allow_writes(tmp_path, monkeypatch):
    monkeypatch.delenv("FLAKECV_READ_ONLY", raising=False)
    assert allow_writes(start=tmp_path) is True                     # no config: allowed
    (tmp_path / "flakecv.toml").write_text('[app]\nallow_writes = false\n')
    assert allow_writes(start=tmp_path) is False
    monkeypatch.setenv("FLAKECV_READ_ONLY", "0")
    assert allow_writes(start=tmp_path) is True
    monkeypatch.setenv("FLAKECV_READ_ONLY", "1")
    (tmp_path / "flakecv.toml").write_text('[app]\nallow_writes = true\n')
    assert allow_writes(start=tmp_path) is False
