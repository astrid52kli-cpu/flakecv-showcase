import pytest

from flakecv import config


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)


def test_explicit_wins(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    monkeypatch.setenv(config.ENV_VAR, str(tmp_path / "b"))
    assert config.data_root(tmp_path / "a") == (tmp_path / "a").resolve()


def test_env(tmp_path, monkeypatch):
    monkeypatch.setenv(config.ENV_VAR, str(tmp_path))
    assert config.data_root(start=tmp_path) == tmp_path.resolve()


def test_toml_relative_to_file(tmp_path):
    (tmp_path / "mydata").mkdir()
    (tmp_path / "flakecv.toml").write_text('[paths]\ndata = "mydata"\n')
    sub = tmp_path / "deep" / "er"; sub.mkdir(parents=True)
    assert config.data_root(start=sub) == (tmp_path / "mydata").resolve()


def test_missing_gives_clear_message(tmp_path):
    with pytest.raises(config.DataRootError) as e:
        config.data_root(start=tmp_path)
    msg = str(e.value)
    assert "--data" in msg and config.ENV_VAR in msg and "flakecv.toml" in msg
